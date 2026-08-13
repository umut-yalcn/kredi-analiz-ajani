"""Agent'in kullanabilecegi analiz araclari.

Her arac guard katmanindan gecer. Tasarim ilkesi: veri, analiz katmanina zaten
PII'siz giriyor - guard'i atlatan bir kod yolu bile kisisel veriye ulasamaz.
Guard, o ilk savunmanin uzerine gelen ikinci ve ucuncu katman.
"""

from __future__ import annotations

import contextvars
import json
from functools import lru_cache
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import tool

from .config import DATA_PATH
from .guard import K_ANONYMITY_THRESHOLD, Guard, GuardViolation
from .schema import (
    ANALYZABLE_COLUMNS,
    BY_NAME,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PII_COLUMNS,
)

# Istek boyunca yasayan guard. Modul seviyesinde tek bir global kullanmak yerine
# ContextVar kullaniyoruz: FastAPI istekleri es zamanli calisir ve global bir
# degisken kullanilsaydi ikinci istek birincinin guard'ini ezerdi. O durumda
# denetim kaydi yanlis istege yazilir - uyumluluk kaniti olarak sunulan bir
# yapida bu kabul edilemez. ContextVar her istege kendi kopyasini verir.
_guard_var: contextvars.ContextVar[Guard] = contextvars.ContextVar("aktif_guard")


def set_guard(guard: Guard) -> None:
    _guard_var.set(guard)


def get_guard() -> Guard:
    """Bu baglama ait guard'i dondurur; yoksa yenisini olusturur."""
    try:
        return _guard_var.get()
    except LookupError:
        guard = Guard()
        _guard_var.set(guard)
        return guard


@lru_cache(maxsize=1)
def load_analysis_frame() -> pd.DataFrame:
    """Veriyi PII kolonlari dusurulmus halde yukler.

    Bu, mimarinin en onemli satiri: PII hicbir zaman analiz katmaninin
    bellegine girmez. Agent yanlis bir sorgu uretse bile ortada sizdiracak
    veri yoktur.
    """
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


@tool
def list_columns() -> str:
    """Analize acik kolonlarin listesini, tiplerini ve aciklamalarini dondurur.

    Analize baslarken hangi kolonlarin var oldugunu ogrenmek icin bunu cagir.
    """
    return _ok(
        {
            "kolonlar": [
                {
                    "ad": name,
                    "tip": BY_NAME[name].dtype,
                    "aciklama": BY_NAME[name].description,
                    "birim": BY_NAME[name].unit,
                }
                for name in ANALYZABLE_COLUMNS
            ],
            "not": (
                "Kisisel veri iceren kolonlar (ad, kimlik no, telefon, e-posta) "
                "bu listede yer almaz ve sorgulanamaz."
            ),
        }
    )


@tool
def describe_column(column: str) -> str:
    """Tek bir kolonun ozet istatistiklerini dondurur.

    Sayisal kolonlarda ortalama/medyan/dagilim, kategorik kolonlarda deger
    dagilimi doner.

    Args:
        column: Incelenecek kolonun adi.
    """
    try:
        get_guard().check_columns("describe_column", [column])
    except GuardViolation as exc:
        return _ok({"hata": str(exc)})

    df = load_analysis_frame()
    series = df[column]

    if column in NUMERIC_COLUMNS:
        gozlem = series.dropna()
        q = gozlem.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

        # Uc degerler k-anonimlige tabidir. Bir kolonun en yuksek degeri tek bir
        # kisiye aitse, o degeri bildirmek o kisiyi isaret eder - toplulastirma
        # gorunumu altinda tekil aciklama olur. Uc degeri yalnizca en az k kisi
        # ayni degeri paylasiyorsa veriyoruz; aksi halde q05/q95 ile yetiniyoruz.
        # Ayrik kolonlarda (vade_ay, aktif_kredi_sayisi) uc degerler genellikle
        # esigi gecer ve normal sekilde raporlanir.
        ozet: dict[str, Any] = {
            "kolon": column,
            "satir_sayisi": int(gozlem.count()),
            "ortalama": round(float(gozlem.mean()), 2),
            "std_sapma": round(float(gozlem.std()), 2),
            "q05": round(float(q.loc[0.05]), 2),
            "q25": round(float(q.loc[0.25]), 2),
            "medyan": round(float(q.loc[0.5]), 2),
            "q75": round(float(q.loc[0.75]), 2),
            "q95": round(float(q.loc[0.95]), 2),
        }

        bastirilan = []
        for etiket, deger in (("min", gozlem.min()), ("max", gozlem.max())):
            paylasan = int((gozlem == deger).sum())
            if paylasan >= K_ANONYMITY_THRESHOLD:
                ozet[etiket] = round(float(deger), 2)
            else:
                bastirilan.append(etiket)
                get_guard().note(
                    "describe_column",
                    (column,),
                    f"{etiket} degeri {paylasan} kisiye ait, k<{K_ANONYMITY_THRESHOLD} - bastirildi",
                )

        if bastirilan:
            ozet["bastirilan_uc_degerler"] = bastirilan
            ozet["not"] = (
                f"{', '.join(bastirilan)} degeri {K_ANONYMITY_THRESHOLD} kisiden "
                "azina ait oldugu icin bastirildi. Dagilimin ucu icin q05/q95 kullan."
            )

        return _ok(ozet)

    counts = series.value_counts()
    suppressed = get_guard().check_group_sizes("describe_column", counts.to_dict())
    counts = counts.drop(index=suppressed, errors="ignore")
    return _ok(
        {
            "kolon": column,
            "satir_sayisi": int(series.count()),
            "dagilim": {str(k): int(v) for k, v in counts.items()},
            "bastirilan_grup_sayisi": len(suppressed),
        }
    )


@tool
def group_aggregate(
    group_by: str,
    metric: str,
    how: Literal["mean", "median", "sum", "count", "rate"] = "mean",
) -> str:
    """Bir kolona gore gruplayip baska bir kolonu toplulastirir.

    Ornek: meslek gruplarina gore ortalama kredi skoru, ya da illere gore
    temerrut orani.

    k-anonimlik kurali geregi 20 satirdan az iceren gruplar sonuctan cikarilir.

    Args:
        group_by: Gruplama yapilacak kategorik kolon.
        metric: Toplulastirilacak kolon.
        how: Toplulastirma yontemi. 'rate' 0/1 kolonlar icin oran hesaplar.
    """
    try:
        get_guard().check_columns("group_aggregate", [group_by, metric])
    except GuardViolation as exc:
        return _ok({"hata": str(exc)})

    if group_by not in CATEGORICAL_COLUMNS:
        return _ok(
            {
                "hata": f"'{group_by}' gruplama icin uygun degil. "
                f"Kategorik kolonlar: {', '.join(CATEGORICAL_COLUMNS)}"
            }
        )

    df = load_analysis_frame()
    grouped = df.groupby(group_by)[metric]

    # k esigi, metrigin gercekten gozlemlendigi satir sayisina uygulanir -
    # gruptaki toplam satira degil. Aksi halde metrigin cogunlukla bos oldugu
    # bir grup, esigi gecmis gibi gorunurdu.
    sizes = grouped.count().to_dict()
    suppressed = get_guard().check_group_sizes("group_aggregate", sizes)

    result = grouped.mean() if how == "rate" else getattr(grouped, how)()
    result = result.drop(index=suppressed, errors="ignore").dropna()

    values = {
        str(k): round(float(v), 4 if how == "rate" else 2)
        for k, v in result.sort_values(ascending=False).items()
    }
    return _ok(
        {
            "gruplama": group_by,
            "metrik": metric,
            "yontem": how,
            "sonuc": values,
            "gozlemlenen_satir_sayisi": {
                str(k): int(v) for k, v in sizes.items() if k not in suppressed
            },
            "bastirilan_grup_sayisi": len(suppressed),
        }
    )


@tool
def segment_stats(column: str, operator: Literal["<", "<=", ">", ">=", "=="], value: float,
                  metric: str) -> str:
    """Bir kosula uyan alt kumede bir metrigin ozetini dondurur.

    Ornek: kredi skoru 1000'in altinda olanlarda ortalama temerrut orani.

    Sonuc kumesi 20 satirdan azsa istek reddedilir - tek bir kisiye
    indirgenebilecek sorgular calistirilamaz.

    Args:
        column: Filtre uygulanacak kolon.
        operator: Karsilastirma operatoru.
        value: Karsilastirma degeri.
        metric: Ozeti alinacak sayisal kolon.
    """
    try:
        get_guard().check_columns("segment_stats", [column, metric])
    except GuardViolation as exc:
        return _ok({"hata": str(exc)})

    df = load_analysis_frame()
    ops = {
        "<": df[column] < value,
        "<=": df[column] <= value,
        ">": df[column] > value,
        ">=": df[column] >= value,
        "==": df[column] == value,
    }
    subset = df[ops[operator]]

    if subset.empty:
        return _ok({"uyari": "Bu kosula uyan satir yok.", "satir_sayisi": 0})

    # k esigi ONCE alt kumenin kendisine uygulanir. Aksi halde metrigin hic
    # gozlemlenmedigi bir segmentte fonksiyon erken donuyor ve alt kume boyutu
    # ("bu kosula 1 kisi uyuyor") k kontrolunden gecmeden disari cikiyordu.
    # Filtrenin kac kisiyi sectigi de basli basina bir bilgidir.
    try:
        get_guard().check_row_count("segment_stats", len(subset))
    except GuardViolation as exc:
        return _ok({"hata": str(exc)})

    # Metrik bazi satirlarda gozlemlenmemis olabilir (orn. reddedilen basvuruda
    # temerrut). k-anonimlik esigi gozlemlenen satir sayisina uygulanir.
    observed = subset[metric].dropna()

    try:
        get_guard().check_row_count("segment_stats", len(observed))
    except GuardViolation as exc:
        return _ok({"hata": str(exc)})

    if observed.empty:
        return _ok(
            {
                "kosul": f"{column} {operator} {value}",
                "satir_sayisi": len(subset),
                "uyari": (
                    f"Bu segmentte '{metric}' hicbir satirda gozlemlenmemis. "
                    "Ornegin reddedilen basvurularda temerrut olcumu yoktur - "
                    "bu segmentin gercek riski veriden okunamaz."
                ),
                "gozlemlenen_satir": 0,
            }
        )

    return _ok(
        {
            "kosul": f"{column} {operator} {value}",
            "satir_sayisi": len(subset),
            "gozlemlenen_satir": len(observed),
            "toplam_icindeki_pay": round(len(subset) / len(df), 4),
            "metrik": metric,
            "ortalama": round(float(observed.mean()), 4),
            "medyan": round(float(observed.median()), 4),
            "genel_ortalama": round(float(df[metric].mean()), 4),
        }
    )


@tool
def correlation(column_a: str, column_b: str) -> str:
    """Iki sayisal kolon arasindaki Pearson korelasyonunu hesaplar.

    Args:
        column_a: Birinci sayisal kolon.
        column_b: Ikinci sayisal kolon.
    """
    try:
        get_guard().check_columns("correlation", [column_a, column_b])
    except GuardViolation as exc:
        return _ok({"hata": str(exc)})

    for col in (column_a, column_b):
        if col not in NUMERIC_COLUMNS:
            return _ok(
                {
                    "hata": f"'{col}' sayisal bir kolon degil. "
                    f"Sayisal kolonlar: {', '.join(NUMERIC_COLUMNS)}"
                }
            )

    df = load_analysis_frame()

    # Korelasyon yalnizca IKI kolonun da gozlemlendigi satirlar uzerinden
    # hesaplanir. Onceden len(df) bildiriliyordu; temerrut gibi reddedilen
    # basvurularda bos kalan bir kolonda bu, kullanilmayan binlerce satiri
    # sayiyordu. Reject inference'i dogru kuran bir projede burada da ayni
    # titizlik gerekir.
    cift = df[[column_a, column_b]].dropna()
    if len(cift) < K_ANONYMITY_THRESHOLD:
        return _ok(
            {
                "hata": f"Bu iki kolonun birlikte gozlemlendigi yalnizca {len(cift)} "
                f"satir var. En az {K_ANONYMITY_THRESHOLD} gerekiyor."
            }
        )

    r = float(cift[column_a].corr(cift[column_b]))
    if abs(r) < 0.1:
        yorum = "ihmal edilebilir"
    elif abs(r) < 0.3:
        yorum = "zayif"
    elif abs(r) < 0.5:
        yorum = "orta"
    else:
        yorum = "guclu"

    return _ok(
        {
            "kolonlar": [column_a, column_b],
            "pearson_r": round(r, 4),
            "yon": "pozitif" if r > 0 else "negatif",
            "guc": yorum,
            "kullanilan_satir_sayisi": len(cift),
            "veri_setindeki_satir_sayisi": len(df),
            "dusen_satir_sayisi": len(df) - len(cift),
        }
    )


ANALYSIS_TOOLS = [
    list_columns,
    describe_column,
    group_aggregate,
    segment_stats,
    correlation,
]
