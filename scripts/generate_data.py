"""Sentetik kredi basvuru veri seti uretir.

Gercek kredi verisi paylasilamaz, bu yuzden istatistiksel olarak anlamli sinyal
tasiyan sentetik bir set uretiyoruz: kredi skoru gecikme gecmisi ve borc/gelir
oraniyla iliskili, temerrut olasiligi da bunlarin bir fonksiyonu.

Set bilerek PII kolonlari (ad, TCKN, telefon, e-posta) iceriyor - guard katmaninin
bu kolonlari gercekten engelledigini gosterebilmek icin.

Kullanim:
    python scripts/generate_data.py --rows 5000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42

ADLAR = [
    "Ahmet", "Mehmet", "Ayse", "Fatma", "Mustafa", "Zeynep", "Ali", "Emine",
    "Hasan", "Elif", "Huseyin", "Meryem", "Ibrahim", "Hatice", "Omer", "Selin",
    "Burak", "Deniz", "Kemal", "Nur", "Serkan", "Ebru", "Volkan", "Pinar",
]
SOYADLAR = [
    "Yilmaz", "Kaya", "Demir", "Sahin", "Celik", "Yildiz", "Yildirim", "Ozturk",
    "Aydin", "Ozdemir", "Arslan", "Dogan", "Kilic", "Aslan", "Cetin", "Kara",
    "Koc", "Kurt", "Ozkan", "Simsek",
]
ILLER = [
    "Istanbul", "Ankara", "Izmir", "Bursa", "Antalya", "Adana", "Konya",
    "Gaziantep", "Kayseri", "Mersin", "Eskisehir", "Samsun", "Denizli", "Trabzon",
]
IL_AGIRLIK = [0.24, 0.13, 0.10, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
MESLEK_GRUPLARI = ["kamu", "ozel_sektor", "serbest", "emekli", "ogrenci", "issiz"]
MESLEK_AGIRLIK = [0.18, 0.42, 0.16, 0.14, 0.06, 0.04]

# Meslek grubuna gore aylik gelirin log-normal parametreleri
GELIR_PROFILI = {
    "kamu": (10.6, 0.35),
    "ozel_sektor": (10.5, 0.55),
    "serbest": (10.4, 0.80),
    "emekli": (10.0, 0.30),
    "ogrenci": (9.2, 0.50),
    "issiz": (9.0, 0.60),
}


def _tckn(rng: random.Random) -> str:
    """Gercek TCKN algoritmasina uymayan, format olarak benzeyen sahte numara."""
    return str(rng.randint(10_000_000_000, 99_999_999_999))


def _telefon(rng: random.Random) -> str:
    return f"05{rng.randint(300000000, 599999999)}"


def generate(n_rows: int) -> pd.DataFrame:
    rng = random.Random(RNG_SEED)
    np_rng = np.random.default_rng(RNG_SEED)

    meslek = np_rng.choice(MESLEK_GRUPLARI, size=n_rows, p=MESLEK_AGIRLIK)
    il = np_rng.choice(ILLER, size=n_rows, p=IL_AGIRLIK)
    yas = np.clip(np_rng.normal(41, 12, n_rows), 18, 78).astype(int)

    mu = np.array([GELIR_PROFILI[m][0] for m in meslek])
    sigma = np.array([GELIR_PROFILI[m][1] for m in meslek])
    aylik_gelir = np.round(np_rng.lognormal(mu, sigma), -2)
    aylik_gelir = np.clip(aylik_gelir, 17_000, 900_000)

    aktif_kredi = np_rng.poisson(1.4, n_rows)
    mevcut_borc = np.round(aylik_gelir * np_rng.gamma(2.0, 1.4, n_rows), -2)

    # Gecikme gecmisi: cogunlukla temiz, kuyrukta agir gecikmeler
    gecikme = np.where(
        np_rng.random(n_rows) < 0.68,
        0,
        np_rng.gamma(2.0, 26, n_rows).astype(int),
    )
    gecikme = np.clip(gecikme, 0, 360).astype(int)

    borc_gelir_orani = mevcut_borc / (aylik_gelir * 12)

    # Kredi skoru: gecikme ve borc yuku ile negatif, gelir ile hafif pozitif iliskili
    skor_ham = (
        1500
        - 2.6 * gecikme
        - 190 * np.clip(borc_gelir_orani, 0, 4)
        - 38 * aktif_kredi
        + 42 * np.log1p(aylik_gelir / 20_000)
        + np_rng.normal(0, 105, n_rows)
    )
    kredi_skoru = np.clip(skor_ham, 0, 1900).astype(int)

    talep = np.round(aylik_gelir * np_rng.uniform(1.5, 14.0, n_rows), -2)
    talep = np.clip(talep, 10_000, 2_500_000)
    vade = np_rng.choice([6, 12, 18, 24, 36, 48, 60], size=n_rows,
                         p=[0.08, 0.22, 0.14, 0.24, 0.18, 0.09, 0.05])

    basvuru_sonucu = np.where(
        (kredi_skoru >= 1100) & (talep <= aylik_gelir * 11) & (gecikme <= 90),
        "onay",
        "red",
    )

    # Temerrut olasiligi: lojistik model.
    logit = (
        -1.0
        - 0.0060 * (kredi_skoru - 1100)
        + 1.25 * np.clip(borc_gelir_orani, 0, 4)
        + 0.011 * gecikme
        + 0.16 * aktif_kredi
    )
    p_temerrut = 1 / (1 + np.exp(-logit))
    temerrut = (np_rng.random(n_rows) < p_temerrut).astype(float)

    # Reddedilen basvuruda kredi hic kullandirilmadigi icin temerrut GOZLEMLENMEZ.
    # Bunu 0 yazmak veriyi bozar: "reddedildi" ile "odedi" ayni sey degildir.
    # Bos birakiyoruz - kredi riskinde bu, reject inference probleminin ta kendisi.
    temerrut = np.where(basvuru_sonucu == "onay", temerrut, np.nan)

    return pd.DataFrame(
        {
            "basvuru_id": [f"BSV-{2026}{i:07d}" for i in range(1, n_rows + 1)],
            "ad_soyad": [f"{rng.choice(ADLAR)} {rng.choice(SOYADLAR)}" for _ in range(n_rows)],
            "tckn": [_tckn(rng) for _ in range(n_rows)],
            "telefon": [_telefon(rng) for _ in range(n_rows)],
            "email": [f"kullanici{i}@ornek.com.tr" for i in range(1, n_rows + 1)],
            "yas": yas,
            "il": il,
            "meslek_grubu": meslek,
            "aylik_gelir": aylik_gelir,
            "talep_edilen_tutar": talep,
            "vade_ay": vade,
            "kredi_skoru": kredi_skoru,
            "mevcut_borc": mevcut_borc,
            "aktif_kredi_sayisi": aktif_kredi,
            "gecikme_gun_max": gecikme,
            "basvuru_sonucu": basvuru_sonucu,
            "temerrut": temerrut,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentetik kredi basvuru verisi uretir")
    parser.add_argument("--rows", type=int, default=5000, help="Uretilecek satir sayisi")
    parser.add_argument("--out", default="data/kredi_basvurulari.csv", help="Cikti dosyasi")
    args = parser.parse_args()

    df = generate(args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    onay_orani = (df["basvuru_sonucu"] == "onay").mean()
    temerrut_orani = df.loc[df["basvuru_sonucu"] == "onay", "temerrut"].mean()
    print(f"[OK] {len(df)} satir yazildi -> {out}")
    print(f"     Onay orani      : {onay_orani:.1%}")
    print(f"     Temerrut orani  : {temerrut_orani:.1%} (onaylananlar icinde)")
    print(f"     Ortalama skor   : {df['kredi_skoru'].mean():.0f}")


if __name__ == "__main__":
    main()
