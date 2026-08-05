"""Guard katmanini ve analiz araclarini API anahtari olmadan gosterir.

Model cagrisi yapmaz - araclari dogrudan cagirir. Sunumda ya da repoyu ilk kez
acan birine korumanin gercek oldugunu gostermek icin:

    python scripts/demo_guard.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.guard import Guard, K_ANONYMITY_THRESHOLD  # noqa: E402
from src.tools import (  # noqa: E402
    correlation,
    describe_column,
    group_aggregate,
    segment_stats,
    set_guard,
)


def baslik(text: str) -> None:
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}")


def main() -> None:
    guard = Guard()
    set_guard(guard)

    baslik("1. ANALIZ  -  Meslek grubuna gore temerrut orani")
    r = json.loads(
        group_aggregate.invoke(
            {"group_by": "meslek_grubu", "metric": "temerrut", "how": "rate"}
        )
    )
    for grup, oran in r["sonuc"].items():
        n = r["gozlemlenen_satir_sayisi"][grup]
        print(f"   {grup:<14} {oran:>7.1%}   (gozlemlenen n={n})")

    baslik("2. ANALIZ  -  Kredi skoru ile temerrut iliskisi")
    r = json.loads(
        correlation.invoke({"column_a": "kredi_skoru", "column_b": "temerrut"})
    )
    print(f"   Pearson r : {r['pearson_r']}")
    print(f"   Yon       : {r['yon']}")
    print(f"   Guc       : {r['guc']}")

    baslik("3. ANALIZ  -  Skoru dusuk olanlarda risk")
    r = json.loads(
        segment_stats.invoke(
            {
                "column": "kredi_skoru",
                "operator": "<",
                "value": 1250,
                "metric": "temerrut",
            }
        )
    )
    print(f"   Kosul            : {r['kosul']}")
    print(f"   Satir sayisi     : {r['satir_sayisi']}")
    print(f"   Gozlemlenen      : {r['gozlemlenen_satir']}")
    print(f"   Segment orani    : {r['ortalama']:.1%}")
    print(f"   Genel ortalama   : {r['genel_ortalama']:.1%}")

    baslik("4. DURUSTLUK  -  Gozlemlenemeyen segment")
    r = json.loads(
        segment_stats.invoke(
            {
                "column": "kredi_skoru",
                "operator": "<",
                "value": 1000,
                "metric": "temerrut",
            }
        )
    )
    print(f"   Kosul        : kredi_skoru < 1000")
    print(f"   Satir sayisi : {r.get('satir_sayisi')}")
    if "uyari" in r:
        print(f"   Sonuc        : {r['uyari']}")
        print("\n   Bu segmentteki basvurularin tamami reddedilmis; kredi")
        print("   kullandirilmadigi icin temerrut hic olculmemis. Sistem")
        print("   sifir yazmak yerine olcumun yapilamadigini soyluyor.")
    else:
        print(f"   Sonuc        : {r.get('ortalama')}")

    baslik("5. GUARD  -  Kisisel veri talebi reddedilir")
    for kolon in ("tckn", "ad_soyad", "telefon"):
        r = json.loads(describe_column.invoke({"column": kolon}))
        durum = r.get("hata", ">>> REDDEDILMEDI - GUVENLIK ACIGI <<<")
        print(f"   {kolon:<12} -> {durum.split('.')[0]}.")

    baslik(f"6. GUARD  -  k-anonimlik (k={K_ANONYMITY_THRESHOLD})")
    r = json.loads(
        segment_stats.invoke(
            {
                "column": "kredi_skoru",
                "operator": "<",
                "value": 900,
                "metric": "aylik_gelir",
            }
        )
    )
    print("   Talep: kredi_skoru < 900 olanlarin geliri")
    if "hata" in r:
        print(f"   Reddedildi: {r['hata']}")
        print("\n   Bu filtre 14 satira iniyor. Bu kadar dar bir kumede")
        print("   ortalama, tek tek kisilerin gelirini ifsa edebilir.")
    else:
        print(f"   {r['satir_sayisi']} satir dondu (esigin uzerinde).")

    baslik("7. GUARD  -  Cikti maskeleme")
    ornek = "Musteri 12345678901 numarali kisi, 05321234567, ornek@mail.com"
    print(f"   Girdi : {ornek}")
    print(f"   Cikti : {guard.mask(ornek)}")

    baslik("8. DENETIM KAYDI")
    for e in guard.audit_trail():
        isaret = "IZIN " if e["allowed"] else "RED  "
        print(f"   [{isaret}] {e['action']:<16} {e['reason']}")
    print()


if __name__ == "__main__":
    main()
