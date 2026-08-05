"""Veri sozlugu: her kolonun ne oldugu, hassasiyet sinifi ve analiz izni.

Bu dosya sistemin tek gercek kaynagidir (single source of truth). Guard katmani
erisim kararlarini, katalog katmani ise vektor aramasini buradan uretir.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Sensitivity(str, Enum):
    """Kolonun hassasiyet sinifi."""

    PII = "pii"  # Kisiyi dogrudan tanimlar - agent asla goremez
    QUASI = "quasi"  # Tek basina tanimlamaz, birlestirilince riskli
    PUBLIC = "public"  # Analize acik


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    description: str
    sensitivity: Sensitivity
    unit: str | None = None

    @property
    def analyzable(self) -> bool:
        """PII kolonlar hicbir kosulda analiz katmanina gecmez."""
        return self.sensitivity is not Sensitivity.PII


CREDIT_APPLICATION_SCHEMA: tuple[Column, ...] = (
    Column(
        "basvuru_id",
        "string",
        "Kredi basvurusunun benzersiz kimligi. Analizde gruplama anahtari olarak kullanilmaz.",
        Sensitivity.PII,
    ),
    Column(
        "ad_soyad",
        "string",
        "Basvuru sahibinin adi ve soyadi.",
        Sensitivity.PII,
    ),
    Column(
        "tckn",
        "string",
        "Basvuru sahibinin T.C. kimlik numarasi.",
        Sensitivity.PII,
    ),
    Column(
        "telefon",
        "string",
        "Basvuru sahibinin cep telefonu numarasi.",
        Sensitivity.PII,
    ),
    Column(
        "email",
        "string",
        "Basvuru sahibinin e-posta adresi.",
        Sensitivity.PII,
    ),
    Column(
        "yas",
        "int",
        "Basvuru sahibinin yasi. Yas gruplarina gore risk analizinde kullanilir.",
        Sensitivity.QUASI,
        unit="yil",
    ),
    Column(
        "il",
        "string",
        "Basvuru sahibinin ikamet ettigi il. Bolgesel risk dagilimi analizinde kullanilir.",
        Sensitivity.QUASI,
    ),
    Column(
        "meslek_grubu",
        "string",
        "Meslek kategorisi: kamu, ozel_sektor, serbest, emekli, ogrenci, issiz.",
        Sensitivity.QUASI,
    ),
    Column(
        "aylik_gelir",
        "float",
        "Beyan edilen aylik net gelir. Odeme gucu ve borc/gelir orani hesabinda kullanilir.",
        Sensitivity.PUBLIC,
        unit="TL",
    ),
    Column(
        "talep_edilen_tutar",
        "float",
        "Basvuruda talep edilen kredi tutari.",
        Sensitivity.PUBLIC,
        unit="TL",
    ),
    Column(
        "vade_ay",
        "int",
        "Talep edilen kredinin vadesi.",
        Sensitivity.PUBLIC,
        unit="ay",
    ),
    Column(
        "kredi_skoru",
        "int",
        "Kredi burosu risk skoru, 0-1900 araligi. Yuksek skor dusuk risk anlamina gelir.",
        Sensitivity.PUBLIC,
    ),
    Column(
        "mevcut_borc",
        "float",
        "Basvuru anindaki toplam mevcut borc bakiyesi.",
        Sensitivity.PUBLIC,
        unit="TL",
    ),
    Column(
        "aktif_kredi_sayisi",
        "int",
        "Basvuru aninda devam eden aktif kredi adedi.",
        Sensitivity.PUBLIC,
    ),
    Column(
        "gecikme_gun_max",
        "int",
        "Son 24 ayda yasanan en uzun odeme gecikmesi. 0 gecikme yok demektir.",
        Sensitivity.PUBLIC,
        unit="gun",
    ),
    Column(
        "basvuru_sonucu",
        "string",
        "Basvurunun sonucu: onay veya red.",
        Sensitivity.PUBLIC,
    ),
    Column(
        "temerrut",
        "float",
        "Kredi kullandirildiktan sonra temerruda dusup dusmedigi. 1 temerrut, 0 saglikli odeme. "
        "Reddedilen basvurularda kredi kullandirilmadigi icin bu deger BOSTUR, sifir degildir. "
        "Bu nedenle temerrut analizleri yalnizca onaylanan basvurular uzerinden yapilir; "
        "reddedilenlerin gercek riski gozlemlenemez.",
        Sensitivity.PUBLIC,
    ),
)

BY_NAME: dict[str, Column] = {c.name: c for c in CREDIT_APPLICATION_SCHEMA}

ANALYZABLE_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in CREDIT_APPLICATION_SCHEMA if c.analyzable
)

PII_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in CREDIT_APPLICATION_SCHEMA if c.sensitivity is Sensitivity.PII
)

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    c.name
    for c in CREDIT_APPLICATION_SCHEMA
    if c.analyzable and c.dtype in ("int", "float")
)

CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in CREDIT_APPLICATION_SCHEMA if c.analyzable and c.dtype == "string"
)
