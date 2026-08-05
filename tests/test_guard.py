"""Guard katmani testleri.

Bu testler API anahtari gerektirmez ve model cagrisi yapmaz - dogrudan guvenlik
katmanini sinar. `pytest` calistirmak, korumalarin gercekten devrede oldugunu
dogrulamak icin yeterlidir.
"""

from __future__ import annotations

import pytest

from src.guard import Guard, GuardViolation, K_ANONYMITY_THRESHOLD
from src.schema import ANALYZABLE_COLUMNS, PII_COLUMNS


class TestKolonIzni:
    def test_analize_acik_kolon_gecer(self):
        guard = Guard()
        guard.check_columns("test", ["kredi_skoru", "temerrut"])
        assert guard.audit_log[-1].allowed is True

    @pytest.mark.parametrize("pii_column", PII_COLUMNS)
    def test_her_pii_kolonu_reddedilir(self, pii_column: str):
        guard = Guard()
        with pytest.raises(GuardViolation, match="kisisel veri"):
            guard.check_columns("test", [pii_column])
        assert guard.audit_log[-1].allowed is False

    def test_pii_temiz_kolonla_karistirilirsa_yine_reddedilir(self):
        """Agent PII'yi masum bir kolonun yanina saklayarak gecemez."""
        guard = Guard()
        with pytest.raises(GuardViolation):
            guard.check_columns("test", ["kredi_skoru", "tckn"])

    def test_uydurma_kolon_reddedilir(self):
        guard = Guard()
        with pytest.raises(GuardViolation, match="veri sozlugunde yok"):
            guard.check_columns("test", ["gizli_musteri_notu"])

    def test_pii_kolonlar_analiz_listesinde_yok(self):
        assert not set(PII_COLUMNS) & set(ANALYZABLE_COLUMNS)


class TestKAnonimlik:
    def test_kucuk_gruplar_bastirilir(self):
        guard = Guard()
        counts = {"buyuk": 500, "orta": K_ANONYMITY_THRESHOLD, "kucuk": 3}
        suppressed = guard.check_group_sizes("test", counts)
        assert suppressed == ["kucuk"]

    def test_esik_degeri_gecer(self):
        guard = Guard()
        assert guard.check_group_sizes("test", {"tam_esik": K_ANONYMITY_THRESHOLD}) == []

    def test_dar_sonuc_kumesi_reddedilir(self):
        guard = Guard()
        with pytest.raises(GuardViolation, match="tek bir kisiye"):
            guard.check_row_count("test", K_ANONYMITY_THRESHOLD - 1)

    def test_bos_sonuc_reddedilmez(self):
        """Sifir satir kimseyi ifsa etmez; bu bir gizlilik ihlali degil."""
        Guard().check_row_count("test", 0)


class TestPIIMaskeleme:
    def test_tckn_maskelenir(self):
        out = Guard().mask("Musterinin kimlik numarasi 12345678901 olarak gecti.")
        assert "12345678901" not in out
        assert "[TCKN_MASKELENDI]" in out

    def test_telefon_maskelenir(self):
        out = Guard().mask("Iletisim: 05321234567")
        assert "05321234567" not in out
        assert "[TELEFON_MASKELENDI]" in out

    def test_email_maskelenir(self):
        out = Guard().mask("Adres: musteri@ornek.com.tr")
        assert "musteri@ornek.com.tr" not in out

    def test_normal_sayilar_bozulmaz(self):
        """Kredi skoru, tutar, yil gibi degerler maskelenmemeli."""
        text = "Ortalama kredi skoru 1403, ortalama tutar 250000 TL, yil 2026."
        assert Guard().mask(text) == text

    def test_maskeleme_denetime_yazilir(self):
        guard = Guard()
        guard.mask("12345678901 ve 05321234567")
        assert len(guard.audit_log) == 2


class TestDenetimKaydi:
    def test_izin_ve_ret_birlikte_kaydedilir(self):
        guard = Guard()
        guard.check_columns("izinli", ["kredi_skoru"])
        with pytest.raises(GuardViolation):
            guard.check_columns("reddedilen", ["tckn"])

        trail = guard.audit_trail()
        assert [e["allowed"] for e in trail] == [True, False]
        assert all(e["timestamp"] and e["reason"] for e in trail)
