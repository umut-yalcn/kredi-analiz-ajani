"""Denetimde bulunan sorunlarin regresyon testleri.

Her test, gercekten yasanmis bir hatayi kilitler. Basliklar bulgunun ne
oldugunu anlatir; birisi bu davranisi geri getirirse test dusmelidir.

API anahtari gerektirmez.
"""

from __future__ import annotations

import json
import threading

import pytest

from src.config import DATA_PATH
from src.guard import Guard, K_ANONYMITY_THRESHOLD
from src.tools import (
    correlation,
    describe_column,
    get_guard,
    load_analysis_frame,
    segment_stats,
    set_guard,
)


@pytest.fixture(autouse=True)
def temiz_guard():
    set_guard(Guard())
    yield


def _c(arac, **kwargs) -> dict:
    return json.loads(arac.invoke(kwargs))


class TestSegmentKAnonimligi:
    """Bulgu 1: gozlemlenen metrik bos oldugunda alt kume boyutu sizdiriliyordu.

    'kredi_skoru < 700' sorgusu {'satir_sayisi': 1} donduruyordu - yani veride
    skoru 700'un altinda tam bir kisi oldugu bilgisi disari cikiyordu.
    """

    def test_dar_segment_metrik_gozlemlenmese_bile_reddedilir(self):
        r = _c(segment_stats, column="kredi_skoru", operator="<", value=700,
               metric="temerrut")
        assert "hata" in r
        assert "satir_sayisi" not in r

    def test_dar_segment_metrik_gozlense_de_reddedilir(self):
        r = _c(segment_stats, column="kredi_skoru", operator="<", value=700,
               metric="aylik_gelir")
        assert "hata" in r
        assert "ortalama" not in r

    def test_genis_segmentte_gozlemlenmemis_uyarisi_korunur(self):
        """Duzeltme, dogru davranisi bozmamali: k'yi gecen ama metrigi hic
        gozlemlenmemis segment hala durustce raporlanmali."""
        r = _c(segment_stats, column="kredi_skoru", operator="<", value=1000,
               metric="temerrut")
        assert "hata" not in r
        assert r["gozlemlenen_satir"] == 0
        assert r["satir_sayisi"] >= K_ANONYMITY_THRESHOLD
        assert "uyari" in r


class TestGuardIzolasyonu:
    """Bulgu 2: _active_guard modul globaliydi; es zamanli isteklerde denetim
    kayitlari birbirine karisiyordu."""

    def test_es_zamanli_istekler_kendi_kaydini_tutar(self):
        sonuc: dict[str, int] = {}

        def istek(ad: str, kolon: str) -> None:
            g = Guard()
            set_guard(g)
            describe_column.invoke({"column": kolon})
            sonuc[ad] = len(g.audit_trail())

        t1 = threading.Thread(target=istek, args=("A", "kredi_skoru"))
        t2 = threading.Thread(target=istek, args=("B", "aylik_gelir"))
        t1.start(); t2.start(); t1.join(); t2.join()

        assert sonuc["A"] >= 1, "A'nin denetim kaydi baska istege dustu"
        assert sonuc["B"] >= 1, "B'nin denetim kaydi baska istege dustu"

    def test_guard_atanmadan_da_calisir(self):
        """ContextVar varsayilansiz; erisim once yapilirsa yenisi uretilmeli."""
        assert isinstance(get_guard(), Guard)

    def test_guard_langgraph_icinden_gorunur(self):
        """ContextVar'a gecisin en sinsi riski: LangGraph araclari baska bir
        baglamda kosturursa ask() icinde kurulan guard araclara gorunmez ve
        denetim kaydi BOS doner. Denetim kaydi bu projede uyumluluk kaniti
        olarak sunuluyor; sessizce bosalmasi kabul edilemez.
        """
        from langchain_core.messages import AIMessage
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode

        from src.tools import ANALYSIS_TOOLS

        graf = StateGraph(MessagesState)
        graf.add_node("araclar", ToolNode(ANALYSIS_TOOLS))
        graf.add_edge(START, "araclar")
        graf.add_edge("araclar", END)
        app = graf.compile()

        g = Guard()
        set_guard(g)
        app.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "describe_column", "args": {"column": "kredi_skoru"},
                             "id": "1", "type": "tool_call"},
                            {"name": "describe_column", "args": {"column": "tckn"},
                             "id": "2", "type": "tool_call"},
                        ],
                    )
                ]
            }
        )

        kayit = g.audit_trail()
        assert kayit, "LangGraph icinden yapilan arac cagrilari denetime yazilmadi"
        assert any(e["allowed"] for e in kayit)
        assert any(not e["allowed"] for e in kayit), "PII reddi kayda gecmedi"


class TestKorelasyonSatirSayisi:
    """Bulgu 3: correlation, NaN'lar dusmesine ragmen len(df) bildiriyordu."""

    def test_nan_iceren_ciftte_gercek_sayi_bildirilir(self):
        df = load_analysis_frame()
        r = _c(correlation, column_a="kredi_skoru", column_b="temerrut")
        beklenen = len(df[["kredi_skoru", "temerrut"]].dropna())

        assert r["kullanilan_satir_sayisi"] == beklenen
        assert r["kullanilan_satir_sayisi"] < len(df)
        assert r["dusen_satir_sayisi"] == len(df) - beklenen

    def test_nansiz_ciftte_hicbir_satir_dusmez(self):
        df = load_analysis_frame()
        r = _c(correlation, column_a="kredi_skoru", column_b="yas")
        assert r["kullanilan_satir_sayisi"] == len(df)
        assert r["dusen_satir_sayisi"] == 0

    def test_eski_yanilticialan_geri_gelmedi(self):
        r = _c(correlation, column_a="kredi_skoru", column_b="temerrut")
        assert "satir_sayisi" not in r, "belirsiz 'satir_sayisi' alani geri gelmis"


class TestUcDegerBastirma:
    """Bulgu 4: sayisal ozetlerde min/max k esigine tabi degildi.

    talep_edilen_tutar'in max degerini yalnizca 2 kisi paylasiyordu; o degeri
    bildirmek toplulastirma gorunumu altinda tekil aciklamadir.
    """

    def test_surekli_kolonda_uc_degerler_bastirilir(self):
        r = _c(describe_column, column="talep_edilen_tutar")
        assert "min" not in r and "max" not in r
        assert set(r["bastirilan_uc_degerler"]) == {"min", "max"}

    def test_yerine_yuzdelikler_verilir(self):
        r = _c(describe_column, column="talep_edilen_tutar")
        assert r["q05"] < r["medyan"] < r["q95"]

    def test_bastirma_denetim_kaydina_gecer(self):
        g = Guard()
        set_guard(g)
        describe_column.invoke({"column": "talep_edilen_tutar"})
        assert any("bastirildi" in e["reason"] for e in g.audit_trail())

    def test_ayrik_kolonda_uc_deger_korunur(self):
        """Asiri bastirma olmamali: cok kisinin paylastigi uc deger verilebilir."""
        df = load_analysis_frame()
        r = _c(describe_column, column="aktif_kredi_sayisi")
        en_dusuk = df["aktif_kredi_sayisi"].min()
        paylasan = int((df["aktif_kredi_sayisi"] == en_dusuk).sum())

        assert paylasan >= K_ANONYMITY_THRESHOLD
        assert r["min"] == pytest.approx(float(en_dusuk))


class TestMaskelemeBaglami:
    """Bulgu 5: 11 haneli her sayi TCKN, 5 ile baslayan 10 haneli her sayi
    telefon sayiliyordu. Bir kredi burosunda toplam portfoy buyuklugu bu
    araliga rahatlikla girer."""

    @pytest.mark.parametrize(
        "metin",
        [
            "Portfoy buyuklugu 12345678901 TL",
            "Toplam 5123456789 kayit",
            "toplam: 12345678901",
            "Musteri sayisi 12345678901",
            "Ortalama tutar 5123456789 TL",
        ],
    )
    def test_olcum_baglamindaki_sayilar_maskelenmez(self, metin):
        assert Guard().mask(metin) == metin

    @pytest.mark.parametrize(
        "metin",
        [
            "TCKN: 12345678901",
            "Kimlik no 12345678901",
            "Tel: 05551234567",
            "iletisim: kullanici@ornek.com",
        ],
    )
    def test_gercek_pii_hala_maskelenir(self, metin):
        assert "MASKELENDI" in Guard().mask(metin)

    @pytest.mark.parametrize(
        "metin",
        [
            "+905551234567",
            "Tel: +905551234567",
            "tel:+905551234567.",
            "05551234567",
            "5551234567",
        ],
    )
    def test_telefon_tum_bicimleriyle_maskelenir(self, metin):
        """Regresyon: desen `\\b(?:\\+90|0)?5[0-9]{9}\\b` seklindeydi.

        Bastaki \\b, dizgenin basinda "+" onunde sinir bulamadigi icin
        "+905551234567" HIC eslesmiyordu. Uluslararasi formattaki numara
        maskeleme katmanindan sessizce siziyordu.
        """
        assert "TELEFON_MASKELENDI" in Guard().mask(metin)

    @pytest.mark.parametrize("metin", ["1234567890", "123456789012", "2026", "905551234567"])
    def test_telefon_olmayan_sayilar_maskelenmez(self, metin):
        """10 hane, 12 hane ve yil gibi degerler telefon sayilmaz.

        Not: "15551234567" bilerek disarida - 11 haneli ve sifirla baslamadigi
        icin TCKN desenine uyar ve maskelenmesi DOGRUDUR.
        """
        assert Guard().mask(metin) == metin

    def test_korunan_sayi_da_denetime_yazilir(self):
        """Maskelenmeme karari da bir karardir; izlenebilir olmali."""
        g = Guard()
        g.mask("Portfoy buyuklugu 12345678901 TL")
        assert any("korundu" in e["reason"] for e in g.audit_trail())


class TestVeriYolu:
    """Bulgu 6: DATA_PATH goreliydi; proje disindan calistirilinca patliyordu."""

    def test_veri_yolu_mutlak(self):
        import os

        assert os.path.isabs(DATA_PATH)

    def test_veri_calisma_dizininden_bagimsiz_yuklenir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        load_analysis_frame.cache_clear()
        assert not load_analysis_frame().empty
