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
from src.guard import Guard, K_ANONYMITY_THRESHOLD, gecmisi_temizle
from src.tools import (
    group_aggregate,
    correlation,
    describe_column,
    get_guard,
    load_analysis_frame,
    segment_stats,
    set_guard,
)


@pytest.fixture(autouse=True)
def temiz_guard():
    # Fark alma gecmisi surec genelinde; testler birbirini etkilemesin.
    gecmisi_temizle()
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

    @pytest.mark.parametrize("kolon", ["kredi_skoru", "yas", "temerrut"])
    def test_ayni_kolon_cokme_yerine_hata_dondurur(self, kolon):
        """Regresyon: correlation(x, x) ValueError ile cokuyordu.

        df[[a, a]] iki AYNI ADLI kolon uretir; cift[a] o zaman Series degil
        DataFrame doner ve .corr() 'truth value of a DataFrame is ambiguous'
        ile patlar. Anlamsiz bir sorgu, ama coken degil aciklayan bir cevap
        vermeli - agent hatayi gorup plan degistirebilsin.
        """
        r = _c(correlation, column_a=kolon, column_b=kolon)
        assert "hata" in r
        assert "pearson_r" not in r

    def test_farkli_kolonlar_hala_calisiyor(self):
        r = _c(correlation, column_a="kredi_skoru", column_b="yas")
        assert "pearson_r" in r


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


class TestMaskelemeKosulsuz:
    """Baglam sezgiseli KALDIRILDI - bagimsiz denetimde atlatilabildigi gosterildi.

        "TCKN degeri: 12345678901 TL"      -> maskelenmiyordu
        "Telefon degeri: 05551234567 adet" -> maskelenmiyordu

    Etiket listesini genisletmek cozum degil; sezgisel oldugu surece bir sonraki
    ifade yine disarida kalir. Son savunma hattinda yanlis pozitif gorunur ve
    zararsiz, yanlis negatif gorunmez ve yikicidir.
    """

    @pytest.mark.parametrize(
        "metin",
        [
            "TCKN degeri: 12345678901 TL",
            "Telefon degeri: 05551234567 adet",
            "Musteri no 12345678901 kisi",
            "TCKN: 12345678901 TL",
            "12345678901 adet",
            "toplam: 12345678901",
            "Toplam portfoy 12345678901 TL",
            "+905551234567",
            "iletisim: a@b.com",
        ],
    )
    def test_desen_eslesen_her_sey_maskelenir(self, metin):
        assert "MASKELENDI" in Guard().mask(metin)

    @pytest.mark.parametrize(
        "metin",
        ["Ortalama talep 125000 TL", "Kredi skoru 1850", "2026", "1234567890"],
    )
    def test_desen_eslesmeyen_sayilar_dokunulmaz(self, metin):
        assert Guard().mask(metin) == metin



class TestDayanakKontrolu:
    """Bulgu 7: agent, arac hatasi aldiginda cevabi UYDURDU.

    Gercek kosum: "kredi_skoru < 800 olanlarin ortalama geliri nedir?" sorusuna
    agent metric='gelir' diye olmayan bir kolonla cagri yapti, guard reddetti,
    agent duzeltmek yerine "1500 satir, 27.500 TL" dedi. Gercekte 4 satir vardi
    ve dogru cagri k esiginden zaten reddedilecekti.

    Araclar hatalarini istisna yerine {"hata": ...} olarak donduruyor - agent'in
    plan degistirebilmesi icin. Ama bu, hatanin sessizce yutulabilmesi demek.
    Kontrol modelin iyi niyetine birakilamaz; kod yolunda olmali.
    """

    def _tool_msg(self, icerik: str):
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=icerik, tool_call_id="1")

    def test_hatali_ciktilar_sayilir(self):
        from src.agent import _basarili_arac_ciktisi_var_mi

        mesajlar = [
            self._tool_msg('{"hata": "Bilinmeyen kolon: gelir"}'),
            self._tool_msg('{"hata": "Bu filtre yalnizca 4 satir donduruyor."}'),
        ]
        assert _basarili_arac_ciktisi_var_mi(mesajlar) == (0, 2)

    def test_basarili_ciktilar_sayilir(self):
        from src.agent import _basarili_arac_ciktisi_var_mi

        mesajlar = [
            self._tool_msg('{"kolon": "yas", "ortalama": 40.5}'),
            self._tool_msg('{"hata": "Bilinmeyen kolon: gelir"}'),
            self._tool_msg('{"kolon": "kredi_skoru", "ortalama": 1403.4}'),
        ]
        assert _basarili_arac_ciktisi_var_mi(mesajlar) == (2, 1)

    def test_arac_cagrisi_yoksa_ikisi_de_sifir(self):
        from src.agent import _basarili_arac_ciktisi_var_mi

        assert _basarili_arac_ciktisi_var_mi([]) == (0, 0)

    def test_json_olmayan_cikti_basarili_sayilir(self):
        """search_data_dictionary duz metin donduruyor; hata sayilmamali."""
        from src.agent import _basarili_arac_ciktisi_var_mi

        assert _basarili_arac_ciktisi_var_mi([self._tool_msg("duz metin")]) == (1, 0)

    def test_hata_kelimesi_iceren_mesru_cikti_hata_sayilmaz(self):
        """Icinde 'hata' gecen ama hata anahtari olmayan cikti."""
        from src.agent import _basarili_arac_ciktisi_var_mi

        m = self._tool_msg('{"kolon": "notlar", "dagilim": {"hata_kaydi": 12}}')
        assert _basarili_arac_ciktisi_var_mi([m]) == (1, 0)


class TestVeriYolu:
    """Bulgu 6: DATA_PATH goreliydi; proje disindan calistirilinca patliyordu."""

    def test_veri_yolu_mutlak(self):
        import os

        assert os.path.isabs(DATA_PATH)

    def test_veri_calisma_dizininden_bagimsiz_yuklenir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        load_analysis_frame.cache_clear()
        assert not load_analysis_frame().empty


class TestFarkAlmaSaldirisi:
    """Bagimsiz denetimde bulundu: k-anonimlik fark alma saldirisina acikti.

    Iki AYRI sorgu da k esigini geciyordu ama aralarindaki fark tek kisiydi.
    ortalama * satir_sayisi grup toplamini verdigi icin o tek kisinin geliri
    KESIN olarak cikariliyordu - gozlemlenen: 31499.88, gercek 31500.0.
    Iki sorgu da guard'dan onay almisti, denetim kaydinda tek ret yoktu.
    """

    def test_ardisik_dar_farkli_sorgu_reddedilir(self):
        g = Guard()
        set_guard(g)
        r1 = _c(segment_stats, column="kredi_skoru", operator="<", value=1893.0,
                metric="aylik_gelir")
        r2 = _c(segment_stats, column="kredi_skoru", operator="<=", value=1893.0,
                metric="aylik_gelir")
        assert "hata" not in r1, "ilk genis sorgu calismali"
        assert "hata" in r2, "tek kisilik fark reddedilmeli"
        assert any(not e["allowed"] for e in g.audit_trail())

    def test_reddedilen_sorgu_deger_sizdirmaz(self):
        set_guard(Guard())
        _c(segment_stats, column="kredi_skoru", operator="<", value=1893.0,
           metric="aylik_gelir")
        r2 = _c(segment_stats, column="kredi_skoru", operator="<=", value=1893.0,
                metric="aylik_gelir")
        for alan in ("ortalama", "medyan", "gozlemlenen_satir", "genel_ortalama"):
            assert alan not in r2

    def test_belirgin_farkli_sorgular_calismaya_devam_eder(self):
        """Asiri bastirma olmamali: mesru analiz engellenmemeli."""
        set_guard(Guard())
        a = _c(segment_stats, column="kredi_skoru", operator="<", value=1200,
               metric="aylik_gelir")
        b = _c(segment_stats, column="kredi_skoru", operator="<", value=1600,
               metric="aylik_gelir")
        assert "hata" not in a and "hata" not in b
        assert a["gozlemlenen_satir"] != b["gozlemlenen_satir"]

    def test_gecmis_taskini_savunmayi_asamaz(self):
        """Bagimsiz denetim: ayni zararsiz sorgu tekrarlanarak gecmis doldurulup
        koruyucu kayit disari itilebiliyordu (liste + kirpma). Kume ve
        fail-closed kapasite ile kapatildi.
        """
        from src.guard import GECMIS_SINIRI, _SORGU_GECMISI, gecmisi_temizle

        gecmisi_temizle()
        set_guard(Guard())
        ilk = _c(segment_stats, column="kredi_skoru", operator="<", value=1893.0,
                 metric="aylik_gelir")
        assert "hata" not in ilk

        for _ in range(GECMIS_SINIRI + 50):
            _c(segment_stats, column="kredi_skoru", operator="<", value=942.0,
               metric="aylik_gelir")

        gecmis = _SORGU_GECMISI["kredi_skoru|aylik_gelir"]
        assert ilk["gozlemlenen_satir"] in gecmis, "koruyucu kayit gecmisten dusmus"
        assert len(gecmis) <= GECMIS_SINIRI

        sonra = _c(segment_stats, column="kredi_skoru", operator="<=", value=1893.0,
                   metric="aylik_gelir")
        assert "hata" in sonra, "taskin sonrasi saldiri gecti"

    def test_ret_mesaji_onceki_sorgu_boyutunu_sizdirmaz(self):
        """Ret gerekcesi baska bir kullanicinin sorgu boyutunu aciklamamali;
        savunmanin kendisi yan kanal olmamali."""
        from src.guard import gecmisi_temizle

        gecmisi_temizle()
        set_guard(Guard())
        ilk = _c(segment_stats, column="kredi_skoru", operator="<", value=1893.0,
                 metric="aylik_gelir")
        red = _c(segment_stats, column="kredi_skoru", operator="<=", value=1893.0,
                 metric="aylik_gelir")
        assert "hata" in red
        assert str(ilk["gozlemlenen_satir"]) not in red["hata"]

    def test_farkli_kolon_ciftleri_birbirini_etkilemez(self):
        set_guard(Guard())
        a = _c(segment_stats, column="kredi_skoru", operator="<", value=1400,
               metric="aylik_gelir")
        b = _c(segment_stats, column="yas", operator="<", value=40, metric="mevcut_borc")
        assert "hata" not in a and "hata" not in b


class TestKategorikKolonCokmesi:
    """Bagimsiz denetimde bulundu: kategorik kolon CIPLAK istisna firlatiyordu.

    ToolNode bu istisnayi yakalamadigi icin TUM ajan kosumu duyuyordu; agent
    hatayi gorup plan degistiremiyordu. "Istanbul'daki ortalama gelir nedir?"
    gibi dogal bir soru bu yola giriyordu.
    """

    @pytest.mark.parametrize(
        "kw",
        [
            {"column": "il", "operator": "<", "value": 1.0, "metric": "aylik_gelir"},
            {"column": "il", "operator": "==", "value": 1.0, "metric": "aylik_gelir"},
            {"column": "yas", "operator": "<", "value": 40, "metric": "il"},
            {"column": "meslek_grubu", "operator": ">", "value": 0, "metric": "basvuru_sonucu"},
        ],
    )
    def test_segment_stats_kategorikte_cokmez(self, kw):
        r = _c(segment_stats, **kw)
        assert "hata" in r
        assert "sayisal degil" in r["hata"]

    @pytest.mark.parametrize("how", ["mean", "median", "sum", "rate"])
    def test_group_aggregate_kategorik_metrikte_cokmez(self, how):
        r = _c(group_aggregate, group_by="meslek_grubu", metric="il", how=how)
        assert "hata" in r

    def test_count_kategorik_metrikte_hala_calisir(self):
        """'count' her tipte anlamli; asiri kisitlama olmamali."""
        r = _c(group_aggregate, group_by="meslek_grubu", metric="il", how="count")
        assert "hata" not in r
        assert r["sonuc"]

    def test_sayisal_kolonlar_etkilenmedi(self):
        r = _c(segment_stats, column="kredi_skoru", operator="<", value=1400,
               metric="aylik_gelir")
        assert "hata" not in r



class TestPiiBellegeGirmez:
    """Bagimsiz denetim: pd.read_csv once 17 kolonun TAMAMINI yukluyor, sonra
    drop ediyordu. 'Analiz katmaninin belleginde o veri hic bulunmaz' iddiasi
    teknik olarak yanlisti."""

    def test_yalnizca_analize_acik_kolonlar_okunur(self):
        from src.schema import ANALYZABLE_COLUMNS, PII_COLUMNS

        load_analysis_frame.cache_clear()
        df = load_analysis_frame()
        assert set(df.columns) == set(ANALYZABLE_COLUMNS)
        assert not [c for c in PII_COLUMNS if c in df.columns]
