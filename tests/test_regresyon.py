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
from src.agent import _dogrulanmayan_sayilar
from src.guard import Guard, GuardViolation, K_ANONYMITY_THRESHOLD, gecmisi_temizle
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
        """Kaydin ICERIGI dogrulaniyor, yalnizca dolulugu degil.

        Onceki hali `len(audit_trail()) >= 1` diyordu; bu izolasyonu degil
        BOS OLMAMAYI olcuyordu. A'nin kaydi B'nin kolonlarini da icerse test
        yine geciyordu ve etkinligi thread zamanlamasina bagliydi: modul
        duzeyi global guard geri getirildiginde bile, thread'ler sirayla
        kostugunda assert geciyordu. Artik her kayit YALNIZCA kendi kolonunu
        icermeli.
        """
        sonuc: dict[str, list[dict]] = {}

        def istek(ad: str, kolon: str) -> None:
            g = Guard()
            set_guard(g)
            describe_column.invoke({"column": kolon})
            sonuc[ad] = g.audit_trail()

        t1 = threading.Thread(target=istek, args=("A", "kredi_skoru"))
        t2 = threading.Thread(target=istek, args=("B", "aylik_gelir"))
        t1.start(); t2.start(); t1.join(); t2.join()

        for ad, kendi, digeri in (("A", "kredi_skoru", "aylik_gelir"),
                                  ("B", "aylik_gelir", "kredi_skoru")):
            kayit = sonuc[ad]
            assert kayit, f"{ad}'nin denetim kaydi bos"
            metin = " ".join(
                str(e.get("columns", "")) + " " + str(e.get("reason", ""))
                for e in kayit
            )
            assert kendi in metin, f"{ad} kendi kolonunu kaydetmemis"
            assert digeri not in metin, f"{ad}'nin kaydina digerinin kolonu karismis"

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
    # Esik 1893 -> 942 olarak degisti: 1893 veri setinin neredeyse
    # tamamini seciyordu ve artik TUMLEYEN kontrolune takiliyor (tek
    # sorguyla tumleyen toplamindan birey cikarilabiliyordu). 1030 ayni
    # fark alma senaryosunu kurar - n=60 ve n=61 - ama tumleyeni genis
    # birakir, yani test hala FARK ALMA savunmasini olcuyor.

    """Bagimsiz denetimde bulundu: k-anonimlik fark alma saldirisina acikti.

    Iki AYRI sorgu da k esigini geciyordu ama aralarindaki fark tek kisiydi.
    ortalama * satir_sayisi grup toplamini verdigi icin o tek kisinin geliri
    KESIN olarak cikariliyordu - gozlemlenen: 31499.88, gercek 31500.0.
    Iki sorgu da guard'dan onay almisti, denetim kaydinda tek ret yoktu.
    """

    def test_ardisik_dar_farkli_sorgu_reddedilir(self):
        g = Guard()
        set_guard(g)
        r1 = _c(segment_stats, column="kredi_skoru", operator="<", value=1030.0,
                metric="aylik_gelir")
        r2 = _c(segment_stats, column="kredi_skoru", operator="<=", value=1030.0,
                metric="aylik_gelir")
        assert "hata" not in r1, "ilk genis sorgu calismali"
        assert "hata" in r2, "tek kisilik fark reddedilmeli"
        assert any(not e["allowed"] for e in g.audit_trail())

    def test_reddedilen_sorgu_deger_sizdirmaz(self):
        set_guard(Guard())
        _c(segment_stats, column="kredi_skoru", operator="<", value=1030.0,
           metric="aylik_gelir")
        r2 = _c(segment_stats, column="kredi_skoru", operator="<=", value=1030.0,
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
        ilk = _c(segment_stats, column="kredi_skoru", operator="<", value=1030.0,
                 metric="aylik_gelir")
        assert "hata" not in ilk

        for _ in range(GECMIS_SINIRI + 50):
            _c(segment_stats, column="kredi_skoru", operator="<", value=1030.0,
               metric="aylik_gelir")

        gecmis = _SORGU_GECMISI["kredi_skoru|aylik_gelir"]
        assert ilk["gozlemlenen_satir"] in gecmis, "koruyucu kayit gecmisten dusmus"
        assert len(gecmis) <= GECMIS_SINIRI

        sonra = _c(segment_stats, column="kredi_skoru", operator="<=", value=1030.0,
                   metric="aylik_gelir")
        assert "hata" in sonra, "taskin sonrasi saldiri gecti"

    def test_ret_mesaji_onceki_sorgu_boyutunu_sizdirmaz(self):
        """Ret gerekcesi baska bir kullanicinin sorgu boyutunu aciklamamali;
        savunmanin kendisi yan kanal olmamali."""
        from src.guard import gecmisi_temizle

        gecmisi_temizle()
        set_guard(Guard())
        ilk = _c(segment_stats, column="kredi_skoru", operator="<", value=1030.0,
                 metric="aylik_gelir")
        red = _c(segment_stats, column="kredi_skoru", operator="<=", value=1030.0,
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


class TestSayiDayanagi:
    """Bagimsiz Codex denetimi: dayanak kontrolu yalnizca "basarili bir arac
    cagrisi var mi" diye bakiyordu. Agent list_columns cagirip ardindan
    "temerrut orani %98,7" dediginde cevap DAYANAKLI sayiliyordu - alakasiz tek
    bir basarili cagri, cevaptaki tum sayilara sinirsiz dayanak sagliyordu.
    """

    def _kos(self, cevaplar):
        from langchain_core.messages import AIMessage

        from src import agent as A

        class Sahte:
            def __init__(s):
                s.c = list(cevaplar)

            def bind_tools(s, _):
                return s

            def with_retry(s, *a, **k):
                return s

            def invoke(s, m, *a, **k):
                return s.c.pop(0) if s.c else AIMessage(content="Yanitlayamiyorum.")

        eski = A.get_llm
        A.get_llm = lambda *a, **k: Sahte()
        try:
            set_guard(Guard())
            return A.ask("soru")
        finally:
            A.get_llm = eski

    @staticmethod
    def _tc(ad, args=None):
        from langchain_core.messages import AIMessage

        return AIMessage(content="", tool_calls=[
            {"name": ad, "args": args or {}, "id": "1", "type": "tool_call"}])

    def test_alakasiz_cagri_uydurma_sayiya_dayanak_olmaz(self):
        from langchain_core.messages import AIMessage

        r = self._kos([self._tc("list_columns"),
                       AIMessage(content="Temerrut orani %98,7 ve 4200 kisi.")])
        assert r["dayanaksiz_cevap"] is True
        assert set(r["dogrulanmayan_sayilar"]) == {"98.7", "4200"}

    def test_gercek_sayilar_dogrulanir(self):
        from langchain_core.messages import AIMessage

        r = self._kos([self._tc("describe_column", {"column": "kredi_skoru"}),
                       AIMessage(content="Ortalama kredi skoru 1403.42, 5000 satir.")])
        assert r["dayanaksiz_cevap"] is False
        assert r["dogrulanmayan_sayilar"] == []

    def test_yuvarlama_yanlis_pozitif_uretmez(self):
        """Agent 1403.42'yi '1403' diye yazabilir; bu uydurma degildir."""
        from langchain_core.messages import AIMessage

        r = self._kos([self._tc("describe_column", {"column": "kredi_skoru"}),
                       AIMessage(content="Ortalama kredi skoru 1403.")])
        assert r["dogrulanmayan_sayilar"] == []

    def test_arac_cagrilmadan_veri_iddiasi_duzeltmeye_gider(self):
        """Rakamsiz ama veriye dair bir iddia da dayanaksiz kalmamali."""
        from langchain_core.messages import AIMessage

        r = self._kos([AIMessage(content="En riskli grup kamudur.")])
        assert r["duzeltme_denemesi"] >= 1

    def test_selamlama_engellenmez(self):
        from langchain_core.messages import AIMessage

        r = self._kos([AIMessage(content="Merhaba, nasil yardimci olabilirim?")])
        assert r["duzeltme_denemesi"] == 0
        assert r["dayanaksiz_cevap"] is False


class TestDenetimKaydiEksiksiz:
    """Bagimsiz Codex denetimi: 'her guard karari kayda gecer' iddiasi
    tutmuyordu. Basarili kontroller yazilmiyordu ve arac duzeyi dogrulama
    hatalari denetim izinde HIC gorunmuyordu - gecersiz bir cagride kayit
    yalnizca 'izin verildi' satirlarindan olusuyordu.
    """

    def test_arac_dogrulama_hatasi_kayda_gecer(self):
        g = Guard()
        set_guard(g)
        _c(group_aggregate, group_by="kredi_skoru", metric="aylik_gelir", how="mean")
        assert any(not e["allowed"] for e in g.audit_trail())

    def test_basarili_k_kontrolu_kayda_gecer(self):
        g = Guard()
        set_guard(g)
        _c(segment_stats, column="kredi_skoru", operator="<", value=1400,
           metric="aylik_gelir")
        gerekceler = " ".join(e["reason"] for e in g.audit_trail())
        assert "k kontrolu gecildi" in gerekceler
        assert "Fark alma kontrolu gecildi" in gerekceler

    def test_bastirma_olmasa_da_grup_karari_kayda_gecer(self):
        g = Guard()
        set_guard(g)
        _c(describe_column, column="il")
        assert any("k esigini gecti" in e["reason"] for e in g.audit_trail())


class TestQwenDenetimi:
    """Dorduncu bagimsiz denetim (qwen3.7-plus, sifir bilgiyle).

    Denetim yedi bulgu bildirdi; ucu gercek cikti, digerleri ya bilincli
    tasarim kararlariydi ya da somurulemiyordu. Buradaki testler HEM
    duzeltilenleri HEM de kasitli davranislari kilitliyor: bir sonraki
    denetci ayni yerlere takildiginda gerekce kodda yazili olsun.
    """

    def test_bosluklu_telefon_maskelenir(self):
        """Maskeleme yalnizca bitisik yazimi taniyordu.

        Model numarayi "+90 555 123 45 67" ya da "(555) 123-4567" diye
        yazdiginda son savunma hatti sessizce devre disi kaliyordu.
        """
        g = Guard()
        for metin in [
            "+90 555 123 45 67",
            "(555) 123-4567",
            "555-123-45-67",
            "0 555 123 45 67",
        ]:
            assert "[TELEFON_MASKELENDI]" in g.mask(metin), metin

    def test_bitisik_tckn_maskelenir(self):
        """\b harf/rakam sinirinda bulundugu icin "x12345678901y"
        maskelenmeden geciyordu."""
        assert "[TCKN_MASKELENDI]" in Guard().mask("x12345678901y")

    def test_binlik_ayracli_tutar_telefon_sanilmaz(self):
        """Maskelemeyi genisletirken mesru tutarlari yakmadigimizin kaniti."""
        assert Guard().mask("Toplam borc 5.000.000.000 TL") == (
            "Toplam borc 5.000.000.000 TL"
        )

    def test_carpanli_ifade_dayanak_kontrolunden_kacamaz(self):
        """"1,4 milyon" yazan cevapta duz tarama yalnizca "1,4" goruyordu;
        o da 10 esiginin altinda kaldigi icin hic denetlenmiyordu."""
        ciktilar = ['{"toplam": 1400000}']
        assert _dogrulanmayan_sayilar("Toplam 1,4 milyon TL.", ciktilar) == []
        assert _dogrulanmayan_sayilar("Toplam 9 milyon TL.", ciktilar) == ["9 milyon"]

    def test_noktali_binlik_yazim_dayanak_kontrolunden_kacamaz(self):
        """"1.403.421" float()'ta patlayip sayi SESSIZCE dusuyordu -
        model rakami boyle yazdiginda uydurma kontrolu hic calismiyordu."""
        ciktilar = ['{"toplam": 1400000}']
        assert _dogrulanmayan_sayilar("Toplam 1.403.421 TL.", ciktilar) == ["1403421"]
        assert _dogrulanmayan_sayilar("Toplam 1.400.000 TL.", ciktilar) == []

    def test_fark_tam_k_kadar_ise_KASITLI_olarak_gecer(self):
        """Denetim bunu 'off-by-one hatasi' diye bildirdi; hata degil.

        Fark tam K_ANONYMITY_THRESHOLD ise aciga cikan grup K kisilik olur.
        k-anonimlik tanimi geregi K ve uzeri gruplar zaten serbest; kosulu
        <= yapmak politikayi sessizce k=21'e cekerdi. Davranis kasitli.
        """
        gecmisi_temizle()
        g = Guard()
        g.check_overlap("t", "yas|aylik_gelir", 4968)
        g.check_overlap("t", "yas|aylik_gelir", 4968 + K_ANONYMITY_THRESHOLD)

        gecmisi_temizle()
        g2 = Guard()
        g2.check_overlap("t", "yas|aylik_gelir", 4968)
        with pytest.raises(GuardViolation):
            g2.check_overlap("t", "yas|aylik_gelir", 4968 + K_ANONYMITY_THRESHOLD - 1)

    def test_farkli_metrik_ayri_gecmis_tutar_ama_ifsa_yolu_yok(self):
        """Denetim bunu KRITIK bildirdi; olgu dogru, sonuc yanlis.

        Gecmis anahtari "kolon|metrik" oldugu icin farkli metrikli iki sorgu
        birbirini blokelamaz. Ancak fark alma saldirisi AYNI metrigin iki
        satir sayisini gerektirir - o da reddediliyor. Iki farkli kolonun
        ortalamasi arasindaki farktan kimsenin degeri cikmaz.

        Anahtari kolona indirmek olculebilir guvenlik kazanci vermeden mesru
        sorgularin reddini artiracagi icin bilerek degistirilmedi.
        """
        gecmisi_temizle()
        g = Guard()
        g.check_overlap("t", "kredi_skoru|aylik_gelir", 4996)
        g.check_overlap("t", "kredi_skoru|mevcut_borc", 4997)   # farkli anahtar

        with pytest.raises(GuardViolation):
            g.check_overlap("t", "kredi_skoru|aylik_gelir", 4997)


class TestOpus5Denetimi:
    """Besinci bagimsiz denetim (Opus 5, sifir bilgiyle) - kanitli bulgular."""

    def test_tumleyen_kume_ile_birey_cikarilamaz(self):
        """KRITIK, olculdu: tek sorguyla bir kisinin geliri 2 kurus hatayla
        cikarildi ve guard TEK BIR RET vermedi.

        segment_stats ayni yanitta ortalama, genel_ortalama ve toplam satir
        sayisini birlikte donduruyordu:
            N * genel_ortalama - n * ortalama = tumleyenin toplami
        Tumleyen tek kisiyse o kisinin degeri KESIN cikar. Fark alma savunmasi
        devreye girmiyordu cunku saldiri iki sorgu degil, BIR sorgu.
        Olculen: cikarilan 410899.98 / gercek 410900.0.
        """
        gecmisi_temizle()
        set_guard(Guard())
        df = load_analysis_frame()
        tepe = float(df["mevcut_borc"].max())
        assert int((df["mevcut_borc"] == tepe).sum()) == 1, "senaryo tek kisilik tepe ister"

        r = _c(segment_stats, column="mevcut_borc", operator="<", value=tepe,
               metric="aylik_gelir")
        assert "hata" in r, "tumleyeni tek kisi olan sorgu REDDEDILMELI"
        assert "ortalama" not in r and "genel_ortalama" not in r

    def test_tumleyen_genisse_sorgu_calisir(self):
        """Tumleyen kontrolu mesru genis sorgulari bozmamali."""
        gecmisi_temizle()
        set_guard(Guard())
        r = _c(segment_stats, column="yas", operator=">", value=40.0,
               metric="mevcut_borc")
        assert "hata" not in r and r["satir_sayisi"] > 20

    def test_describe_column_sayisal_dalda_k_uygulanir(self):
        """Sayisal dalda ortalama/std/ceyreklikler hicbir esige tabi degildi;
        degismez veriye degil koda baglanmaliydi."""
        g = Guard()
        set_guard(g)
        _c(describe_column, column="aylik_gelir")
        assert any("k kontrolu gecildi" in e["reason"] for e in g.audit_trail())

    def test_kucuk_sayilar_da_denetlenir(self):
        """Uydurma tespiti mutlak degeri 10'dan kucuk her sayiyi ATLIYORDU.

        Kredi riskinde raporlanan buyukluklerin cogu bu araliktadir: temerrut
        oranlari, korelasyon katsayilari, ortalama kredi sayisi. Tamami kucuk
        sayilardan olusan uydurma bir cevap hicbir kontrole takilmiyordu.
        """
        ciktilar = ['{"kolon_sayisi": 12}']
        assert _dogrulanmayan_sayilar(
            "Temerrut orani %7,3; korelasyon 0,42.", ciktilar
        ) == ["7.3", "0.42"]

    def test_yuz_kati_sahte_dayanak_uretmez(self):
        """_dayanakli_degerler her sayinin 100 katini da dayanak sayiyordu:
        arac 'satir_sayisi 5000' derse cevaptaki uydurma 500000 dayanakli
        oluyordu."""
        assert _dogrulanmayan_sayilar("Toplam 500000 TL.", ['{"satir_sayisi": 5000}']) == ["500000"]

    def test_oran_yuzde_donusumu_hala_kabul_edilir(self):
        """x100 kaldirilirken mesru oran -> yuzde yazimi bozulmamali;
        yalnizca % isareti VARKEN ve tek yonlu kabul ediliyor."""
        assert _dogrulanmayan_sayilar("Onay orani %68,11.", ['{"oran": 0.6811}']) == []

    def test_ayracli_tckn_maskelenir(self):
        """Telefonda ayrac toleransi vardi, TCKN'de yoktu; ayni katmanin
        kendi ilkesi ('yanlis negatif yikici') TCKN'de uygulanmiyordu."""
        g = Guard()
        for metin in ["123 456 789 01", "12345-678901", "123.456.789.01"]:
            assert "[TCKN_MASKELENDI]" in g.mask(metin), metin

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_tanimsiz_korelasyon_guclu_negatif_demez(self):
        """abs(nan) hicbir esikten kucuk degil, nan > 0 da False: tanimsiz r
        SESSIZCE 'guclu negatif' oluyordu. Sayi olmadigi icin dayanak
        kontrolu de yakalayamiyordu."""
        import src.tools as T

        df = load_analysis_frame().copy()
        df["vade_ay"] = 24
        orijinal = T.load_analysis_frame
        T.load_analysis_frame = lambda: df
        try:
            set_guard(Guard())
            r = _c(correlation, column_a="vade_ay", column_b="aylik_gelir")
        finally:
            T.load_analysis_frame = orijinal
        assert "hata" in r
        assert r.get("guc") is None and r.get("yon") is None

    def test_rate_ikili_olmayan_kolonda_reddedilir(self):
        """'rate' 0/1 kolonlar icindir; kod bunu hic dogrulamiyor, ikili
        olmayan kolonda ortalamayi 'rate' etiketiyle donduruyordu."""
        set_guard(Guard())
        red = _c(group_aggregate, group_by="meslek_grubu", metric="aylik_gelir", how="rate")
        assert "hata" in red
        ok = _c(group_aggregate, group_by="meslek_grubu", metric="temerrut", how="rate")
        assert "hata" not in ok and ok["sonuc"]

    def test_maskeleme_denetim_kaydini_ve_arac_girdisini_de_kapsar(self):
        """mask() YALNIZCA cevap alanina uygulaniyordu.

        Guard bilinmeyen kolon adini denetim kaydina birebir yaziyor; model
        kullanicinin sorusundaki bir deseni arac argumanina koydugunda o desen
        maskelenmeden API cevabina ve CLI ciktisina ulasiyordu.
        """
        from src.agent import _maskeli

        g = Guard()
        cikti = _maskeli(
            {
                "denetim_kaydi": [{"reason": "Bilinmeyen kolon: x12345678901y"}],
                "kullanilan_araclar": [{"girdi": {"column": "a 0532 123 45 67 b"}}],
                "sayi": 42,
                "bayrak": True,
            },
            g,
        )
        assert "12345678901" not in str(cikti["denetim_kaydi"])
        assert "[TCKN_MASKELENDI]" in str(cikti["denetim_kaydi"])
        assert "[TELEFON_MASKELENDI]" in str(cikti["kullanilan_araclar"])
        # Sayi ve bool alanlar bozulmamali
        assert cikti["sayi"] == 42 and cikti["bayrak"] is True

    def test_ortuk_guard_kayitta_gorunur(self):
        """get_guard() sessizce yeni Guard uretiyordu: kontroller calisiyor
        ama kararlar ask()'in denetim kaydina HIC girmiyordu. Denetim kaydi
        uyumluluk kaniti olarak sunuldugu icin sessizce eksilmemeli."""
        # Taze bir thread'de ContextVar bos baslar; ayni thread'de reset
        # etmek yetmiyor cunku onceki testler degeri doldurmus oluyor.
        kayit: list[dict] = []

        def taze_baglamda() -> None:
            kayit.extend(get_guard().audit_trail())

        t = threading.Thread(target=taze_baglamda)
        t.start(); t.join()
        assert any("Ortuk guard" in e["reason"] for e in kayit)

    def test_bastirilan_satir_sayisi_kaba_aralik_olarak_verilir(self):
        """Toplami gizlemek tek basina yetmiyordu: N baska araclardan tam
        alinabildigi icin tek grup bastirildiginda boyutu N - gorunen ile
        birebir geri cozuluyordu."""
        import src.tools as T

        df = load_analysis_frame().copy()
        df.loc[df.index[:5], "il"] = "NADIR_IL"
        orijinal = T.load_analysis_frame
        T.load_analysis_frame = lambda: df
        try:
            set_guard(Guard())
            r = _c(describe_column, column="il")
        finally:
            T.load_analysis_frame = orijinal

        assert r["bastirilan_grup_sayisi"] == 1
        assert "satir_sayisi" not in r          # kesin toplam hala gizli
        assert r["bastirilan_yaklasik_satir"] == "0-20"


class TestOpus5IkinciTur:
    """Altinci bagimsiz denetim: onceki turun duzeltmelerindeki BOSLUKLAR."""

    def test_tumleyen_kontrolu_nan_metrikte_de_calisir(self):
        """Ilk duzeltme toplam=len(df) geciyordu.

        Metrik NaN tasidiginda (temerrut yalnizca onaylanan basvurularda dolu)
        gozlemlenen populasyon daha kucuk ve genel_ortalama da o populasyondan
        geliyor. len(df) verilince tumleyen 1404 sanildi, GERCEKTE 1 kisiydi
        ve guard tek ret vermeden gecti - olculdu.
        """
        gecmisi_temizle()
        set_guard(Guard())
        r = _c(segment_stats, column="kredi_skoru", operator=">", value=1103.0,
               metric="temerrut")
        assert "hata" in r, "gozlemlenen tumleyeni tek kisi olan sorgu REDDEDILMELI"

    def test_describe_column_k_ihlalinde_cokmez(self):
        """GuardViolation ToolNode tarafindan ToolMessage'a cevrilmiyor; tum
        ajan kosumu duserdi. k kontrolu eklenirken try/except atlanmisti."""
        import src.tools as T

        df = load_analysis_frame().copy().head(10)
        orijinal = T.load_analysis_frame
        T.load_analysis_frame = lambda: df
        try:
            set_guard(Guard())
            r = _c(describe_column, column="aylik_gelir")   # istisna FIRLATMAMALI
        finally:
            T.load_analysis_frame = orijinal
        assert "hata" in r and "satir" in r["hata"]

    def test_turkce_virgul_her_zaman_ondalik(self):
        """Virgulden sonra 3 hane varsa binlik ayraci sayan kural iki yonlu
        bozuktu: dogru cevap uydurma damgalaniyor, uydurma sayi onaylaniyordu."""
        from src.agent import _sayilari_cikar

        assert _sayilari_cikar("-0,286") == ["-0.286"]
        assert _sayilari_cikar("1,403") == ["1.403"]
        assert _sayilari_cikar("1.234,56") == ["1234.56"]
        # dogru yuvarlanmis cevap damgalanmamali
        assert _dogrulanmayan_sayilar("Korelasyon -0,286.", ['{"r": -0.2862}']) == []
        # arac ciktisinda 1403 var ama cevap 1,403 (yani 1.403) - uydurma
        assert _dogrulanmayan_sayilar("Toplam 1,403 adet.", ['{"dusen": 1403}']) == ["1.403"]

    def test_bastirma_varken_grup_sayilari_kabalasir(self):
        """Kesin grup sayilari verildiginde bastirilan grubun toplami
        aritmetikle geri cozuluyordu (olculdu: 5.87 TL hatayla tek kisinin
        geliri). Bastirma YOKKEN kesin sayi korunuyor."""
        import src.tools as T

        df = load_analysis_frame().copy()
        df.loc[df.index[0], "il"] = "NADIR"
        orijinal = T.load_analysis_frame
        T.load_analysis_frame = lambda: df
        try:
            set_guard(Guard())
            bastirmali = _c(group_aggregate, group_by="il", metric="aylik_gelir", how="mean")
        finally:
            T.load_analysis_frame = orijinal
        assert bastirmali["bastirilan_grup_sayisi"] == 1
        assert all(str(v).endswith("+") for v in bastirmali["gozlemlenen_satir_sayisi"].values())

        set_guard(Guard())
        temiz = _c(group_aggregate, group_by="meslek_grubu", metric="aylik_gelir", how="mean")
        assert temiz["bastirilan_grup_sayisi"] == 0
        assert all(isinstance(v, int) for v in temiz["gozlemlenen_satir_sayisi"].values())

    def test_gorunen_satir_sayisi_da_kabalasir(self):
        """Kaba aralik tek basina hicbir sey gizlemiyordu: N baska araclardan
        tam alinabildigi icin gizlenen sayi N - gorunen ile birebir cozuluyordu."""
        import src.tools as T

        df = load_analysis_frame().copy()
        df.loc[df.index[:5], "il"] = "NADIR"
        orijinal = T.load_analysis_frame
        T.load_analysis_frame = lambda: df
        try:
            set_guard(Guard())
            r = _c(describe_column, column="il")
        finally:
            T.load_analysis_frame = orijinal
        assert str(r["gorunen_satir_sayisi"]).endswith("+")

    def test_correlation_retleri_denetim_kaydina_girer(self):
        """group_aggregate ve segment_stats ayni durumlarda reddet() cagiriyordu;
        correlation'in ret yollari atlanmisti."""
        g = Guard()
        set_guard(g)
        _c(correlation, column_a="aylik_gelir", column_b="aylik_gelir")
        assert any(not e["allowed"] and e["action"] == "correlation" for e in g.audit_trail())
