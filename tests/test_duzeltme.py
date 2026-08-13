"""Zorunlu tekrar deneme (duzeltme dongusu) testleri.

Model cagirmadan test edilir: gercek LLM yerine, davranisi onceden yazilmis
sahte bir model kullaniliyor. Boylece "agent hata alinca ne yapar" sorusu
deterministik olarak sinanabiliyor - gercek modelle bu ancak sansa bagli
gozlemlenirdi.

API anahtari gerektirmez.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import agent as ajan_modulu
from src.agent import MAKS_DUZELTME, _duzeltici_mesaj, _hata_metinleri, build_agent
from src.guard import Guard
from src.tools import set_guard


class SahteModel:
    """Onceden yazilmis cevaplari sirayla dondurur.

    LangGraph'in bind_tools cagrisini de karsilamasi gerekiyor; kendini
    donduruyor cunku arac baglama davranisini taklit etmeye gerek yok.
    """

    def __init__(self, cevaplar: list[AIMessage]) -> None:
        self.cevaplar = list(cevaplar)
        self.gorulen_istemler: list[list] = []

    def bind_tools(self, _araclar):
        return self

    def with_retry(self, *args, **kwargs):
        # config.dayanikli() gercek modeli Runnable.with_retry ile sariyor;
        # sahte model bu cagriyi karsilamali.
        return self

    def invoke(self, mesajlar, *args, **kwargs) -> AIMessage:
        self.gorulen_istemler.append(mesajlar)
        if self.cevaplar:
            return self.cevaplar.pop(0)
        return AIMessage(content="Baska sozum yok.")


def _arac_cagrisi(ad: str, args: dict, cid: str = "1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": ad, "args": args, "id": cid, "type": "tool_call"}],
    )


@pytest.fixture
def sahte(monkeypatch):
    """get_llm'i sahte modelle degistirir."""

    def kur(cevaplar):
        model = SahteModel(cevaplar)
        monkeypatch.setattr(ajan_modulu, "get_llm", lambda *a, **k: model)
        set_guard(Guard())
        return model

    return kur


class TestDuzelticiMesaj:
    def test_bilinmeyen_kolon_icin_somut_yonlendirme(self):
        mesaj = _duzeltici_mesaj(["Bilinmeyen kolon: gelir"])
        assert "list_columns" in mesaj
        assert "TEKRAR" in mesaj

    def test_pii_icin_tekrar_denemeye_yonlendirmez(self):
        mesaj = _duzeltici_mesaj(["Su kolonlar kisisel veri iceriyor: tckn"])
        assert "hicbir kosulda acilmayacak" in mesaj

    def test_dar_segment_icin_genislet_der(self):
        mesaj = _duzeltici_mesaj(["Bu filtre yalnizca 4 satir donduruyor."])
        assert "genislet" in mesaj.lower()

    def test_taninmayan_hata_icin_genel_yonlendirme(self):
        mesaj = _duzeltici_mesaj(["Beklenmedik bir sey oldu"])
        assert "list_columns" in mesaj

    def test_hata_metinleri_arac_mesajlarindan_toplanir(self):
        from langchain_core.messages import ToolMessage

        mesajlar = [
            ToolMessage(content='{"hata": "Bilinmeyen kolon: gelir"}', tool_call_id="1"),
            ToolMessage(content='{"kolon": "yas"}', tool_call_id="2"),
            ToolMessage(content="duz metin", tool_call_id="3"),
        ]
        assert _hata_metinleri(mesajlar) == ["Bilinmeyen kolon: gelir"]


class TestDuzeltmeDongusu:
    def test_hatadan_sonra_uydurma_engellenir_ve_tekrar_denenir(self, sahte):
        """Gercek gozlenen senaryo: yanlis kolon adi -> hata -> uydurma.

        Simdi araya duzeltme dugumu giriyor; model ikinci turda dogru cagriyi
        yapiyor ve cevap gercek veriye dayaniyor.
        """
        model = sahte(
            [
                _arac_cagrisi("segment_stats", {
                    "column": "kredi_skoru", "operator": "<", "value": 1000,
                    "metric": "gelir",
                }),
                AIMessage(content="Ortalama gelir 27.500 TL, 1500 satirdan."),
                _arac_cagrisi("segment_stats", {
                    "column": "kredi_skoru", "operator": "<", "value": 1000,
                    "metric": "aylik_gelir",
                }, cid="2"),
                AIMessage(content="Ortalama aylik gelir 37.038 TL."),
            ]
        )

        sonuc = build_agent().invoke(
            {"messages": [HumanMessage(content="soru")], "duzeltme_denemesi": 0}
        )

        assert sonuc["duzeltme_denemesi"] == 1
        assert sonuc["messages"][-1].content == "Ortalama aylik gelir 37.038 TL."

        # Duzeltme mesaji gercekten modele gitti mi?
        son_istem = model.gorulen_istemler[-1]
        assert any("DUR." in str(m.content) for m in son_istem)

    def test_basarili_cagri_varsa_duzeltmeye_gitmez(self, sahte):
        sahte(
            [
                _arac_cagrisi("describe_column", {"column": "kredi_skoru"}),
                AIMessage(content="Ortalama kredi skoru 1403."),
            ]
        )
        sonuc = build_agent().invoke(
            {"messages": [HumanMessage(content="soru")], "duzeltme_denemesi": 0}
        )
        assert sonuc["duzeltme_denemesi"] == 0

    def test_hic_arac_cagrilmadiysa_duzeltmeye_gitmez(self, sahte):
        """Arac cagirmadan cevap veren bir soru (ornegin selamlama) engellenmemeli."""
        sahte([AIMessage(content="Merhaba, nasil yardimci olabilirim?")])
        sonuc = build_agent().invoke(
            {"messages": [HumanMessage(content="merhaba")], "duzeltme_denemesi": 0}
        )
        assert sonuc["duzeltme_denemesi"] == 0

    def test_deneme_siniri_sonsuz_dongude_kilitlenmez(self, sahte):
        """Model inatla ayni hatayi yapiyorsa akis sonlanmali.

        Sinir olmasaydi: arac duser -> model yeniden yazar -> yine duser.
        """
        inatci = []
        for i in range(12):
            inatci.append(_arac_cagrisi("describe_column", {"column": "gelir"}, cid=str(i)))
            inatci.append(AIMessage(content=f"Uydurma cevap {i}: 123 TL."))

        sahte(inatci)
        sonuc = build_agent().invoke(
            {"messages": [HumanMessage(content="soru")], "duzeltme_denemesi": 0}
        )

        assert sonuc["duzeltme_denemesi"] == MAKS_DUZELTME
        assert isinstance(sonuc["messages"][-1], AIMessage)

    def test_denemeler_tukenirse_cevap_dayanaksiz_isaretlenir(self, sahte, monkeypatch):
        """Son savunma: duzeltme ise yaramadiysa cevap etiketlenmeli."""
        inatci = []
        for i in range(12):
            inatci.append(_arac_cagrisi("describe_column", {"column": "gelir"}, cid=str(i)))
            inatci.append(AIMessage(content="Ortalama 123 TL."))
        sahte(inatci)

        sonuc = ajan_modulu.ask("soru")

        assert sonuc["dayanaksiz_cevap"] is True
        assert sonuc["cevap"].startswith("[DAYANAKSIZ CEVAP]")
        assert sonuc["duzeltme_denemesi"] == MAKS_DUZELTME
        assert sonuc["arac_ozeti"]["basarili"] == 0

    def test_duzeltme_sonrasi_denetim_kaydi_korunur(self, sahte):
        """Guard baglami duzeltme dongusunde kaybolmamali."""
        sahte(
            [
                _arac_cagrisi("describe_column", {"column": "gelir"}),
                AIMessage(content="Uydurma 5 TL."),
                _arac_cagrisi("describe_column", {"column": "kredi_skoru"}, cid="2"),
                AIMessage(content="Ortalama 1403."),
            ]
        )
        sonuc = ajan_modulu.ask("soru")
        kayit = sonuc["denetim_kaydi"]
        assert any(not e["allowed"] for e in kayit), "reddedilen istek kayda gecmedi"
        assert any(e["allowed"] for e in kayit), "izin verilen istek kayda gecmedi"
        assert sonuc["dayanaksiz_cevap"] is False


class TestSecmeliYenidenDeneme:
    """Gecici hatada tekrar dene, kalici hatada deneme.

    Regresyon: ilk surumde Runnable.with_retry kullaniliyordu ve o yalnizca
    istisna TIPINE gore filtreliyor. Saglayici hem kota hem baglanti hatasini
    ayni tipte sardigi icin kota hatasi da 4 kez deneniyordu. Olculdu: gunluk
    kotasi dolmus bir modelde cagri 36 saniyeden 146 saniyeye cikiyordu.
    """

    @pytest.mark.parametrize(
        "mesaj",
        [
            "429 RESOURCE_EXHAUSTED quota exceeded",
            "400 INVALID_ARGUMENT: Invalid JSON payload",
            "403 PERMISSION_DENIED: API key not valid",
            "404 NOT_FOUND: model bulunamadi",
        ],
    )
    def test_kalici_hatalar_tekrar_denenmez(self, mesaj):
        from src.config import _gecici_mi

        assert _gecici_mi(Exception(mesaj)) is False

    @pytest.mark.parametrize(
        "mesaj",
        [
            "[SSL: INVALID_SESSION_ID] invalid session id",
            "ConnectError: connection refused",
            "503 Service Unavailable",
            "ReadTimeout",
        ],
    )
    def test_gecici_hatalar_tekrar_denenir(self, mesaj):
        from src.config import _gecici_mi

        assert _gecici_mi(Exception(mesaj)) is True

    def test_kalici_hatada_tek_cagri_yapilir(self):
        from src.config import dayanikli

        class Patlak:
            sayac = 0

            def invoke(self, *a, **k):
                Patlak.sayac += 1
                raise Exception("429 RESOURCE_EXHAUSTED")

        with pytest.raises(Exception):
            dayanikli(Patlak(), deneme=4).invoke("x")
        assert Patlak.sayac == 1

    def test_gecici_hatada_sinira_kadar_denenir(self):
        from src.config import dayanikli

        class Patlak:
            sayac = 0

            def invoke(self, *a, **k):
                Patlak.sayac += 1
                raise Exception("SSL error")

        with pytest.raises(Exception):
            dayanikli(Patlak(), deneme=3).invoke("x")
        assert Patlak.sayac == 3

    def test_basarili_cagri_sarmalayicidan_gecer(self):
        from src.config import dayanikli

        class Calisan:
            def invoke(self, x):
                return f"sonuc:{x}"

            def bind_tools(self, _):
                return self

        sarili = dayanikli(Calisan())
        assert sarili.invoke("a") == "sonuc:a"
        # invoke disindaki cagriler sarilan nesneye devredilmeli
        assert sarili.bind_tools([]) is not None


class TestLangChainHatalari:
    """Regresyon: LangChain'in kendi hata mesajlari BASARILI sayiliyordu.

    Sema dogrulamasi basarisiz oldugunda (ornegin how='average') ya da olmayan
    bir arac cagrildiginda LangChain duz metin dondurur - bizim {"hata": ...}
    bicimimizde degil. JSON olarak ayristirilamadigi icin dayanak sayaci bunlari
    basarili sayiyordu; yani agent yalnizca gecersiz cagrilar yapip cevap
    uydurdugunda duzeltme dongusu HIC tetiklenmiyordu.
    """

    def _tm(self, icerik: str):
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=icerik, tool_call_id="1")

    @pytest.mark.parametrize(
        "icerik",
        [
            "Error invoking tool 'group_aggregate' with kwargs {'how': 'average'}",
            "Error: yok_boyle_arac is not a valid tool, try one of [list_columns]",
        ],
    )
    def test_langchain_hatalari_hatali_sayilir(self, icerik):
        from src.agent import _basarili_arac_ciktisi_var_mi

        assert _basarili_arac_ciktisi_var_mi([self._tm(icerik)]) == (0, 1)

    def test_langchain_hatalari_duzeltici_mesaja_girer(self):
        from src.agent import _hata_metinleri

        h = _hata_metinleri([self._tm("Error invoking tool 'x' with kwargs {}")])
        assert len(h) == 1

    def test_mesru_duz_metin_cikti_hata_sayilmaz(self):
        """search_data_dictionary duz metin donduruyor; hata degil."""
        from src.agent import _basarili_arac_ciktisi_var_mi

        assert _basarili_arac_ciktisi_var_mi([self._tm("sorgu sonucu: 3 kolon")]) == (1, 0)

    def test_gecersiz_arac_argumani_graf_dusurmez(self):
        """Sema hatasi ToolMessage'a donusmeli, istisna firlatmamali."""
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode

        from src.tools import ANALYSIS_TOOLS

        g = StateGraph(MessagesState)
        g.add_node("a", ToolNode(ANALYSIS_TOOLS))
        g.add_edge(START, "a")
        g.add_edge("a", END)

        sonuc = g.compile().invoke(
            {"messages": [AIMessage(content="", tool_calls=[
                {"name": "group_aggregate",
                 "args": {"group_by": "il", "metric": "temerrut", "how": "average"},
                 "id": "1", "type": "tool_call"}])]}
        )
        assert isinstance(sonuc["messages"][-1], ToolMessage)
