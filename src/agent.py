"""LangGraph tabanli analiz agent'i.

Akis:

    soru -> [agent] --arac cagrisi var mi?--> [araclar] -> [agent] -> ... -> [maskele] -> cevap

Agent hangi araci hangi sirayla cagiracagina kendi karar verir; sabit bir analiz
hatti yoktur. Guard katmani her arac cagrisinda devrede oldugu icin bu ozgurluk
guvenlik pahasina gelmez - agent yanlis bir sey isterse arac hata dondurur ve
agent plan degistirir.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .catalog import CATALOG_TOOLS
from .config import dayanikli, get_llm
from .guard import Guard, K_ANONYMITY_THRESHOLD
from .tools import ANALYSIS_TOOLS, set_guard

ALL_TOOLS = CATALOG_TOOLS + ANALYSIS_TOOLS

MAX_STEPS = 12

SYSTEM_PROMPT = f"""Sen bir kredi burosu veri analistisin. Kredi basvuru verisi uzerinde
calisiyor, sorulari veriye dayanarak yanitliyorsun.

Calisma bicimin:
- Kolon adini bilmiyorsan once search_data_dictionary ile ara. Kolon adi uydurma.
- Bir iddiada bulunmadan once onu bir arac cagrisiyla dogrula. Veriye bakmadan
  sayi soyleme.
- Birden fazla arac cagrisi gerekiyorsa yap; tek bir cagriyla yetinme zorunlulugun yok.
- Bulgular birbiriyle celisiyorsa bunu gizleme, celiskiyi soyle.

Veri erisimi hakkinda bilmen gerekenler:
- Kisisel veri iceren kolonlar (ad, kimlik numarasi, telefon, e-posta) analize
  kapalidir ve senin erisimine hic acilmaz. Bunlari sorma, denemeye calisma.
- {K_ANONYMITY_THRESHOLD} satirdan az veriye dayanan sonuclar dondurulmez. Bir arac bu
  gerekceyle hata donduruyorsa filtreyi genislet.
- Bir arac hata dondurdugunde bunu kullaniciya oldugu gibi aktarma; ne yapmaya
  calistigini ve neden yapamadigini kendi cumlelerinle acikla.
- Bir arac "Bilinmeyen kolon" hatasi donduruyorsa kolon adini YANLIS yazmissindir.
  list_columns ile dogru adi al ve TEKRAR CAGIR. Cevap uydurma.
- Elinde basarili bir arac ciktisi YOKSA sayi veremezsin. Bu durumda tek dogru
  cevap, soruyu neden yanitlayamadigini soylemektir. Tahmini bir sayi vermek,
  cevap vermemekten cok daha kotudur.

Cevap bicimi:
- Once sonucu soyle. Bulgun ne ise ilk cumlede o olsun.
- Sonra dayanagini ver: hangi metrik, hangi grup, kac satir.
- Kisa tut. Rapor yazmiyorsun, soruyu yanitliyorsun.
- Turkce yaz.
"""


#: Agent cevabi uydurmaya kalktiginda kac kez geri gonderilecegi.
#: Sinir SART: yoksa arac duser -> model yeniden yazar -> yine duser dongusu olusur.
MAKS_DUZELTME = 2

#: Hata tipine gore ne yapmasi gerektigini SOYLEYEN yonlendirmeler.
#: Genel bir "hatani duzelt" mesaji, somut bir talimat kadar ise yaramiyor.
_DUZELTICI_YONLENDIRME: tuple[tuple[str, str], ...] = (
    (
        "Bilinmeyen kolon",
        "Kolon adini yanlis yazdin. Once list_columns cagir, dogru adi oradan al, "
        "sonra analizi ayni araci dogru kolon adiyla TEKRAR cagir.",
    ),
    (
        "kisisel veri",
        "Bu kolon analize kapali ve hicbir kosulda acilmayacak. Bu veriyle "
        "yanit uretmeye calisma; soruyu neden yanitlayamadigini acikla.",
    ),
    (
        "satir donduruyor",
        "Filtren cok dar kaldi. Esigi genislet ve segment_stats'i tekrar cagir; "
        "genisletemiyorsan bu segment hakkinda sonuc verilemeyecegini soyle.",
    ),
    (
        "gruplama icin uygun degil",
        "Gruplama icin kategorik bir kolon secmelisin. list_columns ile bak ve "
        "group_aggregate'i uygun bir kolonla tekrar cagir.",
    ),
    (
        "sayisal bir kolon degil",
        "Korelasyon yalnizca sayisal kolonlar arasinda hesaplanir. "
        "list_columns ile sayisal kolonlari gor ve tekrar dene.",
    ),
)


class AnalizDurumu(MessagesState):
    """Mesajlara ek olarak kac kez duzeltmeye gonderildigini tasir."""

    duzeltme_denemesi: int


def _hata_metinleri(mesajlar: list[Any]) -> list[str]:
    """Arac mesajlarindaki hata aciklamalarini toplar.

    Hem bizim {"hata": ...} bicimimizi hem LangChain'in duz metin hatalarini
    (sema dogrulamasi, olmayan arac) topluyor; duzeltici mesaj ikisini de
    agent'a geri gosterebilsin.
    """
    hatalar = []
    for m in mesajlar:
        if not isinstance(m, ToolMessage):
            continue
        icerik = str(m.content)
        if icerik.startswith(_LANGCHAIN_HATA_ONEKLERI):
            hatalar.append(icerik[:200])
            continue
        try:
            veri = json.loads(icerik)
        except json.JSONDecodeError:
            continue
        if isinstance(veri, dict) and "hata" in veri:
            hatalar.append(str(veri["hata"]))
    return hatalar


def _duzeltici_mesaj(hatalar: list[str]) -> str:
    """Hata tipine gore somut bir yonlendirme uretir."""
    yonergeler = []
    for hata in hatalar:
        for anahtar, yonerge in _DUZELTICI_YONLENDIRME:
            if anahtar in hata and yonerge not in yonergeler:
                yonergeler.append(yonerge)

    if not yonergeler:
        yonergeler.append(
            "Cagrini gozden gecir ve duzelterek tekrar dene; gerekirse once "
            "list_columns ile mevcut kolonlara bak."
        )

    return (
        "DUR. Hicbir arac cagrin basarili sonuc dondurmedi, dolayisiyla elinde "
        "hicbir veri yok. Bu durumda sayi veremezsin - verdigin her sayi "
        "uydurma olur.\n\n"
        "Alinan hatalar:\n"
        + "\n".join(f"  - {h}" for h in hatalar[-3:])
        + "\n\nSimdi yapman gereken:\n"
        + "\n".join(f"  - {y}" for y in yonergeler)
        + "\n\nDuzeltemiyorsan cevap uydurma; soruyu neden yanitlayamadigini soyle."
    )


def build_agent():
    """Derlenmis LangGraph akisini dondurur."""
    llm = dayanikli(get_llm().bind_tools(ALL_TOOLS))

    def call_model(state: AnalizDurumu) -> dict[str, Any]:
        messages = state["messages"]
        # Adim siniri: agent dongude kalirsa elindeki bilgiyle sonlandirmasini iste
        if len(messages) > MAX_STEPS * 2:
            messages = messages + [
                HumanMessage(
                    content="Adim sinirina ulasildi. Simdiye kadar topladigin "
                    "bulgularla cevabini yaz, yeni arac cagrisi yapma."
                )
            ]
            return {"messages": [dayanikli(get_llm()).invoke([SystemMessage(SYSTEM_PROMPT)] + messages)]}

        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT)] + messages)]}

    def duzeltmeye_gonder(state: AnalizDurumu) -> dict[str, Any]:
        """Agent'i, hatayi duzeltip tekrar denemeye zorlar.

        Bu ICSEL bir oz-elestiri degil: modele kendi cevabini degerlendirmesini
        soylemiyoruz. Guard'in urettigi somut, deterministik hata mesajini geri
        veriyoruz. Arastirma bu ayrimda net - modeller dis geri bildirim
        olmadan kendi hatalarini duzeltemiyor, dis geri bildirimle (derleyici,
        arac, dogrulayici) duzeltebiliyor.
        """
        hatalar = _hata_metinleri(state["messages"])
        return {
            "messages": [HumanMessage(content=_duzeltici_mesaj(hatalar))],
            "duzeltme_denemesi": state.get("duzeltme_denemesi", 0) + 1,
        }

    def yonlendir(state: AnalizDurumu) -> str:
        son = state["messages"][-1]
        if getattr(son, "tool_calls", None):
            return "tools"

        # Agent cevabi yazmak uzere. Arkasinda veri var mi?
        basarili, hatali = _basarili_arac_ciktisi_var_mi(state["messages"])
        deneme = state.get("duzeltme_denemesi", 0)
        if basarili == 0 and hatali > 0 and deneme < MAKS_DUZELTME:
            return "duzeltme"
        return END

    graph = StateGraph(AnalizDurumu)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("duzeltme", duzeltmeye_gonder)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", yonlendir, {"tools": "tools", "duzeltme": "duzeltme", END: END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("duzeltme", "agent")

    return graph.compile()


def _extract_text(content: Any) -> str:
    """Mesaj icerigini duz metne cevirir.

    Saglayicilar icerigi farkli bicimlerde donduruyor: bazilari duz string,
    Gemini 3.x ise blok listesi (her blok {'type': 'text', 'text': ...} ve
    yaninda imza gibi ek alanlar). Ajan kodu bu farki gormemeli.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        if parts:
            return "\n".join(p for p in parts if p)

    return str(content)


DESTEKSIZ_CEVAP_UYARISI = (
    "[DAYANAKSIZ CEVAP] Bu soruya hicbir arac cagrisi basarili sonuc dondurmedi. "
    "Asagidaki metin veriye dayanmiyor; icindeki sayilara guvenilmemelidir."
)

_SAYI_DESENI = re.compile(r"\d")


#: LangChain, sema dogrulamasi basarisiz oldugunda ya da olmayan bir arac
#: cagrildiginda kendi hata metnini duz string olarak dondurur - bizim
#: {"hata": ...} bicimimizde degil. Bunlar JSON olarak ayristirilamadigi icin
#: onceden BASARILI sayiliyordu; yani agent yalnizca gecersiz cagrilar yapip
#: cevap uydurdugunda dayanak kontrolu devreye girmiyordu.
_LANGCHAIN_HATA_ONEKLERI = ("Error invoking tool", "Error:")


def _arac_ciktisi_hata_mi(icerik: str) -> bool:
    """Bir arac mesaji hata mi bildiriyor?

    Iki kaynak var: bizim araclarimizin dondurdugu {"hata": ...} JSON'u ve
    LangChain'in kendi urettigi duz metin hatalari.
    """
    if icerik.startswith(_LANGCHAIN_HATA_ONEKLERI):
        return True
    try:
        veri = json.loads(icerik)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(veri, dict) and "hata" in veri


def _basarili_arac_ciktisi_var_mi(mesajlar: list[Any]) -> tuple[int, int]:
    """Kac arac cagrisinin basarili, kacinin hata dondurdugunu sayar.

    Araclar hatalarini {"hata": ...} olarak dondurur; istisna firlatmazlar
    (agent'in plan degistirebilmesi icin). Bu, hatanin sessizce yutulabilmesi
    anlamina da geliyor - agent hatayi gorup duzeltmek yerine cevabi
    uydurabilir. Bu fonksiyon o durumu tespit edilebilir kilar.
    """
    basarili = hatali = 0
    for m in mesajlar:
        if not isinstance(m, ToolMessage):
            continue
        if _arac_ciktisi_hata_mi(str(m.content)):
            hatali += 1
        else:
            basarili += 1
    return basarili, hatali


def ask(question: str) -> dict[str, Any]:
    """Bir soruyu ucdan uca calistirir ve cevabi denetim kaydiyla birlikte dondurur."""
    guard = Guard()
    set_guard(guard)

    agent = build_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)], "duzeltme_denemesi": 0}
    )

    final: AIMessage = result["messages"][-1]
    answer = guard.mask(_extract_text(final.content))

    tool_calls = [
        {"arac": tc["name"], "girdi": tc["args"]}
        for msg in result["messages"]
        if isinstance(msg, AIMessage)
        for tc in (msg.tool_calls or [])
    ]

    basarili, hatali = _basarili_arac_ciktisi_var_mi(result["messages"])

    # Kod yolunda dayanak kontrolu. Gozlenen gercek hata: agent olmayan bir
    # kolon adiyla arac cagirdi, guard reddetti, agent duzeltmek yerine
    # "1500 satir, 27.500 TL" diye bir cevap uydurdu. Gercekte 4 satir vardi
    # ve dogru cagri k esiginden reddedilecekti. Bu kontrol modelin iyi
    # niyetine guvenmez: basarili arac ciktisi yoksa ve cevapta sayi varsa
    # cevabin dayanaksiz oldugu kullaniciya soylenir.
    dayanaksiz = basarili == 0 and bool(_SAYI_DESENI.search(answer))
    if dayanaksiz:
        guard.note("cevap_dayanagi", (), f"Basarili arac ciktisi yok ({hatali} hata), cevapta sayi var")
        answer = f"{DESTEKSIZ_CEVAP_UYARISI}\n\n{answer}"

    return {
        "soru": question,
        "cevap": answer,
        "kullanilan_araclar": tool_calls,
        "denetim_kaydi": guard.audit_trail(),
        "adim_sayisi": len(result["messages"]),
        "arac_ozeti": {"basarili": basarili, "hatali": hatali},
        "duzeltme_denemesi": result.get("duzeltme_denemesi", 0),
        "dayanaksiz_cevap": dayanaksiz,
    }
