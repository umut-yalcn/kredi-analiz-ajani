"""LangGraph tabanli analiz agent'i.

Akis:

    soru -> [agent] --arac cagrisi var mi?--> [araclar] -> [agent] -> ... -> [maskele] -> cevap

Agent hangi araci hangi sirayla cagiracagina kendi karar verir; sabit bir analiz
hatti yoktur. Guard katmani her arac cagrisinda devrede oldugu icin bu ozgurluk
guvenlik pahasina gelmez - agent yanlis bir sey isterse arac hata dondurur ve
agent plan degistirir.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .catalog import CATALOG_TOOLS
from .config import get_llm
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

Cevap bicimi:
- Once sonucu soyle. Bulgun ne ise ilk cumlede o olsun.
- Sonra dayanagini ver: hangi metrik, hangi grup, kac satir.
- Kisa tut. Rapor yazmiyorsun, soruyu yanitliyorsun.
- Turkce yaz.
"""


def build_agent():
    """Derlenmis LangGraph akisini dondurur."""
    llm = get_llm().bind_tools(ALL_TOOLS)

    def call_model(state: MessagesState) -> dict[str, Any]:
        messages = state["messages"]
        # Adim siniri: agent dongude kalirsa elindeki bilgiyle sonlandirmasini iste
        if len(messages) > MAX_STEPS * 2:
            messages = messages + [
                HumanMessage(
                    content="Adim sinirina ulasildi. Simdiye kadar topladigin "
                    "bulgularla cevabini yaz, yeni arac cagrisi yapma."
                )
            ]
            return {"messages": [get_llm().invoke([SystemMessage(SYSTEM_PROMPT)] + messages)]}

        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT)] + messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

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


def ask(question: str) -> dict[str, Any]:
    """Bir soruyu ucdan uca calistirir ve cevabi denetim kaydiyla birlikte dondurur."""
    guard = Guard()
    set_guard(guard)

    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    final: AIMessage = result["messages"][-1]
    answer = guard.mask(_extract_text(final.content))

    tool_calls = [
        {"arac": tc["name"], "girdi": tc["args"]}
        for msg in result["messages"]
        if isinstance(msg, AIMessage)
        for tc in (msg.tool_calls or [])
    ]

    return {
        "soru": question,
        "cevap": answer,
        "kullanilan_araclar": tool_calls,
        "denetim_kaydi": guard.audit_trail(),
        "adim_sayisi": len(result["messages"]),
    }
