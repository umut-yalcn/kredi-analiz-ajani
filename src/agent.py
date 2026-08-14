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
from functools import lru_cache
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
        mesajlar = state["messages"]
        son = mesajlar[-1]

        # Adim siniri BURADA uygulanir. Onceden yalnizca modele "yeni arac
        # cagirma" deniyordu; model dinlemezse graf arac calistirmaya devam
        # ediyordu. Olculdu: MAX_STEPS=12 iken 40 arac cagrisi yapildi.
        # Sinir artik dilek degil, yonlendirme karari.
        if len(mesajlar) > MAX_STEPS * 2 + 4:
            return END

        if getattr(son, "tool_calls", None):
            return "tools"

        # Agent cevabi yazmak uzere. Arkasinda veri var mi?
        basarili, hatali = _basarili_arac_ciktisi_var_mi(mesajlar)
        deneme = state.get("duzeltme_denemesi", 0)

        if basarili == 0 and deneme < MAKS_DUZELTME:
            # Onceden yalnizca hatali > 0 kosuluna bakiliyordu. Yani agent HIC
            # arac cagirmadan "en riskli grup kamudur" derse ne duzeltmeye
            # gonderiliyor ne de isaretleniyordu; grafik dogrudan END'e
            # gidiyordu. Artik cevabin veri hakkinda bir iddia tasiyip
            # tasimadigina da bakiliyor - selamlama gibi iddiasiz cevaplar
            # engellenmiyor.
            if hatali > 0 or _veri_iddiasi_mi(_extract_text(son.content)):
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


#: Cevaptan sayi cikarmak icin. Binlik ayraci ve ondalik virgul/nokta kabul eder.
_SAYI_YAKALA = re.compile(r"(?<![A-Za-z0-9])-?\d[\d.,]*")
#: Lookbehind sart: "AUTH-9204" gibi kontrol kimliklerinde tire eksi
#: isareti sanilip sayi "-9204" olarak okunuyordu. Dayanak kontrolu o
#: uydurma negatif degeri arac ciktisinda da bulup eslestirdigi icin
#: koruma zayifliyordu.


def _sayilari_cikar(metin: str) -> list[str]:
    """Metindeki sayilari normallestirerek dondurur.

    Bicim farki (1.234,56 / 1234.56 / %12,1) karsilastirmayi bozmasin diye
    ayraclar atilip ondalik noktaya cevrilir.
    """
    sonuc = []
    for ham in _SAYI_YAKALA.findall(metin):
        t = ham.strip(".,")
        if not t:
            continue
        # Turkce bicim: son ayrac ondalik ayracidir
        if "," in t and "." in t:
            t = t.replace(".", "").replace(",", ".") if t.rindex(",") > t.rindex(".")                 else t.replace(",", "")
        elif "," in t:
            t = t.replace(",", ".") if len(t.split(",")[-1]) != 3 else t.replace(",", "")
        try:
            d = float(t)
        except ValueError:
            # "1.403.421" gibi noktali binlik yazim float()'ta patliyor ve sayi
            # SESSIZCE dusuyordu: model rakami boyle yazdiginda dayanak
            # kontrolunden hic gecmiyordu. Noktalar binlik ayraci olarak
            # yorumlanip yeniden deneniyor.
            if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", t):
                try:
                    d = float(t.replace(".", ""))
                except ValueError:
                    continue
            else:
                continue
        sonuc.append(f"{d:.4f}".rstrip("0").rstrip("."))
    return sonuc


def _dayanakli_degerler(arac_ciktilari: list[str]) -> list[float]:
    """Arac ciktilarinda gecen sayilar.

    x100 KARSILIKLARI ARTIK KOSULSUZ EKLENMIYOR. Her sayinin 100 kati da
    dayanak sayilinca, arac ciktisindaki herhangi bir sayinin 100 kati cevapta
    serbestce uydurulabiliyordu: "5000 satir" bilgisi uydurma "500.000 TL"
    rakamini dayanakli kiliyordu (olculdu). Oran -> yuzde donusumu artik
    yalnizca cevapta % isareti varken ve TEK YONLU uygulaniyor.
    """
    degerler: list[float] = []
    for c in arac_ciktilari:
        for s in _sayilari_cikar(c):
            degerler.append(float(s))
    return degerler


#: Cevapta yuzde olarak yazilmis sayilari yakalar: "%68,11" ya da "68,11%".
_YUZDE_DESENI = re.compile(r"%\s*(\d[\d.,]*)|(\d[\d.,]*)\s*%")


def _yuzde_sayilari(cevap: str) -> set[str]:
    """Cevapta yuzde isaretiyle yazilmis sayilarin normallestirilmis hali."""
    bulunan: set[str] = set()
    for a, b in _YUZDE_DESENI.findall(cevap or ""):
        for ham in (a, b):
            if ham:
                bulunan.update(_sayilari_cikar(ham))
    return bulunan


#: "1,4 milyon" gibi ifadelerde carpani cozmek icin. Sadece Turkce yazimlar;
#: modelin cevabi Turkce uretmesi sistem prompt'unda zorunlu.
_CARPANLAR = {"bin": 1e3, "milyon": 1e6, "milyar": 1e9}
_CARPANLI_SAYI = re.compile(
    r"(-?\d[\d.,]*)\s*(bin|milyon|milyar)\b", re.IGNORECASE
)


def _carpanli_sayilar(metin: str) -> list[tuple[str, float]]:
    """'1,4 milyon' -> [('1,4 milyon', 1400000.0)] seklinde cozer."""
    sonuc: list[tuple[str, float]] = []
    for eslesme in _CARPANLI_SAYI.finditer(metin):
        ham, kelime = eslesme.group(1), eslesme.group(2)
        sayilar = _sayilari_cikar(ham)
        if sayilar:
            sonuc.append((eslesme.group(0), float(sayilar[0]) * _CARPANLAR[kelime.lower()]))
    return sonuc


def _dogrulanmayan_sayilar(cevap: str, arac_ciktilari: list[str]) -> list[str]:
    """Cevapta gecip arac ciktilarinda BULUNMAYAN sayilar.

    Bagimsiz denetim su acigi gosterdi: dayanak kontrolu yalnizca "basarili bir
    arac cagrisi var mi" diye bakiyordu. Agent list_columns cagirip ardindan
    "temerrut orani %98,7" dediginde cevap dayanakli sayiliyordu - alakasiz tek
    bir basarili cagri, cevaptaki TUM sayilara sinirsiz dayanak sagliyordu.

    Kontrol deterministik: model cagrisi yok, kota yemiyor, keyfi yanlis pozitif
    uretmiyor. YUVARLAMAYA TOLERANSLI - agent 1403.42'yi "1403" diye yazabilir;
    cevaptaki sayinin ondalik hassasiyetinde eslesme araniyor.

    Turetilmis degerler (agent'in iki sayidan hesapladigi oran) dogal olarak
    eslesmeyebilir; bu yuzden ENGELLEYICI degil RAPORLAYICI.
    """
    if not cevap:
        return []

    dayanak = _dayanakli_degerler(arac_ciktilari)
    dogrulanmayan = []

    # Carpanli ifadeler ONCE ayikleniyor: "1,4 milyon" yazan bir cevapta duz
    # tarama yalnizca "1,4" goruyor, o da 10 esiginin altinda kaldigi icin hic
    # denetlenmiyordu - modelin buyuk rakamlari boyle yazmasi olagan oldugundan
    # uydurma kontrolunde kor nokta olusuyordu.
    kalan = cevap
    for etiket, deger in _carpanli_sayilar(cevap):
        kalan = kalan.replace(etiket, " ", 1)
        # Ifade zaten yuvarlama: 1.403.421 -> "1,4 milyon". Bu yuzden esleme
        # binde bes bagil toleransla araniyor, tam esitlikle degil.
        tolerans = max(1.0, abs(deger) * 0.005)
        if any(abs(d - deger) <= tolerans for d in dayanak):
            continue
        dogrulanmayan.append(etiket)

    yuzdeler = _yuzde_sayilari(cevap)
    for ham in _sayilari_cikar(kalan):
        deger = float(ham)
        # ONCEDEN mutlak degeri 10'dan kucuk her sayi ATLANIYORDU. Kredi
        # riskinde raporlanan buyukluklerin cogu bu araliktadir: temerrut
        # oranlari (%7,3), korelasyon katsayilari (0,42), ortalama aktif
        # kredi sayisi (1,8). Tamami kucuk sayilardan olusan uydurma bir
        # cevap hicbir kontrole takilmiyordu (olculdu). Esik kaldirildi;
        # gurultuyu bastirmak icin BAGIL tolerans kullaniliyor.
        basamak = len(ham.split(".")[1]) if "." in ham else 0
        # round() ile tam esitlik ASIMETRIKTI: arac 1403.9 dondurdugunde
        # cevaptaki "1404" (yuvarlama) geciyor, "1403" (kirpma) uydurma
        # damgasi yiyordu. Ikisi de mesru yazim; son basamak genisliginde
        # tolerans araniyor.
        # Tolerans SON BASAMAK genisliginde. Bagil tolerans denendi ve
        # geri alindi: %0.5 bagil pay, 1.400.000 dayanagina karsi uydurma
        # 1.403.421'i kabul ediyordu - buyuk sayilarda fazla comert.
        tolerans = 10.0 ** (-basamak)
        if any(abs(d - deger) < tolerans for d in dayanak):
            continue
        # Oran -> yuzde: arac 0.6811 dondurup cevap "%68,11" diyorsa bu
        # mesru. Yalnizca % isareti VARKEN ve tek yonlu.
        if ham in yuzdeler and any(
            abs(d * 100 - deger) < tolerans for d in dayanak
        ):
            continue
        dogrulanmayan.append(ham)
    return dogrulanmayan


#: Cevabin veriye dair bir iddia tasiyip tasimadigini anlamak icin kullanilan
#: alan sozlugu. Selamlama engellenmemeli, ama "en riskli grup kamudur" gibi
#: rakamsiz bir veri iddiasi da dayanaksiz kalmamali.
@lru_cache(maxsize=1)
def _alan_sozlugu() -> tuple[str, ...]:
    """Kolon adlari + kategorik degerler. Veri iddiasi tespitinde kullanilir."""
    from .schema import ANALYZABLE_COLUMNS, CATEGORICAL_COLUMNS
    from .tools import load_analysis_frame

    terimler = {k.replace("_", " ") for k in ANALYZABLE_COLUMNS}
    terimler |= set(ANALYZABLE_COLUMNS)
    try:
        df = load_analysis_frame()
        for kol in CATEGORICAL_COLUMNS:
            terimler |= {str(v).lower() for v in df[kol].unique()}
    except Exception:  # veri yoksa kolon adlariyla yetin
        pass
    # Cok kisa terimler ("il") kelime icinde yanlis eslesir; sinir gerekiyor.
    return tuple(sorted(terimler))


def _veri_iddiasi_mi(cevap: str) -> bool:
    """Cevap, veri hakkinda bir iddia iceriyor mu?

    Selamlama engellenmemeli; ama "en riskli grup kamudur" gibi RAKAMSIZ bir
    veri iddiasi da dayanaksiz kalmamali. Terimler KELIME SINIRIYLA aranir -
    aksi halde "il" kolonu "olabilirim" icinde eslesiyordu.
    """
    if _SAYI_DESENI.search(cevap):
        return True
    kucuk = cevap.lower()
    for t in _alan_sozlugu():
        kacis = re.escape(t)
        # Turkce sondan eklemeli: "kamu" -> "kamudur", "meslek grubu" -> "...na".
        # Uzun terimlerde son eke izin veriyoruz; kisa terimlerde (il, yas)
        # vermiyoruz, yoksa "il" kelimesi "ilgili" icinde eslesir.
        if len(t) >= 4:
            desen = rf"(?<![a-z0-9ğüşıöç]){kacis}"
        else:
            desen = rf"(?<![a-z0-9ğüşıöç]){kacis}(?![a-z0-9ğüşıöç])"
        if re.search(desen, kucuk):
            return True
    return False


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


def _maskeli(deger: Any, guard: Guard) -> Any:
    """Ic ice yapilardaki tum metinleri maskeler.

    mask() YALNIZCA cevap alanina uygulaniyordu. Oysa guard, bilinmeyen kolon
    adini denetim kaydina ve hata metnine BIREBIR yaziyor; model (ya da prompt
    enjeksiyonuyla yonlendirilen model) kullanicinin sorusundaki bir deseni
    arac argumanina koydugunda o desen maskelenmeden API cevabina ve CLI
    ciktisina ulasiyordu. Olculdu: arac girdisinde 11 haneli desen maskesiz
    dondu. Son savunma hatti tek bir alani degil, DISARI CIKAN HER SEYI
    kapsamali.
    """
    if isinstance(deger, str):
        return guard.mask(deger)
    if isinstance(deger, dict):
        return {k: _maskeli(v, guard) for k, v in deger.items()}
    if isinstance(deger, (list, tuple)):
        return [_maskeli(v, guard) for v in deger]
    return deger


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
        {"arac": tc["name"], "girdi": _maskeli(tc["args"], guard)}
        for msg in result["messages"]
        if isinstance(msg, AIMessage)
        for tc in (msg.tool_calls or [])
    ]

    basarili, hatali = _basarili_arac_ciktisi_var_mi(result["messages"])

    # Kod yolunda dayanak kontrolu. Modelin iyi niyetine guvenmez.
    #
    # Gozlenen ilk hata: agent olmayan bir kolon adiyla arac cagirdi, guard
    # reddetti, agent duzeltmek yerine "1500 satir, 27.500 TL" diye bir cevap
    # uydurdu. Gercekte 4 satir vardi.
    #
    # Bagimsiz denetimde ikinci bir acik cikti: kontrol yalnizca "basarili bir
    # arac cagrisi var mi" diye bakiyordu. Agent list_columns cagirip ardindan
    # "temerrut orani %98,7" dediginde cevap DAYANAKLI sayiliyordu - alakasiz
    # tek bir basarili cagri, cevaptaki tum sayilara sinirsiz dayanak
    # sagliyordu. Artik cevaptaki sayilar arac ciktilariyla karsilastiriliyor.
    arac_ciktilari = [
        str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)
    ]
    dogrulanmayan = _dogrulanmayan_sayilar(answer, arac_ciktilari)

    cevap_sayilari = _sayilari_cikar(answer)
    hicbiri_dayanakli_degil = bool(cevap_sayilari) and len(dogrulanmayan) == len(cevap_sayilari)

    dayanaksiz = (basarili == 0 and _veri_iddiasi_mi(answer)) or hicbiri_dayanakli_degil
    if dayanaksiz:
        guard.note(
            "cevap_dayanagi",
            (),
            f"Dayanaksiz cevap: {basarili} basarili arac ciktisi, "
            f"{len(dogrulanmayan)} dogrulanmayan sayi",
        )
        answer = f"{DESTEKSIZ_CEVAP_UYARISI}\n\n{answer}"
    elif dogrulanmayan:
        # Kismi uydurma: bazi sayilar dayanakli, bazilari degil. Engellemiyoruz
        # (turetilmis degerler dogal olarak eslesmeyebilir) ama raporluyoruz.
        guard.note(
            "cevap_dayanagi",
            (),
            f"Arac ciktilarinda bulunmayan sayilar: {', '.join(dogrulanmayan[:5])}",
        )

    return {
        "soru": question,
        "cevap": answer,
        "kullanilan_araclar": tool_calls,
        "denetim_kaydi": _maskeli(guard.audit_trail(), guard),
        "adim_sayisi": len(result["messages"]),
        "arac_ozeti": {"basarili": basarili, "hatali": hatali},
        "duzeltme_denemesi": result.get("duzeltme_denemesi", 0),
        "dayanaksiz_cevap": dayanaksiz,
        "dogrulanmayan_sayilar": dogrulanmayan,
    }
