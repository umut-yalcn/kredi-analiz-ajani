"""Anahtarinla hangi modellerin gercekten calistigini tespit eder.

Model isimleri ve ucretsiz katman kapsami sik degisiyor. Dokumantasyona ya da
tahmine guvenmek yerine, adaylari tek tek arac cagrisiyla (function calling)
deneyip calisanlari raporluyoruz - ajanin ihtiyaci olan yetenek tam olarak bu.

    python scripts/check_models.py

Anahtarin degerini ekrana basmaz.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from src.config import GOOGLE_FALLBACK_CHAIN  # noqa: E402

load_dotenv()


def _tool_schema() -> list[dict]:
    """Ajanin kullandigina benzer basit bir arac tanimi."""
    return [
        {
            "name": "get_column_stats",
            "description": "Bir kolonun ortalama degerini dondurur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Kolon adi"},
                },
                "required": ["column"],
            },
        }
    ]


def list_available() -> list[str]:
    """Anahtarin erisebildigi, icerik uretebilen modelleri listeler."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    return [
        m.name.removeprefix("models/")
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]


def test_tool_calling(model_id: str) -> tuple[bool, str]:
    """Modelin gercekten arac cagirabildigini dogrular."""
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI

    @tool
    def get_column_stats(column: str) -> str:
        """Bir kolonun ortalama degerini dondurur.

        Args:
            column: Kolon adi.
        """
        return '{"ortalama": 1403}'

    try:
        llm = ChatGoogleGenerativeAI(model=model_id, temperature=0).bind_tools(
            [get_column_stats]
        )
        resp = llm.invoke("kredi_skoru kolonunun ortalamasi nedir? Araci kullan.")
        if resp.tool_calls:
            return True, f"arac cagrisi yapti: {resp.tool_calls[0]['name']}"
        return False, "arac cagirmadi, duz metin dondu"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "429" in msg or "quota" in msg.lower():
            return False, "kota asildi / ucretsiz katmanda kapali"
        if "404" in msg or "not found" in msg.lower():
            return False, "model bulunamadi"
        if "403" in msg or "permission" in msg.lower():
            return False, "erisim izni yok (muhtemelen ucretli katman)"
        return False, msg.split("\n")[0][:110]


EMBEDDING_ADAYLARI = (
    "models/gemini-embedding-001",
    "models/text-embedding-004",
)


def test_embedding(model_id: str) -> tuple[bool, str]:
    """Veri sozlugu aramasi icin embedding modelini dener."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    try:
        vec = GoogleGenerativeAIEmbeddings(model=model_id).embed_query("kredi skoru")
        return True, f"{len(vec)} boyutlu vektor dondu"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "404" in msg or "not found" in msg.lower():
            return False, "model bulunamadi"
        if "429" in msg or "quota" in msg.lower():
            return False, "kota asildi"
        return False, msg.split("\n")[0][:110]


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        print("[HATA] GOOGLE_API_KEY tanimli degil.")
        print("       .env dosyasi olusturup anahtari oraya ekle.")
        print("       Ucretsiz anahtar: https://aistudio.google.com/apikey")
        raise SystemExit(1)

    print("Anahtar tanimli. Modeller sorgulaniyor...\n")

    try:
        available = list_available()
        flash = sorted(m for m in available if "flash" in m and "thinking" not in m)
        print(f"Hesabinda gorunen Flash ailesi modelleri ({len(flash)} adet):")
        for m in flash:
            print(f"  - {m}")
        print()
    except Exception as exc:  # noqa: BLE001
        available = []
        print(f"[UYARI] Model listesi alinamadi: {str(exc)[:120]}")
        print("        Aday listesi uzerinden dogrudan denenecek.\n")

    adaylar = list(GOOGLE_FALLBACK_CHAIN)
    for m in available:
        if "flash" in m and m not in adaylar and "thinking" not in m:
            adaylar.append(m)

    print("Arac cagirma (function calling) testi:\n")
    calisan: list[str] = []
    for model_id in adaylar:
        ok, note = test_tool_calling(model_id)
        print(f"  {'[OK]  ' if ok else '[HATA]'} {model_id:<28} {note}")
        if ok:
            calisan.append(model_id)

    print("\nEmbedding testi (veri sozlugu aramasi icin):\n")
    calisan_embed: list[str] = []
    for model_id in EMBEDDING_ADAYLARI:
        ok, note = test_embedding(model_id)
        print(f"  {'[OK]  ' if ok else '[HATA]'} {model_id:<34} {note}")
        if ok:
            calisan_embed.append(model_id)

    print("\n" + "=" * 60)
    if calisan:
        print(f"Sohbet modeli   : {calisan[0]}")
    else:
        print("Sohbet modeli   : BULUNAMADI")
    if calisan_embed:
        print(f"Embedding modeli: {calisan_embed[0]}")
    else:
        print("Embedding modeli: BULUNAMADI")
    print("=" * 60)

    if calisan:
        print(f"\n.env dosyana ekle:\n    LLM_MODEL={calisan[0]}")
        if calisan_embed:
            print(f"    EMBEDDING_MODEL={calisan_embed[0]}")
    else:
        print("\nHicbir model arac cagrisi yapamadi.")
        print("Kotan dolmus olabilir - birkac dakika bekleyip tekrar dene.")


if __name__ == "__main__":
    main()
