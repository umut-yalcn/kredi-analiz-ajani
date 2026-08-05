"""Saglayici-bagimsiz model kurulumu.

Gelistirme Google AI Studio'nun ucretsiz katmaninda yapilir, teslim/demo asamasinda
tek satir config degisikligiyle Anthropic'e gecilir. Agent kodu hangi saglayicinin
kullanildigini bilmez.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

# Ucretsiz katmanda Flash ve Flash-Lite aileleri aciktir (Pro modelleri Nisan 2026'da
# ucretli katmana tasindi). Ajan, hangi araci ne zaman cagiracagina kendi karar verdigi
# icin Flash-Lite yerine Flash tercih edildi - arac secimi muhakeme isi.
# Hesabinda hangilerinin acik oldugunu gormek icin: python scripts/check_models.py
DEFAULT_MODELS = {
    "google": "gemini-3.6-flash",
    "anthropic": "claude-sonnet-5",
}

# check_models.py bir model erisilemezse sirayla bunlari dener.
GOOGLE_FALLBACK_CHAIN = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
)


class ConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Ortam degiskenlerine gore yapilandirilmis sohbet modelini dondurur."""
    provider = os.getenv("LLM_PROVIDER", "google").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS.get(provider, "")

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ConfigError(
                "GOOGLE_API_KEY tanimli degil. https://aistudio.google.com/apikey "
                "adresinden ucretsiz anahtar alip .env dosyasina ekle."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Gemini 3.x sabit ornekleme kullanir ve temperature'i yok sayar
        # (gecirilirse uyari basar). Yalnizca eski modellerde gonderiyoruz.
        if model.startswith("gemini-3"):
            return ChatGoogleGenerativeAI(model=model)
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ConfigError(
                "ANTHROPIC_API_KEY tanimli degil. https://console.anthropic.com "
                "adresinden anahtar alip .env dosyasina ekle."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, max_tokens=4096)

    raise ConfigError(
        f"Bilinmeyen LLM_PROVIDER: '{provider}'. Desteklenenler: {', '.join(DEFAULT_MODELS)}"
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Veri sozlugu aramasi icin embedding modeli.

    Yerel embedding modeli calistirmiyoruz; API uzerinden gidiyoruz ki
    kurulum agirligi ve donanim ihtiyaci dusuk kalsin.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ConfigError(
            "Embedding icin GOOGLE_API_KEY gerekiyor. "
            "https://aistudio.google.com/apikey adresinden ucretsiz alinabilir."
        )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = os.getenv("EMBEDDING_MODEL", "").strip() or "models/gemini-embedding-001"
    return GoogleGenerativeAIEmbeddings(model=model)


DATA_PATH = os.getenv("DATA_PATH", "data/kredi_basvurulari.csv")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_store")
