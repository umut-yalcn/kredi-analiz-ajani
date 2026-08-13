"""Saglayici-bagimsiz model kurulumu.

Gelistirme Google AI Studio'nun ucretsiz katmaninda yapilir, teslim/demo asamasinda
tek satir config degisikligiyle Anthropic'e gecilir. Agent kodu hangi saglayicinin
kullanildigini bilmez.
"""

from __future__ import annotations

import os
import pathlib
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


# Google AI Studio ucretsiz katmani dakikada 5 istekle sinirli (gozlemlenen hata:
# "generate_content_free_tier_requests, limit: 5"). Agent tek bir soruda 7-10
# model cagrisi yapabildigi icin bu limit sorgunun ortasinda 429 ile carpiyor.
# Model cagrilarini kendimiz yavaslatiyoruz: yavas calismak, yarida kalmaktan iyi.
# Ucretli katmanda ISTEK_HIZI_RPM=0 ile kapatilabilir.
VARSAYILAN_RPM = 4


@lru_cache(maxsize=1)
def _hiz_sinirlayici():
    """Dakikadaki istek sinirini uygulayan limitleyici. 0 ise kapali."""
    try:
        rpm = float(os.getenv("ISTEK_HIZI_RPM", "").strip() or VARSAYILAN_RPM)
    except ValueError:
        rpm = VARSAYILAN_RPM

    if rpm <= 0:
        return None

    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=rpm / 60.0,
        check_every_n_seconds=0.5,
        max_bucket_size=1,
    )


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

        kwargs: dict = {"model": model}

        # Gemini 3.x sabit ornekleme kullanir ve temperature'i yok sayar
        # (gecirilirse uyari basar). Yalnizca eski modellerde gonderiyoruz.
        if not model.startswith("gemini-3"):
            kwargs["temperature"] = temperature

        limitleyici = _hiz_sinirlayici()
        if limitleyici is not None:
            kwargs["rate_limiter"] = limitleyici

        return ChatGoogleGenerativeAI(**kwargs)

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


# Yollar proje kokune gore mutlak cozulur. Goreli birakilsaydi kod yalnizca
# proje dizininden calistirildiginda calisirdi; uvicorn veya test kosumu baska
# bir dizinden baslatildiginda veri bulunamazdi.
_KOK = pathlib.Path(__file__).resolve().parents[1]


def _yol(env_adi: str, varsayilan: str) -> str:
    deger = os.getenv(env_adi, "").strip() or varsayilan
    p = pathlib.Path(deger)
    return str(p if p.is_absolute() else _KOK / p)


#: Model cagrilari gecici ag hatalariyla dusebiliyor. Gozlenen desen:
#: tek bir cagri geciyor, cok cagrili ajan kosumu "SSL: INVALID_SESSION_ID"
#: ile duser. Hiz sinirlayici cagrilar arasinda ~15 saniye bekledigi icin
#: TLS oturumu bayatliyor ve yeniden kullanim basarisiz oluyor.
#: Saglayicinin kendi max_retries'i bu baglanti hatasini kapsamiyor.
LLM_DENEME = 4


def dayanikli(runnable, deneme: int = LLM_DENEME):
    """Bir Runnable'i gecici hatalara karsi yeniden denenir hale getirir.

    bind_tools SONRASI uygulanmali; sonuc artik BaseChatModel degil Runnable'dir
    ama LangGraph yalnizca invoke cagirdigi icin sorun olmaz.
    """
    return runnable.with_retry(
        stop_after_attempt=deneme,
        wait_exponential_jitter=True,
    )


DATA_PATH = _yol("DATA_PATH", "data/kredi_basvurulari.csv")
CHROMA_PATH = _yol("CHROMA_PATH", "chroma_store")
