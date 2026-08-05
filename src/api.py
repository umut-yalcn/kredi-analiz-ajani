"""FastAPI servisi.

Calistirmak icin:
    uvicorn src.api:app --reload

Etkilesimli dokumantasyon: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import ask
from .catalog import build_catalog
from .guard import K_ANONYMITY_THRESHOLD
from .schema import ANALYZABLE_COLUMNS, PII_COLUMNS
from .tools import load_analysis_frame

app = FastAPI(
    title="Agentic Data Analytics",
    description=(
        "Kredi basvuru verisi uzerinde dogal dilde analiz. "
        "Her sorgu kisisel veri korumasi ve k-anonimlik kontrolunden gecer."
    ),
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Veri hakkindaki sorunuz",
        examples=["Hangi meslek grubunda temerrut orani en yuksek?"],
    )


class AskResponse(BaseModel):
    soru: str
    cevap: str
    kullanilan_araclar: list[dict[str, Any]]
    denetim_kaydi: list[dict[str, Any]]
    adim_sayisi: int


@app.on_event("startup")
def _warmup() -> None:
    """Veri sozlugunu onceden indeksle ki ilk istek yavas olmasin."""
    try:
        build_catalog()
    except Exception as exc:  # noqa: BLE001 - servis acilisini bloklamasin
        print(f"[UYARI] Veri sozlugu indekslenemedi: {exc}")


@app.get("/health")
def health() -> dict[str, Any]:
    """Servis ve veri durumu."""
    try:
        n_rows = len(load_analysis_frame())
    except FileNotFoundError:
        n_rows = 0
    return {
        "durum": "ayakta",
        "satir_sayisi": n_rows,
        "analize_acik_kolon_sayisi": len(ANALYZABLE_COLUMNS),
        "korunan_kolon_sayisi": len(PII_COLUMNS),
        "k_anonimlik_esigi": K_ANONYMITY_THRESHOLD,
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    """Analize acik kolonlar. Korunan kolonlarin yalnizca sayisi paylasilir."""
    return {
        "analize_acik": list(ANALYZABLE_COLUMNS),
        "korunan_kolon_sayisi": len(PII_COLUMNS),
        "not": "Kisisel veri iceren kolonlarin adlari da paylasilmaz.",
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> dict[str, Any]:
    """Dogal dilde bir soruyu agent'a iletir."""
    try:
        return ask(req.question)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Veri seti bulunamadi. Once: python scripts/generate_data.py",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analiz basarisiz: {exc}")
