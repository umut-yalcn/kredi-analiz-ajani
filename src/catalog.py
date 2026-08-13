"""Veri sozlugu uzerinde anlamsal arama (Chroma).

Kullanici "odeme gecmisi kotu olanlar" diye soruyor; veride ise kolon adi
'gecikme_gun_max'. Kolon adlarini prompt'a gomup modelin dogru esleme yapmasini
ummak yerine, veri sozlugunu vektorleyip semantik olarak ariyoruz. Sema buyudukce
(gercek bir kredi burosunda yuzlerce kolon) bu yaklasim olceklenir, prompt'a
gomme yaklasimi olceklenmez.

Embedding API uzerinden alinir; yerel model indirilmez.
"""

from __future__ import annotations

import json
import threading
import time
from functools import lru_cache
from typing import Any

import chromadb
from langchain_core.tools import tool

from .config import CHROMA_PATH, get_embeddings
from .schema import CREDIT_APPLICATION_SCHEMA, Sensitivity

COLLECTION_NAME = "veri_sozlugu"


#: Embedding cagrisi ag kaynakli gecici hatalarla dusebiliyor (gozlemlenen:
#: "SSL: INVALID_SESSION_ID"). Tek bir gecici hata tum agent kosumunu
#: dusurmemeli - ozellikle canli demoda.
EMBEDDING_DENEME = 4
EMBEDDING_BEKLEME = 2.0


class GoogleEmbeddingFunction:
    """Chroma'nin bekledigi embedding arayuzunu LangChain modeline baglar.

    Chroma (>=1.x) belge ve sorgu icin AYRI cagri yapar: belgeler `__call__`
    uzerinden, sorgular `embed_query` uzerinden gider. Yalnizca `__call__`
    tanimlanirsa arama calisma aninda AttributeError ile duser. Ayrim ayrica
    dogru: embedding modeli sorgu ile belgeyi farkli gorevler olarak kodlar.
    """

    def __init__(self) -> None:
        self._model = get_embeddings()

    @staticmethod
    def _metinler(input: Any) -> list[str]:  # noqa: A002 - Chroma imzasi
        if isinstance(input, str):
            return [input]
        return [str(t) for t in input]

    def _dene(self, fn):
        """Gecici ag hatalarina karsi yeniden dener."""
        son_hata: Exception | None = None
        for deneme in range(EMBEDDING_DENEME):
            try:
                return fn()
            except Exception as hata:  # saglayici hatalarini tek tek ayirmiyoruz
                son_hata = hata
                if deneme < EMBEDDING_DENEME - 1:
                    # Ustel geri cekilme: gozlemlenen SSL kesintileri
                    # ~30 saniyede toparliyor, dogrusal bekleme yetmiyor.
                    time.sleep(EMBEDDING_BEKLEME * (2 ** deneme))
        raise RuntimeError(
            f"Embedding {EMBEDDING_DENEME} denemede alinamadi: {son_hata}"
        ) from son_hata

    def __call__(self, input: Any) -> list[list[float]]:  # noqa: A002 - Chroma imzasi
        metinler = self._metinler(input)
        return self._dene(lambda: self._model.embed_documents(metinler))

    def embed_query(self, input: Any) -> list[list[float]]:  # noqa: A002 - Chroma imzasi
        metinler = self._metinler(input)
        return self._dene(lambda: [self._model.embed_query(t) for t in metinler])

    def name(self) -> str:
        return "google-gemini-embedding"


#: chromadb.PersistentClient, sinif duzeyinde paylasilan bir sozluk kullaniyor
#: ve bu sozluk kilitsiz. LangGraph araclari thread havuzunda kosturdugu icin
#: iki thread ayni anda ilk istemciyi olusturmaya kalkinca yaris olusuyor ve
#: KeyError firliyor. lru_cache bunu engellemiyor: onbellek dolmadan once
#: fonksiyon iki kez cagrilabilir. Olusturmayi kilitliyoruz.
_istemci_kilidi = threading.Lock()


@lru_cache(maxsize=1)
def _collection():
    with _istemci_kilidi:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=GoogleEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )


def build_catalog(force: bool = False) -> int:
    """Veri sozlugunu vektor deposuna yazar. Yalnizca analize acik kolonlar indekslenir.

    PII kolonlar indekse hic girmez - modelin varliklarindan haberdar olmasi
    icin bir sebep yok.
    """
    collection = _collection()
    if collection.count() > 0 and not force:
        return collection.count()

    if force and collection.count() > 0:
        collection.delete(where={"analyzable": True})

    columns = [c for c in CREDIT_APPLICATION_SCHEMA if c.analyzable]
    collection.upsert(
        ids=[c.name for c in columns],
        documents=[
            f"{c.name}: {c.description} Veri tipi {c.dtype}."
            + (f" Birim: {c.unit}." if c.unit else "")
            for c in columns
        ],
        metadatas=[
            {
                "name": c.name,
                "dtype": c.dtype,
                "sensitivity": c.sensitivity.value,
                "analyzable": True,
            }
            for c in columns
        ],
    )
    return collection.count()


@tool
def search_data_dictionary(query: str, top_k: int = 4) -> str:
    """Dogal dilde tarif edilen bir kavrama karsilik gelen kolonlari bulur.

    Kolon adini bilmiyorsan once bunu kullan. Ornek: "odeme gecmisi" sorgusu
    gecikme kolonunu, "borclanma duzeyi" sorgusu borc kolonlarini dondurur.

    Args:
        query: Aradigin kavramin dogal dildeki tarifi.
        top_k: Dondurulecek kolon sayisi.
    """
    build_catalog()
    res = _collection().query(query_texts=[query], n_results=max(1, min(top_k, 8)))

    hits: list[dict[str, Any]] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "kolon": meta["name"],
                "tip": meta["dtype"],
                "aciklama": doc,
                "benzerlik": round(1 - float(dist), 4),
            }
        )

    return json.dumps({"sorgu": query, "bulunan_kolonlar": hits}, ensure_ascii=False)


CATALOG_TOOLS = [search_data_dictionary]
