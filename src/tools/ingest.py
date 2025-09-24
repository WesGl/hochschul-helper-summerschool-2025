# src/tools/ingest.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from chromadb import PersistentClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DB_DIR = Path("vectordb")
COLL_NAME = "hka"


EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
EMBEDDING_MODEL_LOCAL = os.getenv("EMBEDDING_MODEL_LOCAL", "sentence-transformers/all-MiniLM-L6-v2")


_model = None


def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
    return _model


def chunk_text(text: str, chunk_size=1000, overlap=150) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return chunks


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def ingest_pdfs(pdf_dir: Path = Path("data/pdfs")):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(DB_DIR))
    coll = client.get_or_create_collection(COLL_NAME, metadata={"hnsw:space": "cosine"})

    embedder = get_embedder()

    docs, ids, metas = [], [], []
    for pdf in pdf_dir.glob("**/*.pdf"):
        text = pdf_to_text(pdf)
        for idx, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{pdf.name}:{idx}")
            metas.append({"source": pdf.name, "chunk": idx})

    if not docs:
        return 0

    vectors = embedder.encode(docs, show_progress_bar=True).tolist()

    def _batched(n: int, size: int):
        for i in range(0, n, size):
            yield i, min(i + size, n)

    BATCH_SIZE = 1000  # has to be < 5461 for hnsw index

    total = len(docs)
    for start, end in _batched(total, BATCH_SIZE):
        coll.add(
            documents=docs[start:end],
            embeddings=vectors[start:end] if vectors is not None else None,
            ids=ids[start:end],
            metadatas=metas[start:end] if metas is not None else None,
        )
    return len(docs)
