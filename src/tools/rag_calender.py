from __future__ import annotations

import os
from datetime import datetime

from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.models import LLM

from .ingest import get_embedder

RAG_MODEL = os.getenv("RAG_MODEL", "deepseek/deepseek-chat-v3.1:free")
CHROMA_COLLECTION_NAME = "timetable_test"
CHROMA_PERSIST_DIR = "./vectordb"


_llm = LLM(RAG_MODEL)
today = datetime.now().strftime("%d.%m.%Y")
SYSTEM_PROMPT = """Du bist ein Assistent für die Stundenpläne der Hochschule. 
        Deine Aufgabe ist es, Fragen zu Vorlesungen, Übungen und Räumen anhand 
        der bereitgestellten Dokumenten-Auszüge (Kontext) zu beantworten.
        Datum ist dabei irrelevant, ausschießlich Wochentag, Uhrzeit, Raum und Fach sind relevant.
        Kontext ist immer für nächste Woche.

        Richtlinien:
        - Nutze NUR die Informationen aus dem bereitgestellten Kontext. 
        - Wenn eine Information nicht im Kontext enthalten ist, sage klar: 
        "Dazu liegen mir keine Informationen vor."
        - Formuliere deine Antwort knapp, präzise und in vollständigen deutschen Sätzen.
        - Gib immer an, zu welchem Fach, Wochentag, Uhrzeit und Raum sich deine Antwort bezieht, 
        wenn diese Daten verfügbar sind.
        - Falls mehrere passende Termine im Kontext stehen, liste sie alle übersichtlich auf.
        - Zitiere die Quelle mit dem Dateinamen (`source_file`), wenn möglich.
        - Antworte niemals mit frei erfundenen Daten.

        Dein Ziel ist es, den Benutzer zuverlässig durch den Stundenplan zu leiten."""
USER_PROMPT = """Hier sind relevante Dokumenten-Auszüge (Kontext) aus den Stundenplänen.
            Das Datum ist irrelevant, nur Wochentag, Uhrzeit, Raum und Fach sind relevant.
            Alle Termine beziehen sich auf die nächste Woche. Heute ist {today}.:

            {context}

            Frage des Nutzers:
            {question}

            Bitte beantworte die Frage ausschließlich auf Grundlage des Kontexts. 
            Füge am Ende deiner Antwort IMMER die Quelle(n) aus den Metadaten (`source_file`) hinzu."""


def make_retriever(faculty=None, major=None, semester=None, k=5):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PERSIST_DIR, collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings)
    # create metadata filter dict if provided
    metadata_filter = {}
    if faculty:
        metadata_filter["faculty"] = faculty
    if major:
        metadata_filter["major"] = major
    if semester:
        metadata_filter["semester"] = semester

    retriever = db.as_retriever(search_kwargs={"k": k})
    # The Chroma retriever supports passing filter args at query time; we'll show how below.
    return db, retriever, metadata_filter


def retrieve(query: str, k: int = 6, faculty=None, major=None, semester=None):
    db, retriever, meta_filter = make_retriever(faculty, major, semester, k)
    docs = db.similarity_search(query, k=6)
    return docs


def answer(query: str):
    print(f"TOOL Calendar RAG answer was called")
    hits = retrieve(query)
    context = "\n\n---\n\n".join([d.page_content for d in hits])
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(context=context, question=query, today=today)},
    ]
    out = _llm.chat(msg)

    # naive confidence estimation: length & number of hits
    conf = min(0.95, 0.3 + 0.1 * len(hits)) if hits else 0.2
    # cites = [f"{m['source']}#{m['chunk']}" for _, m in hits]
    cites = [d.metadata.get("source_file", "unknown") for d in hits if hasattr(d, "metadata")]
    print(f"TOOL Calendar RAG answer finished")
    return out, conf, cites
