# src/tools/search.py
import os

from tavily import TavilyClient

from ..models import LLM

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEARCH_MODEL = os.getenv("RAG_MODEL", "openai/gpt-4o-mini")
_llm = LLM(SEARCH_MODEL)


client = TavilyClient(api_key=TAVILY_API_KEY)


SYSTEM = "Du fasst Webresultate sachlich zusammen und gibst die wichtigsten Quellen an."


def search_and_answer(query: str):
    res = client.search(query=query, max_results=5)
    # res enthält "results": [{title, url, content, score}, ...]
    blocks = []
    for r in res.get("results", []):
        snippet = r.get("content", "")[:1000]
        blocks.append(f"- {r['title']}\n {r['url']}\n {snippet}")
    context = "\n\n".join(blocks)

    msg = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Web-Kontext:\n{context}\n\nFrage: {query}"},
    ]
    out = _llm.chat(msg)
    cites = [r.get("url") for r in res.get("results", [])]
    return {"answer": out, "citations": cites}
