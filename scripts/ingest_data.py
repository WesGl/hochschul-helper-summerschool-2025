# scripts/ingest_data.py
from pathlib import Path
from src.tools.ingest import ingest_pdfs


if __name__ == "__main__":
    n = ingest_pdfs(Path("data/pdfs"))
    print(f"Ingested chunks: {n}")