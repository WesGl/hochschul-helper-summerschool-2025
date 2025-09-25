import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === Config ===
HTML_DIR = Path("./data/timetables_html")
CHROMA_PERSIST_DIR = "./vectordb"
CHROMA_COLLECTION_NAME = "timetables"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# regex for filename metadata extraction
# Example filename patterns:
#   "Architecture_and_Civil_Engineering-Architektur_(B)-ARTB.1.html"
#   "AB_Architektur_sem1_2025.html"
# Adjust the pattern to your filenames.
FILENAME_META_RE = re.compile(
    r"(?P<faculty>[^-]+)-(?P<major>[^-]+)-(?P<semester>[^.]+)\.html",
    re.IGNORECASE
)


def parse_filename_meta(fname: str) -> Dict[str,str]:
    """Return metadata dict extracted from filename; fallback to parts."""
    m = FILENAME_META_RE.search(fname)
    if m:
        return {
            "faculty": m.group("faculty").strip(),
            "major": m.group("major").strip(),
            "semester": m.group("semester").strip()
        }
    # fallback: split on underscores/dashes
    name = Path(fname).stem
    parts = re.split(r"[-_]", name)
    return {
        "faculty": parts[0] if len(parts) > 0 else "",
        "major": parts[1] if len(parts) > 1 else "",
        "semester": parts[2] if len(parts) > 2 else ""
    }


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove script/style
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_files_and_prepare_documents(html_dir: Path) -> List[Document]:
    docs: List[Document] = []
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for p in sorted(html_dir.glob("*.html")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(raw)
        meta = parse_filename_meta(p.name)
        # Build a header with metadata for context
        header = f"Filename: {p.name}\nFaculty: {meta.get('faculty')}\nMajor: {meta.get('major')}\nSemester: {meta.get('semester')}\n\n"
        full_text = header + text
        chunks = splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source_file": p.name,
                        "faculty": meta.get("faculty"),
                        "major": meta.get("major"),
                        "semester": meta.get("semester"),
                        "chunk_index": i,
                    }
                )
            )
    return docs


def main():
    # initialize embeddings and Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Loading documents...")
    documents = load_files_and_prepare_documents(HTML_DIR)
    print(f"Prepared {len(documents)} chunks from HTML files.")

    print("Creating/connecting to Chroma collection...")
    db = Chroma.from_documents(documents,
                               embedding=embeddings,
                               persist_directory=CHROMA_PERSIST_DIR,
                               collection_name=CHROMA_COLLECTION_NAME)
    db.persist()
    print("Ingestion complete. Chroma persisted to:", CHROMA_PERSIST_DIR)


if __name__ == "__main__":
    main()
