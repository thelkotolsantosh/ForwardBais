"""Minimal RAG pipeline built without LangChain.

Pipeline:
documents -> text -> chunks -> embeddings -> FAISS -> retrieval -> Ollama
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DEFAULT_INDEX_DIR = Path("storage")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

_embedding_model = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def discover_files(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {source.suffix}. "
                f"Use one of: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return [source]

    return sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """Split text into overlapping character chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def build_index(
    source: Path,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunk_size: int = 800,
    overlap: int = 120,
) -> int:
    files = discover_files(source)

    if not files:
        raise ValueError(
            f"No supported documents found in {source}. "
            "Add .txt, .md or .pdf files."
        )

    records = []

    for path in files:
        text = read_document(path)

        for chunk_id, chunk in enumerate(
            chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            records.append(
                {
                    "source": str(path),
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    if not records:
        raise ValueError("The documents contained no readable text.")

    model = get_embedding_model()
    vectors = model.encode(
        [record["text"] for record in records],
        show_progress_bar=True,
    )
    vectors = normalize_embeddings(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config = {
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": len(records),
    }
    (index_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    return len(records)


def retrieve(
    query: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
    top_k: int = 4,
) -> list[dict]:
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "No vector index found. Run:\n"
            "python app.py index --source data"
        )

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    records = json.loads(chunks_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))

    model = get_embedding_model()
    query_vector = model.encode([query])
    query_vector = normalize_embeddings(query_vector)

    k = min(top_k, len(records))
    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = dict(records[idx])
        item["score"] = float(score)
        results.append(item)

    return results


def build_prompt(question: str, contexts: Iterable[dict]) -> str:
    context_text = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in contexts
    )

    return f"""You are a retrieval-augmented assistant.

Answer the question using only the supplied context.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""


def answer_with_ollama(
    question: str,
    contexts: list[dict],
    model: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434",
) -> str:
    prompt = build_prompt(question, contexts)

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return (
            "Could not connect to Ollama.\n\n"
            "Start Ollama and make sure the requested model is installed.\n"
            f"Example: ollama pull {model}\n\n"
            f"Technical error: {exc}"
        )

    data = response.json()
    return data.get("response", "").strip()
