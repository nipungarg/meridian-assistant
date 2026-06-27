"""Build the Chroma vector index from the knowledge-pack PDFs.

KB = files 01-11. The Booking API spec (12) is deliberately excluded - it drives the mock
API and tool schemas, not customer answers, and is the most internal document. The example
messages (13) are eval-only. Embeddings use OpenAI; cosine space so similarity = 1 - dist.

Run:  python -m meridian.ingestion.build_index [--reset]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from meridian.config import get_settings
from meridian.ingestion.chunkers import chunk_document
from meridian.ingestion.pdf_extract import extract_pdf

# Files included in the customer-facing knowledge base.
KB_FILES = [
    "01_service_area_north.pdf", "02_service_area_central.pdf", "03_hvac_pricing.pdf",
    "04_plumbing_pricing.pdf", "05_electrical_pricing.pdf", "06_warranty_terms.pdf",
    "07_cancellation_policy.pdf", "08_branch_hours.pdf", "09_faq_booking.pdf",
    "10_faq_payments.pdf", "11_faq_emergencies.pdf",
]


def _embedding_function():
    s = get_settings()
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=s.openai_api_key, model_name=s.openai_embedding_model
    )


def _client() -> chromadb.ClientAPI:
    s = get_settings()
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(s.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )


def get_collection(create: bool = False):
    s = get_settings()
    client = _client()
    kwargs = dict(name=s.chroma_collection, embedding_function=_embedding_function(),
                  metadata={"hnsw:space": "cosine"})
    if create:
        return client.get_or_create_collection(**kwargs)
    return client.get_collection(name=s.chroma_collection, embedding_function=_embedding_function())


def build_index(reset: bool = True, files_dir: str | Path | None = None) -> dict:
    s = get_settings()
    files_dir = Path(files_dir or s.files_dir)
    client = _client()
    if reset:
        try:
            client.delete_collection(s.chroma_collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=s.chroma_collection, embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    per_file: dict[str, int] = {}
    for fname in KB_FILES:
        path = files_dir / fname
        if not path.exists():
            continue
        extracted = extract_pdf(path)
        chunks = chunk_document(extracted)
        per_file[fname] = len(chunks)
        for ch in chunks:
            ids.append(ch.metadata["chunk_id"])
            docs.append(ch.text)
            metas.append(ch.metadata)

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    return {"total_chunks": len(ids), "per_file": per_file, "collection": s.chroma_collection}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the collection")
    args = parser.parse_args()
    summary = build_index(reset=True if args.reset else True)
    print(f"Built collection '{summary['collection']}' with {summary['total_chunks']} chunks:")
    for f, n in summary["per_file"].items():
        print(f"  {f:32s} {n:3d} chunks")
