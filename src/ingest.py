"""Build a searchable index from accepted PDFs listed in manifest.csv."""
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.csv"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dimensional, Apache 2.0
CHUNK_WORDS = 200
OVERLAP_WORDS = 40


def load_accepted():
    m = pd.read_csv(MANIFEST, encoding="utf-8")
    accepted = m[m["status"] == "accepted"]
    missing = [d for d in accepted["doc_id"] if not (RAW / f"{d}.pdf").exists()]
    if missing:
        raise FileNotFoundError(f"Accepted rows without a PDF: {missing}")
    return accepted[["doc_id", "title", "licence"]]


def pdf_pages(path):
    """Yield (page_number, text) for each page with extractable text."""
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield i, text


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    step = size - overlap
    for start in range(0, max(len(words) - overlap, 1), step):
        piece = " ".join(words[start:start + size])
        if piece:
            yield piece


def build_chunks(docs):
    rows = []
    for _, doc in docs.iterrows():
        path = RAW / f"{doc['doc_id']}.pdf"
        for page_no, text in pdf_pages(path):
            for piece in chunk_words(text):
                rows.append({
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "licence": doc["licence"],
                    "page": page_no,
                    "text": piece,
                })
    return pd.DataFrame(rows)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    docs = load_accepted()
    print(f"Accepted documents: {len(docs)}")

    chunks = build_chunks(docs)
    print(f"Chunks: {len(chunks)}")
    print(chunks.groupby("doc_id").size().sort_values(ascending=False).to_string())

    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(
        chunks["text"].tolist(),
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    chunks.to_parquet(PROCESSED / "chunks.parquet", index=False)
    np.save(PROCESSED / "embeddings.npy", vectors.astype(np.float32))
    print(f"Saved {len(chunks)} chunks and embeddings {vectors.shape} to {PROCESSED}")


if __name__ == "__main__":
    main()