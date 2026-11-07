"""Retrieve the most relevant passages for a question from the saved index."""
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

EMBED_MODEL = "all-MiniLM-L6-v2"  # must match the model used in ingest.py
TOP_K = 5


class Retriever:
    def __init__(self):
        self.chunks = pd.read_parquet(PROCESSED / "chunks.parquet")
        self.vectors = np.load(PROCESSED / "embeddings.npy")
        self.model = SentenceTransformer(EMBED_MODEL)

        if len(self.chunks) != len(self.vectors):
            raise ValueError("chunks and embeddings differ in length; rerun ingest.py")

    def search(self, question, k=TOP_K):
        query = self.model.encode([question], normalize_embeddings=True)[0]
        # Vectors are normalised at ingestion, so the dot product is the cosine similarity
        scores = self.vectors @ query
        top = np.argsort(scores)[::-1][:k]

        results = self.chunks.iloc[top].copy()
        results["score"] = scores[top]
        return results[["doc_id", "page", "score", "text"]].reset_index(drop=True)


def show(results, width=300):
    for rank, row in enumerate(results.itertuples(), start=1):
        snippet = row.text[:width].replace("\n", " ")
        print(f"{rank}. [{row.doc_id}, p.{row.page}] score={row.score:.3f}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is the stay-green trait?"
    retriever = Retriever()
    show(retriever.search(question))