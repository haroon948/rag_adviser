# RAG Soil Advisor

A retrieval-augmented question-answering assistant for drought tolerance and heat and climate stress in crops. It answers from a fixed set of open-licensed documents, cites the source passages, and declines when the documents don't contain the answer.

## Status

- [x] Local LLM and embedding model check
- [ ] Document manifest with licence checks
- [ ] Ingestion and retrieval
- [ ] Answer generation with citations
- [ ] Evaluation set and results

## Setup

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Requires Ollama with `llama3.2:3b` pulled locally.