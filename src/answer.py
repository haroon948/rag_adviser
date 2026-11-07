"""Answer a question using only passages retrieved from the corpus, with citations."""
import re
import sys

import ollama

from retrieve import Retriever, show

LLM_MODEL = "llama3.2:3b"
TOP_K = 8             # passages retrieved before filtering
MAX_PER_DOC = 2       # cap chunks from one paper so the context covers more sources
MIN_SCORE = 0.45      # below this, the corpus is treated as not covering the question

SYSTEM = """You answer questions about crop drought tolerance and heat stress.
Use only the numbered passages provided. Do not use outside knowledge.
Cite passages in square brackets using their label, for example [Enyew_etal_2022, p.4].
If the passages do not state the answer explicitly, reply exactly: "The corpus does not cover this question."
Never name a source that is not in the passages.
Keep the answer under 150 words."""


def select_passages(results):
    kept, counts = [], {}
    for row in results.itertuples():
        if row.score < MIN_SCORE:
            continue
        if counts.get(row.doc_id, 0) >= MAX_PER_DOC:
            continue
        counts[row.doc_id] = counts.get(row.doc_id, 0) + 1
        kept.append(row)
    return kept


def build_prompt(question, passages):
    blocks = [
        f"[{p.doc_id}, p.{p.page}]\n{p.text}"
        for p in passages
    ]
    context = "\n\n".join(blocks)
    return f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"


def answer(question, retriever):
    results = retriever.search(question, k=TOP_K)
    passages = select_passages(results)
    if not passages:
        return "The corpus does not cover this question.", []

    system = SYSTEM.replace(
        "Cite passages in square brackets using their label, for example [Enyew_etal_2022, p.4].",
        "Do not write any citations or source names in your answer."
    )
    reply = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": build_prompt(question, passages)},
        ],
        options={"temperature": 0.0},
    )
    text = reply.message.content.strip()
    labels = sorted({f"[{p.doc_id}, p.{p.page}]" for p in passages})
    return f"{text}\n\nBased on: {', '.join(labels)}", passages


def check_citations(text, passages):
    allowed = {p.doc_id for p in passages}
    cited = re.findall(r"\[([^\]]+)\]", text)
    bad = []
    for c in cited:
        # A citation may hold several sources separated by semicolons
        for part in c.split(";"):
            name = part.strip()
            if not name or not re.search(r"[A-Za-z]", name):
                continue
            if not any(a in name for a in allowed):
                bad.append(name)
    return bad

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the stay-green trait?"
    retriever = Retriever()
    text, passages = answer(question, retriever)

    bad = check_citations(text, passages)
    if bad:
        print("Could not produce a grounded answer. The draft cited sources that are not in the retrieved passages:")
        for name in bad:
            print(f"  - {name}")
        print("\nSources retrieved for this question are listed below.")
    elif passages:
        print(text)
    else:
        print(text)

    print()
    if passages:
        print("Sources:")
        for p in passages:
            print(f"  [{p.doc_id}, p.{p.page}] score={p.score:.3f}")