"""Evaluate retrieval and grounded answering against the labelled question set.

Usage from the project root:
    python src\\evaluate.py          # all questions
    python src\\evaluate.py dev      # development split, for tuning
    python src\\evaluate.py test     # held-out split, run once when reporting
"""
import sys
from pathlib import Path

import pandas as pd

from answer import TOP_K, answer, check_citations, select_passages
from retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.csv"
RESULTS = ROOT / "results"
REFUSAL = "The corpus does not cover this question."
HIT_DEPTH = 5  # a retrieval hit means an expected document is in the top 5 passages


def outcome(text, passages):
    if not passages or text.startswith(REFUSAL):
        return "refused"
    if check_citations(text, passages):
        return "rejected"
    return "answered"


def is_hit(passages, expected):
    return any(p.doc_id in expected for p in passages[:HIT_DEPTH])


def evaluate(df, retriever):
    rows = []
    for q in df.itertuples():
        raw = retriever.search(q.question, k=TOP_K)
        text, passages = answer(q.question, retriever)
        result = outcome(text, passages)
        bad = check_citations(text, passages) if passages else []

        if q.expect == "answer":
            expected = {e.strip() for e in str(q.expected_docs).split(";") if e.strip()}
            if not expected:
                raise ValueError(f"{q.qid}: answer question has no expected_docs")
            hit = is_hit(passages, expected)
            correct = hit and result == "answered"
        else:
            hit = None
            correct = result == "refused"

        rows.append({
            "qid": q.qid,
            "split": q.split,
            "expect": q.expect,
            "question": q.question,
            "top_score": round(float(raw.score.iloc[0]), 3) if len(raw) else None,
            "n_passages": len(passages),
            "retrieval_hit": hit,
            "outcome": result,
            "correct": correct,
            "bad_citations": "; ".join(bad),
            "answer": text,
        })
    return rows


def summarise(res):
    ans = res[res.expect == "answer"]
    ref = res[res.expect == "refuse"]
    return pd.Series({
        "answerable": len(ans),
        "retrieval_hit_rate": ans.retrieval_hit.astype(float).mean(),
        "answered_rate": (ans.outcome == "answered").mean(),
        "correct_answer_rate": ans.correct.astype(float).mean(),
        "false_refusal_rate": (ans.outcome == "refused").mean(),
        "out_of_scope": len(ref),
        "correct_refusal_rate": ref.correct.astype(float).mean(),
        "false_answer_rate": (ref.outcome == "answered").mean(),
        "rejected_rate": (res.outcome == "rejected").mean(),
    })


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else None
    df = pd.read_csv(QUESTIONS, encoding="utf-8")
    if split:
        df = df[df.split == split]
    if df.empty:
        raise SystemExit(f"No questions for split {split!r}")

    retriever = Retriever()
    res = pd.DataFrame(evaluate(df, retriever))

    RESULTS.mkdir(exist_ok=True)
    tag = split or "all"
    res.to_csv(RESULTS / f"eval_results_{tag}.csv", index=False, encoding="utf-8")
    summary = summarise(res)
    summary.to_csv(RESULTS / f"eval_summary_{tag}.csv", header=["value"], encoding="utf-8")

    print(summary.round(3).to_string())
    misses = res[~res.correct.astype(bool)]
    print(f"\nFailures: {len(misses)}")
    for m in misses.itertuples():
        print(f"  {m.qid} [{m.expect}] outcome={m.outcome} hit={m.retrieval_hit} :: {m.question}")


if __name__ == "__main__":
    main()