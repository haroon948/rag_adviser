# RAG Soil Advisor

A retrieval-augmented question-answering assistant for drought tolerance and heat and climate stress in crops. It answers only from a fixed set of open-licensed research papers, cites the passages it used, and declines when the corpus doesn't cover the question.

## What it does

Given a question, the assistant:

1. Embeds the question and ranks passages from the corpus by cosine similarity.
2. Drops weak matches (score below 0.45) and caps each paper at two passages.
3. Asks a local language model (`llama3.2:3b` via Ollama) to answer from those passages only.
4. Attaches the source labels of the passages to the answer, and rejects any draft that names a source outside the passages.

Citations are attached at the level of the whole answer, not per sentence. A reader can see which passages were supplied, but not which passage supports each claim.

## Corpus

Twenty accepted papers on crop drought tolerance and heat or climate stress, covering wheat, barley, rice, maize, sorghum, cassava, and tomato. Each paper was accepted only if its publisher page states a Creative Commons licence. The full list, with licences and the reason for each decision (including rejected and out-of-scope papers), is in `manifest.csv`. The PDFs are not redistributed in this repository.

Reference lists and citation-heavy text are removed at ingestion, so bibliography entries are not retrieved as evidence.

## Evaluation

Question set: 27 items in `eval/questions.csv`, split into `dev` (14) and `test` (13). The dev split was used to tune the threshold and prompt. The test split was run once, with the settings unchanged.

| Split | Answerable questions | Retrieval found the expected paper | Answered correctly | Out-of-scope refused | Rejected for bad citations |
|-------|----------------------|------------------------------------|--------------------|----------------------|----------------------------|
| Dev   | 10                   | 10 of 10                           | 9 of 10            | 4 of 4               | 0                          |
| Test  | 10                   | 10 of 10                           | 10 of 10           | 3 of 3               | 0                          |

Known limitations of these results:

- **Small sample.** One miss in the dev split changes the rate by 10 percentage points. The test split has 10 answerable questions, so these figures should be read as a check on behaviour, not a measure of accuracy.
- **One documented failure (A08).** The assistant retrieved the correct rice paper but refused the question about reproductive-stage drought and rice yield. A prompt change to fix it was tested and didn't resolve it, so it was left as a known limitation rather than tuned further.
- **Hand-written expected sources.** The question set and the expected papers were written by the author, not by an independent reviewer.
- **Whole-answer attribution.** See above.

To reproduce:

```bash
python src\evaluate.py dev
python src\evaluate.py test
```

Results are written to `results/eval_summary_*.csv` and `results/eval_results_*.csv`.

## Setup

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Requires Ollama with `llama3.2:3b` pulled locally:

    ollama pull llama3.2:3b

Build the index from the PDFs in `data/raw/`, which must match the accepted rows in `manifest.csv`:

    python src\ingest.py

Then ask a question:

    python src\answer.py "What is the stay-green trait?"

## Project structure

```
manifest.csv    document list with licence checks and decisions
eval/           labelled question set
src/            ingestion, retrieval, answering, evaluation
results/        evaluation summaries and per-question results
data/           PDFs and built index (not tracked)
```

## Status

- [x] Document manifest with licence checks
- [x] Ingestion with reference filtering
- [x] Retrieval
- [x] Answer generation with code-attached citations
- [x] Evaluation set and results

## Licence

Code: MIT. Documents: each retains its own licence as listed in `manifest.csv`.

## Author

Haruna Isola Aremu