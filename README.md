# RAG similarity demo

A Q&A RAG demo that shows embeddings and cosine similarity for every question.

- Every question in `faq.json` is embedded with `all-MiniLM-L6-v2` (384 dimensions).
- A new question is embedded and compared with each stored question using cosine similarity.
- **Similarity ≥ 99%** → the stored answer is returned from the dataset.
- **Below 99%** → "request transferred to the LLM" is shown, with the closest Q&A pairs as context (no LLM is called).

**Live dashboard:** https://omerkhalid-1.github.io/rag-similarity-demo/

The website in `docs/` is fully self-contained (nothing is downloaded from other sites):

- `docs/kb.json` holds the embeddings, computed locally by `build_kb.py`: every dataset question and answer, every button question, and the 2-D map. The buttons work instantly.
- Questions typed by hand are embedded in the browser with the model stored in `docs/models/` (Transformers.js in `docs/lib/`), loaded the first time you type a question.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python rag_demo.py --demo   # terminal version
.venv/bin/python server.py            # dashboard at http://localhost:8501
```

## Change the dataset

```bash
# 1. edit faq.json (and the VARIATIONS list in build_kb.py if you like)
.venv/bin/python build_kb.py        # re-computes docs/kb.json locally
git add -A && git commit -m "Update dataset" && git push
```
