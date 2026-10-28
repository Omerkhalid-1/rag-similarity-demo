# RAG similarity demo

A Q&A RAG demo that shows embeddings and cosine similarity for every question.

- Every question in `faq.json` is embedded with `all-MiniLM-L6-v2` (384 dimensions).
- A new question is embedded and compared with each stored question using cosine similarity.
- **Similarity ≥ 99%** → the stored answer is returned from the dataset.
- **Below 99%** → "request transferred to the LLM" is shown, with the closest Q&A pairs as context (no LLM is called).

**Live dashboard:** https://omerkhalid-1.github.io/rag-similarity-demo/

The website in `docs/` runs entirely in the browser (Transformers.js). The first visit downloads the model (~90 MB), then it is cached.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python rag_demo.py --demo   # terminal version
.venv/bin/python server.py            # dashboard at http://localhost:8501
```

To change the dataset, edit `faq.json` (and `docs/faq.json` for the website).
