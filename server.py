"""
RAG dashboard server (no extra packages: standard library + what rag_demo.py already uses)

Same logic as rag_demo.py:
  - every question in faq.json is embedded
  - the prompt is embedded and compared with cosine similarity
  - best similarity >= threshold (99% by default) -> stored answer, no LLM
  - otherwise -> "transfer request to the LLM" (shown only; no LLM is called)

Run:
    .venv/bin/python server.py
then open http://localhost:8501
"""

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

PORT = int(os.environ.get("PORT", 8501))
HERE = os.path.dirname(os.path.abspath(__file__))

print("Loading embedding model (all-MiniLM-L6-v2, 384 dimensions)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

with open(os.path.join(HERE, "faq.json")) as f:
    faq = json.load(f)
questions = [item["question"] for item in faq]
kb_vectors = model.encode(questions, normalize_embeddings=True)

# 2-D PCA of the stored questions (for the embedding map)
pca_mean = kb_vectors.mean(axis=0)
_, _, vt = np.linalg.svd(kb_vectors - pca_mean, full_matrices=False)
pca_components = vt[:2]


def to_2d(vectors):
    return ((np.atleast_2d(vectors) - pca_mean) @ pca_components.T).tolist()


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def handle_ask(body):
    question = str(body.get("question", "")).strip()
    threshold = float(body.get("threshold", 0.99))
    top_k = int(body.get("top_k", 3))

    q_vec = model.encode(question, normalize_embeddings=True)
    sims = [cosine_similarity(q_vec, v) for v in kb_vectors]
    order = sorted(range(len(faq)), key=lambda i: -sims[i])
    best = order[0]
    top_idx = order[:top_k]
    cache_hit = sims[best] >= threshold

    result = {
        "question": question,
        "threshold": threshold,
        "vector": q_vec.tolist(),
        "point": to_2d(q_vec)[0],
        "sims": sims,
        "order": order,
        "best": best,
        "top": top_idx,
        "route": "dataset" if cache_hit else "llm",
    }
    if cache_hit:
        result["answer"] = faq[best]["answer"]
        result["source"] = "dataset"
        result["answer_vector"] = model.encode(faq[best]["answer"], normalize_embeddings=True).tolist()
    else:
        result["source"] = "transfer"
    return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(HERE, "static"), **kwargs)

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/kb":
            self.send_json({
                "faq": faq,
                "vectors": kb_vectors.tolist(),
                "points": to_2d(kb_vectors),
            })
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not str(body.get("question", "")).strip():
            self.send_json({"error": "empty question"}, 400)
            return
        self.send_json(handle_ask(body))

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"Port {PORT} is already in use - another copy of the dashboard is probably running.\n"
                         f"Open http://localhost:{PORT}, or stop it with:  lsof -ti :{PORT} | xargs kill")
    print(f"Dashboard running at http://localhost:{PORT}  (Ctrl+C to stop)")
    server.serve_forever()
