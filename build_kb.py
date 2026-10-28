"""
Pre-compute the embeddings for the website (docs/).

Embeds, locally with the same model as rag_demo.py / server.py:
  - every question and answer in faq.json
  - the example "variation" questions shown on the main screen
  - the 2-D PCA projection used by the embedding map
and writes everything to docs/kb.json, so the website needs no model for any button.
(Questions typed by hand are embedded in the browser with the model stored in docs/models/.)

Run after editing faq.json:
    .venv/bin/python build_kb.py
"""

import json
import os
import shutil

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))

# Variations of the dataset questions, to show what happens near / below the threshold
VARIATIONS = {
    "Near duplicate (high similarity)": ["What is RAG", "How do I reset my password", "What are your support hours"],
    "Paraphrase → transferred to LLM": ["How can I change my password?", "How can I cancel my plan?",
                                        "Why does an LLM make things up?"],
    "Off-topic → transferred to LLM": ["What's the weather today?", "Who won the World Cup?"],
}

print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

with open(os.path.join(HERE, "faq.json")) as f:
    faq = json.load(f)
questions = [item["question"] for item in faq]

q_vecs = model.encode(questions, normalize_embeddings=True)
a_vecs = model.encode([item["answer"] for item in faq], normalize_embeddings=True)

extra = [q for qs in VARIATIONS.values() for q in qs]
extra_vecs = model.encode(extra, normalize_embeddings=True)

# 2-D PCA of the stored questions (same as server.py)
mean = q_vecs.mean(axis=0)
_, _, vt = np.linalg.svd(q_vecs - mean, full_matrices=False)
components = vt[:2]


def r(a):
    return np.round(a, 6).tolist()


kb = {
    "model": "all-MiniLM-L6-v2",
    "faq": faq,
    "vectors": r(q_vecs),
    "answerVectors": r(a_vecs),
    "points": r((q_vecs - mean) @ components.T),
    "pca": {"mean": r(mean), "components": r(components)},
    "variations": VARIATIONS,
    # pre-computed vectors for every button, keyed by the exact text
    "queries": {**dict(zip(questions, r(q_vecs))), **dict(zip(extra, r(extra_vecs)))},
}

out = os.path.join(HERE, "docs", "kb.json")
with open(out, "w") as f:
    json.dump(kb, f, separators=(",", ":"))
shutil.copy(os.path.join(HERE, "faq.json"), os.path.join(HERE, "docs", "faq.json"))
print(f"Wrote {out}: {len(faq)} Q&A pairs, {len(kb['queries'])} pre-computed button questions, "
      f"{os.path.getsize(out) // 1024} KB")
