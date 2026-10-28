"""
Pre-compute the embeddings for the website (docs/).

Embeds, locally with the exact model file the website uses
(docs/models/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx - all-MiniLM-L6-v2, 8-bit, 23 MB):
  - every question and answer in faq.json
  - the example "variation" questions shown on the main screen
  - the 2-D PCA projection used by the embedding map
and writes everything to docs/kb.json, so the website needs no model for any button.
Questions typed by hand are embedded in the browser with the same model file, so the scores match exactly.

Run after editing faq.json:
    .venv/bin/python build_kb.py
"""

import json
import os
import shutil

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "docs", "models", "Xenova", "all-MiniLM-L6-v2")

# Variations of the dataset questions, to show what happens near / below the threshold
VARIATIONS = {
    "Near duplicate (high similarity)": ["How do I cancel my subscription", "How do I reset my password",
                                         "What is RAG", "What are your support hours"],
    "Paraphrase → transferred to LLM": ["How can I change my password?", "How can I cancel my plan?",
                                        "Why does an LLM make things up?"],
    "Off-topic → transferred to LLM": ["What's the weather today?", "Who won the World Cup?"],
}

print("Loading embedding model (all-MiniLM-L6-v2, 8-bit ONNX)...")
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # match the browser's numbers
session = ort.InferenceSession(os.path.join(MODEL_DIR, "onnx", "model_quantized.onnx"), opts,
                               providers=["CPUExecutionProvider"])
tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
tokenizer.no_padding()  # tokenizer.json pads to 128 tokens; the browser does not
tokenizer.enable_truncation(max_length=512)


def encode_one(text):
    """Mean-pooled, L2-normalized sentence embedding (same as Transformers.js pooling='mean', normalize=true)."""
    e = tokenizer.encode(text)
    ids = np.array([e.ids], dtype=np.int64)
    mask = np.array([e.attention_mask], dtype=np.int64)
    hidden = session.run(None, {"input_ids": ids, "attention_mask": mask,
                                "token_type_ids": np.zeros_like(ids)})[0][0]
    vec = hidden.mean(axis=0)
    return vec / np.linalg.norm(vec)


def encode(texts):
    # One text at a time, like the browser: the 8-bit model scales activations over the whole
    # input, so batching (with padding) would give slightly different vectors.
    return np.array([encode_one(t) for t in texts])


with open(os.path.join(HERE, "faq.json")) as f:
    faq = json.load(f)
questions = [item["question"] for item in faq]

q_vecs = encode(questions)
a_vecs = encode([item["answer"] for item in faq])

extra = [q for qs in VARIATIONS.values() for q in qs]
extra_vecs = encode(extra)

# 2-D PCA of the stored questions (same as server.py)
mean = q_vecs.mean(axis=0)
_, _, vt = np.linalg.svd(q_vecs - mean, full_matrices=False)
components = vt[:2]


def r(a):
    return np.round(a, 6).tolist()


kb = {
    "model": "all-MiniLM-L6-v2 (8-bit ONNX)",
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
