"""
Simple Q&A RAG demo (semantic cache + LLM fallback)

1. Embed every QUESTION in faq.json (the "knowledge base").
2. For each user prompt:
     - embed the prompt and print its vector
     - compute cosine similarity against every stored question and print them
     - if best similarity >= 0.99  -> return the stored answer directly (no LLM)
     - else                        -> "transfer request to the LLM" (shown only; no LLM is called)

Run:
    .venv/bin/python rag_demo.py            # interactive
    .venv/bin/python rag_demo.py --demo     # runs preset questions
"""

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain download; the xet CDN failed on this machine

import numpy as np
from sentence_transformers import SentenceTransformer

SIMILARITY_THRESHOLD = 0.99   # 99% -> treat as "same question"
TOP_K = 3                     # how many closest Q&A pairs are passed on with the transfer
DIMS_TO_SHOW = 8              # how many embedding numbers to print

# ---------------------------------------------------------------------------
# 1. Load dataset and embed it
# ---------------------------------------------------------------------------
print("Loading embedding model (all-MiniLM-L6-v2, 384 dimensions)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("faq.json") as f:
    faq = json.load(f)

questions = [item["question"] for item in faq]
# normalize_embeddings=True -> every vector has length 1, so cosine = dot product
kb_vectors = model.encode(questions, normalize_embeddings=True)


def show_vector(label, vec):
    head = ", ".join(f"{x:+.4f}" for x in vec[:DIMS_TO_SHOW])
    print(f"  {label:<45} [{head}, ...]  dim={len(vec)}  norm={np.linalg.norm(vec):.3f}")


print(f"\n=== Knowledge base: {len(faq)} Q&A pairs embedded ===")
for q, v in zip(questions, kb_vectors):
    show_vector(q, v)


# ---------------------------------------------------------------------------
# 2. Cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# 3. Answer a prompt
# ---------------------------------------------------------------------------
def answer(user_question):
    print("\n" + "=" * 90)
    print(f"PROMPT: {user_question}")
    print("=" * 90)

    q_vec = model.encode(user_question, normalize_embeddings=True)
    print("\n[1] Prompt embedding:")
    show_vector(user_question, q_vec)

    scores = [(i, cosine_similarity(q_vec, kb_vectors[i])) for i in range(len(faq))]
    scores.sort(key=lambda s: s[1], reverse=True)

    print("\n[2] Cosine similarity vs. every stored question:")
    for i, s in scores:
        bar = "#" * max(0, int(s * 30))
        print(f"  {s:+.4f}  {bar:<30}  {faq[i]['question']}")

    best_i, best_score = scores[0]
    print(f"\n[3] Best match: \"{faq[best_i]['question']}\"  similarity={best_score:.4f} "
          f"({best_score * 100:.2f}%)  threshold={SIMILARITY_THRESHOLD * 100:.0f}%")

    if best_score >= SIMILARITY_THRESHOLD:
        print("    -> MATCH >= 99%: returning stored answer (no LLM call)")
        print("\n    Stored answer's embedding:")
        show_vector(faq[best_i]["answer"][:45], model.encode(faq[best_i]["answer"], normalize_embeddings=True))
        print(f"\nANSWER (from dataset): {faq[best_i]['answer']}")
    else:
        top = scores[:TOP_K]
        print(f"    -> below {SIMILARITY_THRESHOLD * 100:.0f}%: no match in the dataset")
        print("\n>>> REQUEST TRANSFERRED TO THE LLM <<<")
        print(f"    Context passed along (top {TOP_K} closest Q&A pairs):")
        for i, s in top:
            print(f"       {s:.4f}  Q: {faq[i]['question']}")
            print(f"               A: {faq[i]['answer']}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        for q in [
            "What is RAG?",                       # exact -> cache hit
            "what is rag",                        # same words, different casing
            "Can you explain retrieval augmented generation?",  # paraphrase -> LLM
            "How can I change my password?",      # similar -> LLM
            "What's the weather today?",          # unrelated -> transferred to LLM
        ]:
            answer(q)
    else:
        print("\nType a question (or 'quit').")
        while True:
            q = input("\n> ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            if q:
                answer(q)
