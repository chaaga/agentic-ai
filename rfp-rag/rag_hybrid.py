# ============================================================
# RAG (Retrieval Augmented Generation) - Hybrid Variant
# Same retrieve -> rerank pipeline as rag_with_rerank.py, but
# the generation step falls back to the model's own knowledge
# when no confident match is found, instead of refusing.
# ============================================================
"""
HOW THIS DIFFERS FROM rag_with_rerank.py
-----------------------------------------
rag_with_rerank.py treats "answer ONLY from context" as a hard rule:
if nothing clears CONFIDENCE_THRESHOLD, it refuses to answer at all.
That's the right call for a strict, fully-grounded RFP assistant, but
it also means a question like "What is 2+2?" gets refused, since it
has no match in the knowledge base.

This variant uses a single flexible prompt instead:

    "Answer using the provided context if it's relevant. If no
    context is provided, or it doesn't address the question,
    answer from your own knowledge."

CONFIDENCE GATE (unchanged in spirit):
- score >= CONFIDENCE_THRESHOLD -> pass the matched answer(s) as context
- score <  CONFIDENCE_THRESHOLD -> pass empty context, let the model
  answer from its own training knowledge

KNOWN LIMITATIONS
-----------------
- This collapses the "no match" case into "ask the model anyway",
  so an in-domain question with no good match (e.g. "do you support
  SSO?" worded in a way that misses the index) gets a confident-sounding
  but ungrounded answer instead of "no good match found". For an RFP
  assistant where every answer must be traceable to a source document,
  rag_with_rerank.py's stricter behavior is usually the safer default.
- Same retrieval limitations as rag_with_rerank.py (question-only
  reranking, no chunking, hand-picked threshold).
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import ollama

# Minimum rerank score for a candidate to be used as context.
# Scale depends on the reranker model: bge-reranker outputs 0-1
# (sigmoid), so 0.3 means "at least 30% confident". Re-tune if
# the reranker model changes.
CONFIDENCE_THRESHOLD = 0.15

# At most this many matches are combined into the LLM context.
# More context helps multi-part questions but risks distracting
# the model with weaker matches.
MAX_CONTEXTS = 3

# ---- STEP 1: EMBEDDING MODEL --------------------------------
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1"
)

# ---- STEP 2: VECTOR DATABASE --------------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="rag_demo",
    embedding_function=emb_fn
)

# ---- STEP 3: INGEST KNOWLEDGE BASE --------------------------
def ingest(file_path: str):
    print("\nIngesting knowledge base...")
    df = pd.read_excel(file_path)
    questions = df["Question"].astype(str).tolist()
    answers = df["Answer"].astype(str).tolist()

    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.add(
        documents=[f"{q}\n{a}" for q, a in zip(questions, answers)],
        metadatas=[{"question": q, "answer": a} for q, a in zip(questions, answers)],
        ids=[f"id_{i}" for i in range(len(questions))],
    )
    print(f"Indexed {len(questions)} question-answer pairs\n")


# ---- STEP 4: RETRIEVE ---------------------------------------
def retrieve(query: str, top_k: int = 5) -> list:
    print(f"Retrieving top {top_k} matches for:\n   '{query}'\n")
    results = collection.query(query_texts=[query], n_results=top_k)

    candidates = results["metadatas"][0]
    for candidate, distance in zip(candidates, results["distances"][0]):
        print(f"   [{distance:.4f}] {candidate['question'][:70]}")
    return candidates


# ---- STEP 5: RERANK -----------------------------------------
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query: str, candidates: list) -> list:
    print(f"\nReranking {len(candidates)} candidates...")
    pairs = [(query, c["question"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        print(f"   [{score:6.2f}] {candidate['question'][:70]}")

    ranked = [(float(score), candidate) for score, candidate in zip(scores, candidates)]
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    print(f"\nPrinting {len(candidates)} candidates after sorting...")

    for score, candidate in ranked:
        print(f"   [{score:6.2f}] {candidate['question'][:70]}")
    return ranked


# ---- STEP 6: GENERATE ---------------------------------------
# Hybrid prompt: use the context if it's there and relevant,
# otherwise fall back to the model's own knowledge.
def generate(query: str, context: str) -> str:
    print("\nGenerating response with Ollama (llama3.1:8b)...")
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using the provided context if it's "
                    "relevant. If no context is provided, or it doesn't "
                    "address the question, answer from your own knowledge."
                ),
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"},
        ],
        options={"temperature": 0.0, "num_ctx": 8192},
    )
    return response["message"]["content"].strip()


# ---- FULL RAG PIPELINE --------------------------------------
def rag_pipeline(query: str):
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    candidates = retrieve(query, top_k=5)
    ranked = rerank(query, candidates)

    # Keep only matches the reranker is confident about, up to MAX_CONTEXTS
    good = [(score, c) for score, c in ranked if score >= CONFIDENCE_THRESHOLD]
    good = good[:MAX_CONTEXTS]

    if not good:
        print("\nNo confident match found — answering from the model's own knowledge.")
        context = ""
    else:
        print(f"\nUsing {len(good)} match(es) as context (best score: {good[0][0]:.2f})")
        context = "\n\n---\n\n".join(c["answer"] for score, c in good)
        print(f"\nContext sent to the model:\n{context}\n")

    answer = generate(query, context)
    print(f"\nFinal Answer:\n{answer}")
    print("=" * 60)


# ---- DEMO RUNNER --------------------------------------------
if __name__ == "__main__":
    # Run once to (re)build the index, then comment out again:
    ingest("data/repository.xlsx")

    demo_questions = [
        "What is 2+2?",
        "Can your solution scale as our usage grows?",
        "Do you support single sign on?",
        "How does your system handle large numbers of users?",
        "What security certifications do you have?",
        "Can your system be hosted on-premises?",
        "Can your system be integrated with external systems?",
    ]

    for q in demo_questions:
        input("\nPress Enter for next question...")  # Pause for audience
        rag_pipeline(q)
