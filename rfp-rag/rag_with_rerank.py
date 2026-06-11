# ============================================================
# RAG (Retrieval Augmented Generation) - Tech Talk Demo
# Shows: Embeddings → Vector Search → Reranking → LLM Response
# ============================================================
"""
HOW THIS SCRIPT GETS TO THE BEST RESPONSE
-----------------------------------------
Two-stage retrieval, then grounded generation:

1. RETRIEVE (fast, approximate). The embedding model is a "bi-encoder":
   it encodes the query and each document SEPARATELY into vectors, and
   ChromaDB returns the top-k closest by vector distance. Because every
   document was embedded ahead of time at ingest, this scales to
   thousands of documents - but the query and document never look at
   each other directly, so subtle matches get mediocre scores.

2. RERANK (slow, accurate). The cross-encoder reads query + candidate
   TOGETHER in a single pass, so it can weigh exact word interactions
   between the two texts. Far more accurate, but it must run the full
   model once per pair - too slow for the whole knowledge base, cheap
   for 5 candidates. Stage 1 provides recall, stage 2 provides
   precision; the best match is whichever candidate scores highest.

3. CONFIDENCE GATE. Only candidates scoring above CONFIDENCE_THRESHOLD
   are kept (at most MAX_CONTEXTS of them), so the amount of context
   adapts to how many good matches exist. If none qualify, we say so
   instead of generating an answer from a bad match. The threshold
   lives on the reranker's score scale, so it must be re-tuned
   whenever the reranker model changes (raw logits vs 0-1 sigmoid).

4. GENERATE. The LLM is told to answer ONLY from the retrieved answer.
   The model supplies fluency; the knowledge base supplies the facts.

KNOWN LIMITATIONS
-----------------
- Reranking scores the query against the stored QUESTION only, so a
  match can be missed when the wording differs (e.g. "single sign on"
  vs "SSO") even though the stored answer text would have matched.
- The confidence threshold is hand-picked, not calibrated against any
  labeled data, and silently breaks when the reranker model changes.
- Each Excel row is indexed whole - no chunking, so very long answers
  dilute their embedding.
- "Answer ONLY from the context" is an instruction, not a guarantee;
  the LLM can still drift or hallucinate.
- No conversation memory: every question is answered in isolation.

ENHANCEMENT IDEAS
-----------------
- Rerank against question + answer text, not just the question.
- Decompose compound questions into sub-questions, retrieve for each
  sub-question separately, and merge the contexts (agentic RAG).
- Hybrid search: combine vector similarity with keyword search (BM25),
  which handles exact terms, IDs, and abbreviations better.
- Rewrite/expand the query before retrieval (e.g. expand abbreviations).
- Build a small golden set of question -> expected-match pairs and
  measure retrieval hit rate, so changes can be evaluated objectively.
- Calibrate the confidence threshold from those measurements.
- Add a metadata column (category) and filter the search by it.
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
# Converts text into vectors (lists of numbers)
# Similar meaning = similar vectors = close in vector space
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1"
)

# ---- STEP 2: VECTOR DATABASE --------------------------------
# Stores text + their vector representations on disk
# Enables fast similarity search across thousands of documents
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="rag_demo",
    embedding_function=emb_fn
)

# ---- STEP 3: INGEST KNOWLEDGE BASE --------------------------
# This is the "R" in RAG — building the retrieval index
def ingest(file_path: str):
    print("\nIngesting knowledge base...")
    df = pd.read_excel(file_path)
    questions = df["Question"].astype(str).tolist()
    answers = df["Answer"].astype(str).tolist()

    # Clear old data
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    # Index question + answer together
    # This helps match questions phrased differently
    collection.add(
        documents=[f"{q}\n{a}" for q, a in zip(questions, answers)],
        metadatas=[{"question": q, "answer": a} for q, a in zip(questions, answers)],
        ids=[f"id_{i}" for i in range(len(questions))],
    )
    print(f"Indexed {len(questions)} question-answer pairs\n")


# ---- STEP 4: RETRIEVE ---------------------------------------
# Find the most similar documents to the query
def retrieve(query: str, top_k: int = 5) -> list:
    print(f"Retrieving top {top_k} matches for:\n   '{query}'\n")
    results = collection.query(query_texts=[query], n_results=top_k)

    # Each candidate is a dict with "question" and "answer" keys
    candidates = results["metadatas"][0]
    for candidate, distance in zip(candidates, results["distances"][0]):
        print(f"   [{distance:.4f}] {candidate['question'][:70]}")
    return candidates


# ---- STEP 5: RERANK -----------------------------------------
# Cross-encoder reads BOTH texts together for better matching
# Much more accurate than embeddings but slower — so we use it
# only on the small set of candidates retrieved above
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query: str, candidates: list) -> list:
    print(f"\nReranking {len(candidates)} candidates...")
    pairs = [(query, c["question"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        print(f"   [{score:6.2f}] {candidate['question'][:70]}")

    # Pair each candidate with its score, best first
    ranked = [(float(score), candidate) for score, candidate in zip(scores, candidates)]
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    print(f"\nPrinting {len(candidates)} candidates after sorting...")

    for score, candidate in ranked:
        print(f"   [{score:6.2f}] {candidate['question'][:70]}")
    return ranked


# ---- STEP 6: GENERATE ---------------------------------------
# The "G" in RAG — LLM uses the retrieved answer as context
# Running 100% locally via Ollama — no data leaves your machine
def generate(query: str, context: str) -> str:
    print("\nGenerating response with Ollama (llama3.1:8b)...")
    prompt = (
        f"Answer the question using ONLY the context below.\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        # 2048 was enough for one answer; with up to MAX_CONTEXTS answers
        # in the prompt, a larger window avoids silent truncation
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
        print("\nNo confident match found — consider reviewing the knowledge base.")
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
