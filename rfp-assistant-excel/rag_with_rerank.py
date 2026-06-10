# ============================================================
# RAG (Retrieval Augmented Generation) - Tech Talk Demo
# Shows: Embeddings → Vector Search → Reranking → LLM Response
# ============================================================

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import ollama

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
    print("\n📥 Ingesting knowledge base...")
    df = pd.read_excel(file_path)

    # Clear old data
    existing = collection.get()['ids']
    if existing:
        collection.delete(ids=existing)

    questions = df['Question'].astype(str).tolist()
    answers   = df['Answer'].astype(str).tolist()

    # Index question + answer together
    # This helps match questions phrased differently
    documents = [f"{q}\n{a}" for q, a in zip(questions, answers)]

    collection.add(
        documents=documents,
        metadatas=[{"question": q, "answer": a}
                   for q, a in zip(questions, answers)],
        ids=[f"id_{i}" for i in range(len(questions))]
    )
    print(f"✅ Indexed {len(questions)} question-answer pairs\n")


# ---- STEP 4: RETRIEVE ---------------------------------------
# Find the most similar documents to the query
def retrieve(query: str, top_k: int = 5) -> list:
    print(f"🔍 Retrieving top {top_k} matches for:\n   '{query}'\n")
    results = collection.query(query_texts=[query], n_results=top_k)

    candidates = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        candidates.append({
            "question": meta['question'],
            "answer":   meta['answer'],
            "distance": round(dist, 4)
        })
        print(f"   [{round(dist,4)}] {meta['question'][:70]}")

    return candidates


# ---- STEP 5: RERANK -----------------------------------------
# Cross-encoder reads BOTH texts together for better matching
# Much more accurate than embeddings but slower — so we use it
# only on the small set of candidates retrieved above
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, candidates: list) -> dict:
    print(f"\n🔁 Reranking {len(candidates)} candidates...")
    pairs  = [[query, c['question']] for c in candidates]
    scores = reranker.predict(pairs)

    for i, (c, s) in enumerate(zip(candidates, scores)):
        c['rerank_score'] = round(float(s), 4)
        print(f"   [{round(float(s),4):6.2f}] {c['question'][:70]}")

    best = max(candidates, key=lambda x: x['rerank_score'])
    print(f"\n✅ Best match: {best['question'][:70]}")

    return best


# ---- STEP 6: GENERATE ---------------------------------------
# The "G" in RAG — LLM uses the retrieved answer as context
# Running 100% locally via Ollama — no data leaves your machine
def generate(query: str, context: str) -> str:
    print(f"\n🤖 Generating response with Ollama (llama3.1:8b)...")
    prompt = (
        f"Answer the question using ONLY the context below.\n"
        #f"Do not add information not present in the context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.0, 'num_ctx': 2048}
    )
    return response['message']['content'].strip()


# ---- FULL RAG PIPELINE --------------------------------------
def rag_pipeline(query: str) -> str:
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)

    # Step 1: Retrieve similar documents from vector DB
    candidates = retrieve(query, top_k=5)

    # Step 2: Rerank to find the best match
    best_match = rerank(query, candidates)

    # Step 3: Generate response grounded in the retrieved answer
    if best_match['rerank_score'] < -5:
        print("\n⚠️ Low confidence in retrieved match — consider reviewing the knowledge base.")
    else:
        print(f"\n✅ High confidence in retrieved match (score: {best_match['rerank_score']})")
        final_answer = generate(query, best_match['answer'])
        print(f"\n💬 Final Answer:\n{final_answer}")

    print("="*60)
   # return final_answer


# ---- DEMO RUNNER --------------------------------------------
if __name__ == "__main__":
    # Ingest once
    #ingest("data/repository.xlsx")

    # Demo questions — great for live tech talk demo
    demo_questions = [
        "Do you support single sign on?",
        "How does your system handle large numbers of users?",
        "What security certifications do you have?",
        "Can your system be hosted on-premises?",
        "Can your system be integrated with external systems?",
    ]

    for q in demo_questions:
        rag_pipeline(q)
        input("\nPress Enter for next question...")  # Pause for audience