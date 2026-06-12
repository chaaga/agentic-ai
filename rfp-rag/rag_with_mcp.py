import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from mcp.server.fastmcp import FastMCP
import os
file_path = os.path.join(os.path.dirname(__file__), "data/repository.xlsx")

mcp = FastMCP(
    name="rfp_knowledge_base",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8052,  # only used for SSE transport (set this to any port)
)

CONFIDENCE_THRESHOLD = 0.15

MAX_CONTEXTS = 3

emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1"
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="rag_demo",
    embedding_function=emb_fn
)

df = pd.read_excel(file_path)
questions = df["Question"].astype(str).tolist()
answers = df["Answer"].astype(str).tolist()

# Clear old data
existing_ids = collection.get()["ids"]
if existing_ids:
    collection.delete(ids=existing_ids)

collection.add(
    documents=[f"{q}\n{a}" for q, a in zip(questions, answers)],
    metadatas=[{"question": q, "answer": a} for q, a in zip(questions, answers)],
    ids=[f"id_{i}" for i in range(len(questions))],
)
print(f"Indexed {len(questions)} question-answer pairs\n")

def retrieve(query: str, top_k: int = 5) -> list:
    print(f"Retrieving top {top_k} matches for:\n   '{query}'\n")
    results = collection.query(query_texts=[query], n_results=top_k)

    # Each candidate is a dict with "question" and "answer" keys
    candidates = results["metadatas"][0]
    for candidate, distance in zip(candidates, results["distances"][0]):
        print(f"   [{distance:.4f}] {candidate['question'][:70]}")
    return candidates

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
    
    good = [c for score, c in ranked if score >= CONFIDENCE_THRESHOLD]
    good = good[:MAX_CONTEXTS]

    return good

@mcp.tool()
def get_office_location() -> str:
    """Return the city where the company HQ is located."""
    return "New Delhi"

@mcp.tool()
def get_relevant_info(query: str) -> list:
    """Return relevant question/answer pairs from the knowledge base for the query."""
    retrieved = retrieve(query)
    reranked = rerank(query, retrieved)
    return reranked

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

