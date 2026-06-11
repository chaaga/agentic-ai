# ============================================================
# RAG (Retrieval Augmented Generation) - Tech Talk Demo
# Simplified version 
# ============================================================

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ollama

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
    print("\n Ingesting knowledge base...")
    df = pd.read_excel(file_path)

    existing = collection.get()['ids']
    if existing:
        collection.delete(ids=existing)

    questions = df['Question'].astype(str).tolist()
    answers   = df['Answer'].astype(str).tolist()

    # Index question + answer together for better matching
    documents = [f"{q}\n{a}" for q, a in zip(questions, answers)]

    collection.add(
        documents=documents,
        metadatas=[{"question": q, "answer": a}
                   for q, a in zip(questions, answers)],
        ids=[f"id_{i}" for i in range(len(questions))]
    )
    print(f"Indexed {len(questions)} question-answer pairs\n")


# ---- STEP 4: RETRIEVE ---------------------------------------
def retrieve(query: str, top_k: int = 1) -> dict:
    print(f"Searching for: '{query}'")
    results = collection.query(query_texts=[query], n_results=top_k)

    best = results['metadatas'][0][0]
    distance = results['distances'][0][0]
    print(f"   Matched : {best['question'][:70]}")
    print(f"   Distance: {round(distance, 4)}")
    ## What ChromaDB Does Under the Hood
    """
    code passes raw text
            │
            ▼
    ChromaDB calls emb_fn(new_q)
            │
            ▼
    Generates embedding vector e.g. [0.23, -0.11, 0.87, ...]
            │
            ▼
    Compares against all stored vectors using dot product
            │
            ▼
    Returns top N closest matches
    """
    return best


# ---- STEP 5: GENERATE ---------------------------------------
def generate(query: str, context: str) -> str:
    print(f"\nGenerating response with Ollama...")
    prompt = (
        f"Answer the question using ONLY the context below.\n"
        f"Do not add information not present in the context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.0, 'num_ctx': 2048, 'num_gpu': 99}
    )
    return response['message']['content'].strip()


# ---- FULL RAG PIPELINE --------------------------------------
def rag_pipeline(query: str) -> str:
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)

    # Step 1: Retrieve best match from vector DB
    best_match = retrieve(query)
    print(f"Best_Match for query: {best_match}")
    # Step 2: Generate response grounded in retrieved answer
    final_answer = generate(query, best_match['answer'])

    print(f"\nAnswer:\n{final_answer}")
    print("="*60)
    return final_answer


# ---- DEMO RUNNER --------------------------------------------
if __name__ == "__main__":
    ingest("data/repository.xlsx")

    demo_questions = [
        "Do you support single sign on?",
        "How does your system handle large numbers of users?",
        "What security certifications do you have?",
        "Can your system be deployed on-premises?",
        "Can your system be integrated with external systems?",
        "What is 2+2",
    ]

    for q in demo_questions:
        rag_pipeline(q)
        input("\nPress Enter for next question...")