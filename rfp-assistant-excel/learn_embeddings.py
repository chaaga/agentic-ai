import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import logging
import warnings
import os

# --- SILENCE WARNINGS ---
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ---- STEP 1: EMBEDDING MODEL --------------------------------
FAST_EMBEDDING_MODEL="all-MiniLM-L6-v2"
QA_EMBEDDING_MODEL="multi-qa-mpnet-base-dot-v1"

emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=FAST_EMBEDDING_MODEL
)

# ---- STEP 2: VECTOR DATABASE --------------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="rag_demo_fast",
   # name="rag_demo_qa",
    embedding_function=emb_fn
)

# ---- STEP 3: INGEST KNOWLEDGE BASE --------------------------
def ingest(file_path: str):
    
    print("\n Ingesting knowledge base...")
    df = pd.read_excel(file_path)

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


"""
# Check where cache lives
print(os.path.expanduser("~/.cache/huggingface/hub"))

# This will NOT download if already cached
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
print("Model:", model)  # prints model architecture confirming it loaded

for doc in documents[:3]:
    print(doc)
print("="*60)

try:
    client.delete_collection(name="rag_demo_qa")
    print("Collection deleted to reset dimensions.")
except Exception:
    print("Collection didn't exist, creating fresh.")

"""
def printEmbeddings():
# You need to explicitly request embeddings — they're not returned by default
    all_data = collection.get(include=["embeddings", "documents", "metadatas"])
    embeddings = all_data['embeddings']  # list of vectors, one per document
    documents = all_data['documents']  # list of document texts
    print(f"There are {len(documents)} documents in the collection")

    for i, (embedding, meta, doc) in enumerate(zip(embeddings, all_data['metadatas'], documents)):
        if i>= 2:
            break  
        print(f"\nRecord {i}:")
        print("="*60) 
        print(f"Document: {doc[:120]}")
        print(f"\nQuestion Length: {len(meta['question'])}: question is: {meta['question'][:60]}")
        print(f"\nAnswer Length: {len(meta['answer'])}: answer is: {meta['answer'][:60]}")
        print(f"  Vector length : {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Last 5 values : {embedding[-5:]}")
        print("="*60)

# ---- DEMO RUNNER --------------------------------------------
if __name__ == "__main__":
    ingest("data/repository.xlsx")
    printEmbeddings()
