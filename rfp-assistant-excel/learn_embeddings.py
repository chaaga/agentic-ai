"""
Learning script: how text embeddings and a vector database work.

Reads Q&A pairs from an Excel file (data/repository.xlsx), joins each
question + answer into one document, embeds each document with a
sentence-transformers model (one 384-dim vector per Q&A pair), and
stores them in a ChromaDB collection (deleted and rebuilt fresh on
every run). Then prints a couple of records so you can see the raw
vectors behind the text.

HOW TO RUN INTERACTIVELY (VS Code):
The `# %%` markers below split the file into Jupyter-style cells.
Click "Run Cell" above any marker (or press Shift+Enter inside a cell)
to execute just that step in the Interactive Window. Every variable
(df, documents, embeddings, ...) stays alive in the session — inspect
them in the Jupyter "Variables" panel or by typing the name in the
Interactive Window. Running the file normally (python learn_embeddings.py)
still executes everything top to bottom.

KNOWN LIMITATION — no chunking:
Each Q&A pair is embedded as a single document. The embedding model
(all-MiniLM-L6-v2) only reads the first ~256 tokens (~190 words) and
silently truncates the rest, so for long answers the tail is invisible
to semantic search. The proper fix is to split long answers into
overlapping ~150-word chunks, embed each chunk as its own record
(with the question prepended for context), and keep the full answer in
metadata. Left as-is here because this script's purpose is to
demonstrate the basic embed-and-store flow, not a production
ingestion pipeline.

HOW SIMILARITY SEARCH WORKS (Step 5):
collection.query() embeds the query text with the same model, then
returns the stored vectors closest to it. Chroma reports DISTANCES,
not similarities: 0 = identical, SMALLER = more similar (the opposite
convention from cosine similarity, where closer to 1 = more similar).
The default metric is squared Euclidean (L2) distance — the straight
"ruler" distance between two points in 384-dim space. Because
all-MiniLM-L6-v2 produces unit-length vectors, L2 and cosine give
identical rankings (L2^2 = 2 - 2*cosine_sim, so distance 0.65 ~=
cosine similarity 0.67). The metric is set at collection creation via
metadata={"hnsw:space": "l2" | "cosine" | "ip"} and should match the
model: "dot" models like multi-qa-mpnet-base-dot-v1 emit
non-normalized vectors and want "ip" (inner product). To find the
nearest vectors quickly, Chroma uses an HNSW index — a graph where
each point links to its near neighbors, so a search hops point-to-
point toward the query instead of measuring distance to every stored
vector (approximate, but effectively exact at this scale).
"""
# %% Setup: imports and warning noise
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import logging
import warnings

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# %% STEP 1: Embedding model — text in, 384 numbers out
FAST_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QA_EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"  # 768-dim, slower, QA-tuned

emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=FAST_EMBEDDING_MODEL
)

# Try it directly: embed two sentences and look at the raw vectors
sample_vectors = emb_fn(["Do you support SSO?", "Can users log in with SAML?"])
print(f"Each text became {len(sample_vectors[0])} numbers")
print("First 5 values of vector 0:", sample_vectors[0][:5])

# %% STEP 2: Vector database — fresh collection every run
client = chromadb.PersistentClient(path="./chroma_db")

name = "rag_demo_fast"
# name = "rag_demo_qa"
try:
    client.delete_collection(name)
except Exception:
    pass  # didn't exist yet
collection = client.create_collection(name=name, embedding_function=emb_fn)

# %% STEP 3a: Load the knowledge base from Excel
file_path = "data/repository.xlsx"
df = pd.read_excel(file_path)

questions = df['Question'].astype(str).tolist()
answers = df['Answer'].astype(str).tolist()
print(f"Loaded {len(df)} rows")
df.head()

# %% STEP 3b: Build documents — one string per Q&A pair
# Index question + answer together for better matching
documents = [f"{q}\n{a}" for q, a in zip(questions, answers)]
documents[:3]

# %% STEP 3c: Embed and store — embeddings are computed inside add()
collection.add(
    documents=documents,
    metadatas=[{"question": q, "answer": a}
               for q, a in zip(questions, answers)],
    ids=[f"id_{i}" for i in range(len(questions))]
)
print(f"Indexed {len(questions)} question-answer pairs")

# %% STEP 4: Peek inside the collection — see the stored vectors
# Embeddings must be requested explicitly — not returned by default
all_data = collection.get(include=["embeddings", "documents", "metadatas"])
embeddings = all_data['embeddings']
stored_docs = all_data['documents']
print(f"There are {len(stored_docs)} documents in the collection")
# Empty? The Step 2 cell wipes the collection — re-run Steps 2 -> 3c in order.
assert len(stored_docs) > 0, "Collection is empty: re-run the ingest cells (Steps 2 through 3c) first"

for i in range(2):
    embedding, meta, doc = embeddings[i], all_data['metadatas'][i], stored_docs[i]
    print(f"\nRecord {i}:")
    print("=" * 60)
    print(f"Document: {doc[:120]}")
    print(f"\nQuestion Length: {len(meta['question'])}: question is: {meta['question'][:60]}")
    print(f"\nAnswer Length: {len(meta['answer'])}: answer is: {meta['answer'][:60]}")
    print(f"  Vector length : {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Last 5 values : {embedding[-5:]}")
    print("=" * 60)

# %% STEP 5: Semantic search — the payoff
# The query is embedded with the same model, then nearest vectors win.
# Note the top hit shares no keywords with the query — that's embeddings.
results = collection.query(query_texts=["How do I sign in once for all apps?"], n_results=3)
for doc, dist in zip(results['documents'][0], results['distances'][0]):
    print(f"distance={dist:.4f}  |  {doc[:100]}")
