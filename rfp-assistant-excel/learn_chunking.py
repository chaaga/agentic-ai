"""
Learning script: chunking a plain text file for embeddings.

Reads a long text file (data/sample_article.txt), then runs several
chunking SCENARIOS (different chunk sizes and overlaps) against the
same text. For each scenario, chunks are embedded with a
sentence-transformers model and loaded into a fresh in-memory
("ephemeral") ChromaDB collection, then a few semantic search queries
are run so you can compare how chunking choices affect the results.

HOW TO RUN INTERACTIVELY (VS Code):
The `# %%` markers below split the file into Jupyter-style cells.
Click "Run Cell" above any marker (or press Shift+Enter inside a cell)
to execute just that step in the Interactive Window. Running the file
normally (python learn_chunking.py) still executes everything top to
bottom.

WHY CHUNKING MATTERS:
Unlike learn_embeddings.py, where each Excel row was already a short,
self-contained Q&A pair, a text file is one long block of text. The
embedding model only "reads" the first ~256 tokens (~190 words) of
whatever you give it, so embedding the whole file as one document would
silently throw away everything after the first page. Splitting the text
into smaller chunks means each chunk fits comfortably within that limit,
and search can return the specific passage that's relevant instead of
the entire document.

WHY OVERLAP:
If chunk boundaries land in the middle of an idea, a sentence describing
that idea can be split across two chunks, and neither chunk alone
captures the full meaning. Overlapping the end of one chunk with the
start of the next means most ideas appear intact in at least one chunk.

WHY EPHEMERAL CLIENT:
chromadb.EphemeralClient() keeps everything in memory and never touches
disk. Each scenario below gets its own throwaway client, so we can
re-chunk and re-embed the same text many times with different settings
without collections from one scenario leaking into another (and without
needing to delete persisted collections in between).
"""
# %% Setup: imports and warning noise
import chromadb
from chromadb.utils import embedding_functions
import logging
import warnings

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# %% STEP 1: Load the text file
file_path = "data/sample_article.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()
print(f"Loaded {len(text)} characters / {len(words)} words")

# %% STEP 2: Chunking function — overlapping (or non-overlapping) word windows
def chunk_text(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks

# %% STEP 3: Embedding model (loaded once, reused by every scenario)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def run_scenario(label: str, chunk_size: int, overlap: int, show_chunks: bool = False):
    print("\n" + "=" * 70)
    print(f"SCENARIO: {label}  (chunk_size={chunk_size}, overlap={overlap})")
    print("=" * 70)

    chunks = chunk_text(words, chunk_size, overlap)
    print(f"Split into {len(chunks)} chunks")

    if show_chunks:
        for i, c in enumerate(chunks):
            print(f"\nChunk {i} ({len(c.split())} words):\n{c}")

    # Fresh in-memory DB for this scenario only
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("chunking_demo")
    except Exception:
        pass  # didn't exist yet
    collection = client.create_collection(name="chunking_demo", embedding_function=emb_fn)
    collection.add(
        documents=chunks,
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    results = collection.query(query_texts=TEST_QUERIES, n_results=2)

    for query, docs, distance, metadata in zip(TEST_QUERIES, results['documents'], results['distances'], results['metadatas']):
        print(f"\nQuery: {query}")
        for doc, dist, meta in zip(docs, distance, metadata):
            print(f"  distance={dist:.4f}  chunk={meta['chunk_index']}  |  {doc[:120]}...")

# %% STEP 4: Driver — embed, store, and query one chunking scenario
TEST_QUERIES = [
    "How did graphical interfaces change computing?",
    "What role does artificial intelligence play in future computers?",
    "How does faster networking affect cloud computing on personal devices?",
]

# %% STEP 5: Scenario A — baseline (chunk bigger than any sentence, with overlap)
run_scenario("Baseline: 200-word chunks, 30-word overlap", chunk_size=200, overlap=30)

# %% STEP 6: Scenario B — no overlap (compare against the baseline above)
# Same chunk size as the baseline, but overlap=0. Watch for queries whose
# best-matching idea now sits right on a chunk boundary and gets split
# between two chunks instead of appearing whole in one.
run_scenario("No overlap: 100-word chunks, 0-word overlap", chunk_size=100, overlap=0)

# %% STEP 7: Scenario C — chunk size smaller than the longest sentence
# The article's longest sentences run 25-30+ words. With 12-word chunks and
# no overlap, those sentences get cut into 2-3 pieces, each missing part of
# the idea. show_chunks=True prints every chunk so you can see the cuts.
run_scenario("Tiny chunks: 12-word chunks, 0-word overlap", chunk_size=12, overlap=0, show_chunks=False)

# %%
