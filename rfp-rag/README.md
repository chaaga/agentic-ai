# RFP Assistant

A local Retrieval-Augmented Generation (RAG) tool that drafts answers to new RFP
(Request for Proposal) questions by matching them against a repository of
previously answered questions, then refining the result with a local LLM.

Pipeline: **embed → vector search (ChromaDB) → cross-encoder rerank → LLM draft (Ollama)**.

### Why a self-hosted LLM (Ollama)?

RFP content is often commercially sensitive — pricing, security posture,
client names, contract terms, etc. To avoid sending this data to a
third-party API, the LLM step runs entirely locally via
[Ollama](https://ollama.com/) (`llama3.1:8b`). Embeddings and reranking also
run locally via `sentence-transformers`. Nothing in the pipeline leaves your
machine.

## Project layout

```
rfp-rag/
├── requirements.txt
├── .python-version          # pinned Python version (3.12)
├── rfp_assistant.py          # main pipeline — run this
├── rag_with_rerank.py         # standalone RAG demo with reranking
├── simple_rag_pipeline.py     # minimal RAG demo (no reranking)
├── learn_embeddings.py        # script for inspecting embeddings
├── learn_chunking.py          # script for exploring chunking strategies
├── evaluate_retrieval.py       # retrieval eval harness (golden set)
├── evaluate_generation.py      # generation eval harness
├── data/
│   ├── repository.xlsx        # past Q&A pairs (knowledge base)
│   └── new_questions.xlsx     # new RFP questions to answer
└── chroma_db/                  # local vector DB (auto-generated, gitignored)
```

## Setup

### 1. Prerequisites

- Python 3.12 (see `.python-version`)
- [Ollama](https://ollama.com/) installed and running locally:
  1. Download and install Ollama for your OS from
     [ollama.com/download](https://ollama.com/download).
  2. Verify it's running:
     ```
     ollama --version
     ```
  3. Pull the model used by this project (`llama3.1:8b`, ~4.7 GB):
     ```
     ollama pull llama3.1:8b
     ```
  4. (Optional) Confirm it works:
     ```
     ollama run llama3.1:8b "Say hello"
     ```

### 2. Create a virtual environment

```
cd rfp-rag
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run will also download the embedding model
(`multi-qa-mpnet-base-dot-v1`) and the cross-encoder reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) from Hugging Face — this can take a
few minutes.

## Usage

1. Place your knowledge base in `data/repository.xlsx`
   with `Question` and `Answer` columns.
2. Place new RFP questions in `data/new_questions.xlsx`
   with a `Question` column.
3. Run the assistant:

```
python rfp_assistant.py
```

This will:
- Re-index `repository.xlsx` into ChromaDB (`chroma_db/`)
- For each new question, retrieve and rerank the closest past answers
- Draft a response, scored as `AUTO-PASS`, `REVIEW NEEDED`, or `LOW CONFIDENCE`
- Write results to `data/rfp_draft_responses.xlsx`

## Other scripts

- **`simple_rag_pipeline.py`** / **`rag_with_rerank.py`** — minimal, standalone
  demos of the RAG pipeline.
- **`learn_embeddings.py`** — ingests the repository and prints out raw
  embedding vectors for inspection.
- **`learn_chunking.py`** — explores chunking strategies on a sample document.
- **`evaluate_retrieval.py`** / **`evaluate_generation.py`** — eval harnesses
  against a golden set of question -> expected-match pairs.

## Notes

- `chroma_db/` is regenerated automatically each time `rfp_assistant.py` runs
  (it clears and re-ingests `repository.xlsx`), so it's safe to delete if you
  want a clean rebuild.
- `data/` and `chroma_db/` are gitignored — they contain local data and
  generated indexes, not source code.
