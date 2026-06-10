import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import ollama
import numpy as np

# --- CONFIGURATION ---
REPO_FILE = "data/repository.xlsx"
QUERY_FILE = "data/new_questions.xlsx"
OUTPUT_FILE = "data/rfp_draft_responses.xlsx"

# FIX 1: Use a better embedding model purpose-built for semantic similarity
# "multi-qa-mpnet-base-dot-v1" is trained specifically on Q&A retrieval tasks
# It handles vocabulary mismatch much better than all-mpnet-base-v2
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1"
)

client = chromadb.PersistentClient(path="./chroma_db")

# FIX 2: Use dot product space — matches the model's training objective
collection = client.get_or_create_collection(
    name="rfp_knowledge",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "ip"}  # Inner product = dot product similarity
)

rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


import re

def clean_question(text: str) -> str:
    """
    Remove section prefixes like '5.8', '4.1', 'Q:' from repo questions.
    These add noise to embeddings and hurt retrieval accuracy.
    """
    return re.sub(r'^[\d\.]+\s*', '', text).strip()

def ingest_repository():
    print("Indexing repository...")
    df = pd.read_excel(REPO_FILE)
    questions = df['Question'].astype(str).tolist()
    answers = df['Answer'].astype(str).tolist()

    all_ids = collection.get()['ids']
    if all_ids:
        collection.delete(ids=all_ids)

    documents = []
    metadatas = []
    ids = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        cleaned_q = clean_question(q)

        # Index question + answer together but with cleaned question
        # Cleaning removes section numbers that pollute the embedding
        combined = f"{cleaned_q}\n{a}"

        documents.append(combined)
        metadatas.append({
            "question": q,        # keep original for display
            "clean_question": cleaned_q,
            "answer": a
        })
        ids.append(f"id_{i}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Indexed {len(questions)} entries.")


def normalize_cross_encoder_score(score):
    """Sigmoid normalization for ms-marco cross-encoder logits."""
    sigmoid = 1 / (1 + np.exp(-score))
    return round(sigmoid * 100, 2)


def split_into_questions(query: str) -> list[str]:
    """
    Use LLM to detect and split compound queries into individual questions.
    Returns a list — single question returns a list of one item.
    """
    prompt = (
        f"Extract all individual questions from the text below.\n"
        f"Output each question on a new line, no numbering, no preamble.\n"
        f"If there is only one question, output just that question.\n\n"
        f"Text: {query}\n\n"
        f"Questions:"
    )
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.0, 'num_ctx': 2048}
    )
    raw = response['message']['content'].strip()
    questions = [q.strip() for q in raw.split('\n') if q.strip()]
    return questions if questions else [query]  # fallback to original if split fails


def process_rfp():
    df_new = pd.read_excel(QUERY_FILE)
    results_list = []
    repo_size = collection.count()
    n_results = min(15, repo_size)

    for _, row in df_new.iterrows():
        new_q = str(row['Question'])
        print(f"\nProcessing: {new_q[:80]}...")

        # ── NEW: Detect and split compound questions ──────────────
        sub_questions = split_into_questions(new_q)
        is_compound = len(sub_questions) > 1
        print(f"  Sub-questions detected: {len(sub_questions)}")

        # ── Retrieve best match for EACH sub-question separately ──
        retrieved_contexts = []
        for sub_q in sub_questions:
            search = collection.query(
                query_texts=[sub_q],      # search per sub-question, not whole query
                n_results=n_results
            )
            metadatas = search['metadatas'][0]

            # Rerank candidates for this sub-question
            pairs  = [[sub_q, m['clean_question']] for m in metadatas]
            scores = rerank_model.predict(pairs)

            best_index  = int(np.argmax(scores))
            best_answer = metadatas[best_index]['answer']
            matched_q   = metadatas[best_index]['question']
            raw_score   = scores[best_index]
            confidence  = normalize_cross_encoder_score(raw_score)

            print(f"  [{sub_q[:50]}]")
            print(f"    Matched   : {matched_q[:60]}")
            print(f"    Confidence: {confidence}%")

            retrieved_contexts.append({
                "sub_question":     sub_q,
                "matched_question": matched_q,
                "answer":           best_answer,
                "raw_score":        raw_score,
                "confidence":       confidence
            })

        # ── Determine overall confidence ──────────────────────────
        # Use the LOWEST confidence across all sub-questions
        # since the weakest match determines overall reliability
        overall_confidence  = min(ctx['confidence'] for ctx in retrieved_contexts)
        overall_raw_score   = min(ctx['raw_score']  for ctx in retrieved_contexts)
        matched_question    = " | ".join(ctx['matched_question'] for ctx in retrieved_contexts)
        best_answer         = " | ".join(ctx['answer']           for ctx in retrieved_contexts)

        status = (
            "🟢 AUTO-PASS"     if overall_confidence > 80 else
            "🟡 REVIEW NEEDED" if overall_confidence > 50 else
            "🔴 LOW CONFIDENCE"
        )

        # ── Generate final draft ──────────────────────────────────
        if overall_confidence >= 80 and not is_compound:
            # Single high-confidence question — use repo answer directly
            final_draft = retrieved_contexts[0]['answer']

        elif overall_confidence >= 50 or is_compound:
            # Compound question OR medium confidence — LLM combines/rewords
            combined_context = "\n\n".join([
                f"Q: {ctx['sub_question']}\nA: {ctx['answer']}"
                for ctx in retrieved_contexts
            ])
            total_answer_words = sum(
                len(ctx['answer'].split()) for ctx in retrieved_contexts
            )
            prompt = (
                f"You are copying and lightly rewording past answers to fit new questions.\n"
                f"Your output must be based ONLY on the provided answers below.\n"
                f"Do NOT add any new information. Do NOT make the response longer.\n"
                f"Address each question clearly using only the context provided.\n\n"
                f"New Question(s): {new_q}\n\n"
                f"Context (your only source):\n{combined_context}\n\n"
                f"Response (same length as context, no new facts):"
            )
            response = ollama.chat(
                model='llama3.1:8b',
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.0,
                    'num_predict': max(50, total_answer_words + 20),
                    'num_ctx': 2048,
                    'num_gpu': 99
                }
            )
            final_draft = response['message']['content'].strip()

        else:
            # Low confidence across all sub-questions
            final_draft = f"[LOW CONFIDENCE — MANUAL REVIEW REQUIRED]\n\n{best_answer}"

        results_list.append({
            "New Question":          new_q,
            "Drafted Answer":        final_draft,
            "Confidence Score":      round(float(overall_confidence), 2),
            "Review Status":         status,
            "Matched Past Question": matched_question,
            "Original Past Answer":  best_answer,
            "Raw Rerank Score":      round(float(overall_raw_score), 4),
            "Sub Questions":         str(sub_questions),        # useful for debugging
            "Is Compound":           "Yes" if is_compound else "No"
        })

    output_df = pd.DataFrame(results_list)
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nDone. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    ingest_repository()
    process_rfp()