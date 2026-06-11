# ============================================================
# Generation Evaluation - LLM as Judge
# Grades the FINAL answers the pipeline produces, using a second
# LLM (a different model family, to reduce self-preference bias).
# Slow and statistical - run occasionally; run the deterministic
# evaluate_retrieval.py after every change.
# ============================================================
"""
WHAT IT MEASURES
----------------
For each positive golden-set case, the full pipeline runs (retrieve,
rerank, gate, generate) and the judge answers two questions:

- FAITHFUL:  is every claim in the answer supported by the context
             the pipeline retrieved? (catches hallucination)
- CORRECT:   does the answer convey the same key facts as the
             reference answer stored in the knowledge base?
             (catches answering from the wrong context)

For negative cases (knowledge base cannot answer), the ideal outcome
is a refusal at the confidence gate (deterministic, free). If junk
slips past the gate, the judge checks whether the LLM at least
declined rather than fabricating an answer.

CAVEATS
-------
- The judge is itself a small model: treat verdicts as noisy signals.
  Aggregate counts matter; single verdicts deserve a manual look.
- Judges favor long, confident answers and are lenient with scores;
  binary yes/no verdicts resist this better than 1-5 ratings.
- Spot-check the judge's reasons before trusting the numbers.

USAGE
-----
    python evaluate_generation.py        # full golden set (slow)
    python evaluate_generation.py 2      # first 2 of each group
"""

import io
import json
import re
import sys
from contextlib import redirect_stdout

import ollama

from rag_with_rerank import (
    retrieve, rerank, generate, collection,
    CONFIDENCE_THRESHOLD, MAX_CONTEXTS,
)
from evaluate_retrieval import GOLDEN_SET

# Different model family than the llama3.1 generator on purpose:
# models grade their own family's output too kindly
JUDGE_MODEL = "mistral:7b"

FAITHFULNESS_PROMPT = """You are grading a RAG system's answer.

Question: {query}

Context the system was given:
{context}

Answer the system produced:
{answer}

Is every factual claim in the answer supported by the context?
Reply with EXACTLY one line of JSON: {{"verdict": "yes" or "no", "reason": "<one sentence>"}}"""

CORRECTNESS_PROMPT = """You are grading a RAG system's answer against a reference.

Question: {query}

Reference answer (ground truth):
{reference}

Answer the system produced:
{answer}

Does the system's answer convey the same key facts as the reference?
Reply with EXACTLY one line of JSON: {{"verdict": "yes" or "no", "reason": "<one sentence>"}}"""

REFUSAL_PROMPT = """A question was asked that CANNOT be answered from the system's knowledge base.

Question: {query}

Answer the system produced:
{answer}

Did the system decline to answer (saying the information is not available),
rather than attempting a substantive answer?
Reply with EXACTLY one line of JSON: {{"verdict": "yes" or "no", "reason": "<one sentence>"}}"""


def ask_judge(prompt: str) -> dict:
    """One grading call. Returns {"verdict": ..., "reason": ...}."""
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    text = response["message"]["content"]

    # Judges usually obey "reply with one line of JSON", but not always:
    # pull the first {...} block out of whatever came back
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"verdict": "unparseable", "reason": text[:120]}


def run_pipeline(query: str):
    """Same logic as rag_pipeline(), but returns (context, answer)
    instead of printing. (None, None) means the gate refused."""
    with redirect_stdout(io.StringIO()):
        ranked = rerank(query, retrieve(query))
        good = [(s, c) for s, c in ranked if s >= CONFIDENCE_THRESHOLD][:MAX_CONTEXTS]
        if not good:
            return None, None
        context = "\n\n---\n\n".join(c["answer"] for s, c in good)
        answer = generate(query, context)
    return context, answer


def find_reference_answer(expected: str) -> str:
    """Look up the stored answer for the golden entry, identified by
    the same question substring the retrieval harness uses."""
    for meta in collection.get()["metadatas"]:
        if expected in meta["question"]:
            return meta["answer"]
    raise ValueError(f"No knowledge-base entry matches: {expected!r}")


def evaluate(limit: int = None):
    positives = [case for case in GOLDEN_SET if case[1] is not None][:limit]
    negatives = [case for case in GOLDEN_SET if case[1] is None][:limit]

    print(f"Judging {len(positives)} positive and {len(negatives)} negative queries")
    print(f"(generator: llama3.1:8b, judge: {JUDGE_MODEL})")

    faithful = 0
    correct = 0
    print("\n--- Positive cases ---")
    for query, expected in positives:
        context, answer = run_pipeline(query)
        if answer is None:
            print(f"  GATE REFUSED (no answer to judge)  {query}")
            continue

        f_verdict = ask_judge(FAITHFULNESS_PROMPT.format(
            query=query, context=context, answer=answer))
        c_verdict = ask_judge(CORRECTNESS_PROMPT.format(
            query=query, reference=find_reference_answer(expected), answer=answer))

        if f_verdict["verdict"] == "yes":
            faithful += 1
        if c_verdict["verdict"] == "yes":
            correct += 1

        print(f"  faithful={f_verdict['verdict']:<3} correct={c_verdict['verdict']:<3}  {query}")
        if f_verdict["verdict"] != "yes":
            print(f"      judge on faithfulness: {f_verdict['reason']}")
        if c_verdict["verdict"] != "yes":
            print(f"      judge on correctness:  {c_verdict['reason']}")

    refusals = 0
    print("\n--- Negative cases: system should refuse ---")
    for query, _ in negatives:
        context, answer = run_pipeline(query)
        if answer is None:
            refusals += 1
            print(f"  PASS (refused at gate)  {query}")
            continue

        r_verdict = ask_judge(REFUSAL_PROMPT.format(query=query, answer=answer))
        if r_verdict["verdict"] == "yes":
            refusals += 1
            print(f"  PASS (LLM declined)  {query}")
        else:
            print(f"  FAIL (fabricated)  {query}")
            print(f"      answer was: {answer[:120]}")

    print("\n--- Summary ---")
    print(f"  Faithful:  {faithful}/{len(positives)}")
    print(f"  Correct:   {correct}/{len(positives)}")
    print(f"  Refusals:  {refusals}/{len(negatives)}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    evaluate(limit)
