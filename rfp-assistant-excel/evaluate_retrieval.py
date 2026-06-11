# ============================================================
# Retrieval Evaluation Harness
# Measures whether the right knowledge-base entry reaches the
# LLM - deterministic, fast (no LLM calls), run after any change
# to models, thresholds or top_k to compare scores objectively.
# ============================================================
"""
HOW IT WORKS
------------
GOLDEN_SET holds (test query, expected entry) pairs:

- The test query is phrased DIFFERENTLY from the stored question,
  the way a real user would ask it.
- The expected entry is a unique substring of the stored question
  (substring, so long numbered questions stay readable here).
- expected=None marks a NEGATIVE case: the knowledge base cannot
  answer it, so the pipeline should refuse (confidence gate).

For each query we run the real retrieve + rerank + gate logic from
rag_with_rerank.py and check:

- HIT:  did the expected entry survive the confidence gate?
        (if it did not reach the LLM, nothing downstream can help)
- RANK: where did the reranker place it? (1 = best)
- Negative cases pass when the gate rejects ALL candidates.

Tuning CONFIDENCE_THRESHOLD is a balance: lower it and more
positives pass, but negatives start slipping through. The right
value passes both groups.
"""

import io
from contextlib import redirect_stdout

# Importing runs the module's setup code (loads models, opens the
# vector DB) but NOT its demo loop, which is guarded by
# `if __name__ == "__main__"` - similar to a Java main() not running
# when you merely reference the class.
from rag_with_rerank import retrieve, rerank, CONFIDENCE_THRESHOLD, MAX_CONTEXTS

GOLDEN_SET = [
    # (query as a user might phrase it,        unique substring of the expected stored question)
    ("Do you support single sign on?",          "Do you support SSO?"),
    ("Is there an API we can use?",             "Do you provide API access?"),
    ("How can external systems integrate with your application?",
                                                "How can external system integrate"),
    ("How frequently do you release new versions?",
                                                "How often do you upgrade your application?"),
    ("Can customers decide when upgrades are applied?",
                                                "choice on when to accept the upgrades"),
    ("If there is a security breach, can we cut off access to our data?",
                                                "disable access to our data in the event of a breach"),
    ("What security certifications do you hold?",
                                                "2.9 List any nationally recognized industry certifications"),
    ("Can your solution scale as our usage grows?",
                                                "5.8 Scalability and Customization"),
    # Negative cases: not answerable from the knowledge base
    ("Can your system be hosted on-premises?",  None),
    ("Do you offer in-person training at our office in every country?", None),
    ("What is your favorite color?",            None),
]


def run_pipeline_quietly(query: str) -> list:
    """Run retrieve + rerank without their demo prints cluttering
    the report. redirect_stdout sends print() output into a throwaway
    buffer for the duration of the `with` block."""
    with redirect_stdout(io.StringIO()):
        candidates = retrieve(query)
        ranked = rerank(query, candidates)
    return ranked


def find_rank(ranked: list, expected: str):
    """1-based position of the expected entry in the reranked list,
    or None if it was not retrieved at all."""
    for position, (score, candidate) in enumerate(ranked, start=1):
        if expected in candidate["question"]:
            return position
    return None


def evaluate():
    hits = 0
    rank_one = 0
    positives = [case for case in GOLDEN_SET if case[1] is not None]
    negatives = [case for case in GOLDEN_SET if case[1] is None]

    print(f"Evaluating {len(positives)} positive and {len(negatives)} negative queries")
    print(f"(CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}, MAX_CONTEXTS={MAX_CONTEXTS})")

    print("\n--- Positive cases: expected entry must pass the gate ---")
    for query, expected in positives:
        ranked = run_pipeline_quietly(query)
        gated = [(s, c) for s, c in ranked if s >= CONFIDENCE_THRESHOLD][:MAX_CONTEXTS]

        rank = find_rank(ranked, expected)
        hit = any(expected in c["question"] for s, c in gated)

        if hit:
            hits += 1
            if rank == 1:
                rank_one += 1
            print(f"  PASS (rank {rank})  {query}")
        elif rank is not None:
            expected_score = ranked[rank - 1][0]
            print(f"  FAIL (rank {rank}, score {expected_score:.2f} gated out)  {query}")
        else:
            print(f"  FAIL (not in top {len(ranked)} retrieved)  {query}")

    print("\n--- Negative cases: gate must reject every candidate ---")
    refusals = 0
    for query, _ in negatives:
        ranked = run_pipeline_quietly(query)
        gated = [(s, c) for s, c in ranked if s >= CONFIDENCE_THRESHOLD]

        if not gated:
            refusals += 1
            print(f"  PASS (refused)  {query}")
        else:
            score, candidate = gated[0]
            print(f"  FAIL (accepted [{score:.2f}] '{candidate['question'][:60]}')  {query}")

    print("\n--- Summary ---")
    print(f"  Hit rate:          {hits}/{len(positives)}")
    print(f"  Ranked #1:         {rank_one}/{len(positives)}")
    print(f"  Correct refusals:  {refusals}/{len(negatives)}")


if __name__ == "__main__":
    evaluate()
