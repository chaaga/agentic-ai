from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
query = "Can your solution scale as our usage grows?"

variants = [
    # What the eval actually compared
    ("API question (the rank-1 winner)",
     "Do you provide API access?"),
    ("5.8 full, as stored",
     "5.8 Scalability and Customization: Is your solution scalable to accommodate the growth of our marketing activities? Can it be customized to meet our specific compliance and brand oversight needs?"),
    # Controlled variants of 5.8 to isolate what hurts
    ("5.8 without the '5.8 ...:' prefix",
     "Is your solution scalable to accommodate the growth of our marketing activities? Can it be customized to meet our specific compliance and brand oversight needs?"),
    ("5.8 first sentence only",
     "Is your solution scalable to accommodate the growth of our marketing activities?"),
    ("5.8 first sentence, 'marketing activities' -> 'usage'",
     "Is your solution scalable to accommodate the growth of our usage?"),
    ("Plain short scalability question",
     "Is your solution scalable?"),
    # Sanity check: the query against itself
    ("Identity (query vs query)",
     "Can your solution scale as our usage grows?"),
]

pairs = [(query, text) for _, text in variants]
scores = reranker.predict(pairs)

print(f"Query: {query}\n")
for (label, _), score in zip(variants, scores):
    print(f"  [{score:.4f}] {label}")
