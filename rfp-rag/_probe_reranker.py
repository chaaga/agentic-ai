from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
query = "Can your solution scale as our usage grows?"

variants = [
    # What the eval actually compared
    (1, "Do you provide API access?"),
    (2, "5.8 Scalability and Customization: Is your solution scalable to accommodate the growth of our marketing activities? Can it be customized to meet our specific compliance and brand oversight needs?"),
    (3, "Is your solution scalable to accommodate the growth of our marketing activities? Can it be customized to meet our specific compliance and brand oversight needs?"),
    (4, "Is your solution scalable to accommodate the growth of our marketing activities?"),
    (5, "Is your solution scalable to accommodate the growth of our usage?"),
]

pairs = [(query, text) for _, text in variants]
scores = reranker.predict(pairs)

print(f"Query: {query}\n")
for (index, desc,), score in zip(variants, scores):
    print(f"  [{score:.4f}] {index}: {desc}")
`