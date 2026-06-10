import chromadb
from chromadb.utils import embedding_functions
import os

print("--- Starting ChromaDB & Embedding Sanity Check ---")

try:
    # 1. Initialize an Ephemeral Client (In-memory, doesn't save files)
    # This is perfect for a quick test to see if the libraries are working.
    client = chromadb.EphemeralClient()
    
    # 2. Set up the Embedding Function 
    # This downloads the model from HuggingFace to your local machine (first time only)
    print("Loading embedding model (all-mpnet-base-v2)...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    
    # 3. Create a temporary test collection
    collection = client.create_collection(name="test_collection", embedding_function=emb_fn)
    
    # 4. Add test data
    # We are testing if it can link the concept of 'SSO' to 'Authentication'
    print("Adding sample data to local vector store...")
    collection.add(
        ids=["id1", "id2"],
        documents=[
            "Our platform supports Single Sign-On (SSO) via SAML 2.0 and OIDC.", 
            "The system uses an Aurora Postgres database hosted on AWS."
        ]
    )
    
    # 5. Perform a Semantic Search
    query = "Tell me about your authentication methods"
    print(f"Querying: '{query}'")
    
    results = collection.query(query_texts=[query], n_results=1)
    
    # 6. Display Results
    matched_doc = results['documents'][0][0]
    distance = results['distances'][0][0]
    
    # A distance under 0.6 usually indicates a strong semantic match
    confidence = round((1 - distance) * 100, 2)

    print("\n--- RESULTS ---")
    print(f"Top Match Found: {matched_doc}")
    print(f"Confidence Score: {confidence}%")
    print(f"Vector Distance: {distance}")
    
    if "SSO" in matched_doc:
        print("\n✅ SUCCESS: The model successfully linked 'Authentication' to 'SSO'!")
    else:
        print("\n⚠️ PARTIAL SUCCESS: The code ran, but the match wasn't what we expected.")

except Exception as e:
    print(f"\n❌ ERROR: The test failed.")
    print(f"Details: {e}")