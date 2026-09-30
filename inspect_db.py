"""Inspect the contents of the ChromaDB collection."""

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME


client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Collection: {collection.name}")
print(f"Total chunks: {collection.count()}")

results = collection.get(
    limit=3,
    include=["documents", "metadatas"],
)

for index, document in enumerate(results["documents"]):
    print("\n" + "=" * 60)
    print(f"Example {index + 1}")
    print("=" * 60)

    print("Metadata:")
    print(results["metadatas"][index])

    print("\nText:")
    print(document[:500])