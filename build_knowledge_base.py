import json
from document_loader import load_document, split_into_chunks
from embedding import create_embeddings


DOCUMENT_PATH = "../data/sample.txt"
OUTPUT_PATH = "../data/knowledge_base.json"


def build_knowledge_base():

    print("Loading document...")

    text = load_document(DOCUMENT_PATH)

    print("Splitting document into chunks...")

    chunks = split_into_chunks(text)

    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")

    embeddings = create_embeddings(chunks)

    knowledge_base = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        knowledge_base.append({
            "id": i,
            "text": chunk,
            "embedding": embedding
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(knowledge_base, file)

    print("\nKnowledge base created successfully!")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_knowledge_base()