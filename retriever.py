import json
import math

from embedding import create_embeddings


KNOWLEDGE_BASE_PATH = "../data/knowledge_base.json"


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


def load_knowledge_base():

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def search_knowledge_base(question, top_k=3):

    knowledge_base = load_knowledge_base()

    question_embedding = create_embeddings([question])[0]

    results = []

    for item in knowledge_base:

        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        results.append({
            "text": item["text"],
            "score": score
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":

    question = input("Ask your question: ")

    results = search_knowledge_base(question)

    print("\n--- SEARCH RESULTS ---")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])