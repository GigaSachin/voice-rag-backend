from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


def create_embeddings(texts):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    return [embedding.values for embedding in response.embeddings]


if __name__ == "__main__":

    texts = [
        "Voice RAG is a Retrieval Augmented Generation system.",
        "Users can ask questions using their voice."
    ]

    embeddings = create_embeddings(texts)

    print("Embedding successful!")
    print("Number of embeddings:", len(embeddings))
    print("Embedding dimensions:", len(embeddings[0]))
    print("First 5 values:", embeddings[0][:5])