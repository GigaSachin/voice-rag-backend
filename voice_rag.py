from google import genai
from dotenv import load_dotenv
import os

from retriever import search_knowledge_base


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")


# ==========================================
# 2. GEMINI CLIENT
# ==========================================

client = genai.Client(api_key=api_key)


# ==========================================
# 3. GENERATE ANSWER
# ==========================================

def generate_answer(question: str):

    # --------------------------------------
    # Retrieve relevant information
    # --------------------------------------

    results = search_knowledge_base(
        question,
        top_k=3
    )


    # --------------------------------------
    # CASE 1: No knowledge-base results
    # --------------------------------------

    if not results:

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question clearly and accurately
using your general knowledge.

User Question:
{question}

Give a concise and useful answer.
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return {
            "answer": response.text,
            "sources": []
        }


    # --------------------------------------
    # Check similarity score
    # --------------------------------------

    best_score = results[0]["score"]


    # --------------------------------------
    # CASE 2: Relevant knowledge found
    # --------------------------------------

    if best_score >= 0.50:

        context = "\n\n".join(
            result["text"]
            for result in results
        )

        prompt = f"""
You are a helpful Voice RAG assistant.

Answer the user's question using the information
provided in the knowledge base context below.

Prefer the knowledge base information when it is
relevant to the question.

Knowledge Base Context:
-----------------------
{context}
-----------------------

User Question:
{question}

Give a clear and concise answer.
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return {
            "answer": response.text,
            "sources": results
        }


    # --------------------------------------
    # CASE 3: Knowledge base not relevant
    # --------------------------------------

    prompt = f"""
You are a helpful AI assistant.

The knowledge base does not contain relevant
information for this question.

Answer the user's question using your general
knowledge.

Do NOT pretend that the knowledge base contains
information that it does not contain.

User Question:
{question}

Give a clear and concise answer.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources": []
    }


# ==========================================
# 4. LOCAL TESTING
# ==========================================

if __name__ == "__main__":

    question = input("Ask your question: ")

    result = generate_answer(question)

    print("\n--- ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCES ---")

    if result["sources"]:

        for source in result["sources"]:

            print(
                f"\nSimilarity: {source['score']:.4f}"
            )

            print(source["text"])

    else:

        print("General Gemini knowledge used.")