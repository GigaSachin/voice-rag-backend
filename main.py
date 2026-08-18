from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag import generate_answer

app = FastAPI(title="Voice RAG API")

# Allow Lovable/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Voice RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(data: dict):

    question = data.get("question", "").strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    result = generate_answer(question)

    return result
