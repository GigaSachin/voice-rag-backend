from pathlib import Path
import json


# ---------------------------------------------------------
# KNOWLEDGE BASE PATH
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# If retriever.py is inside src/
# and data/ is in the project root:
KNOWLEDGE_BASE_PATH = BASE_DIR.parent / "data" / "knowledge_base.json"


# ---------------------------------------------------------
# LOAD KNOWLEDGE BASE
# ---------------------------------------------------------

def load_knowledge_base():
    """
    Load knowledge_base.json from the project's data folder.
    """

    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at: {KNOWLEDGE_BASE_PATH}"
        )

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# SEARCH KNOWLEDGE BASE
# ---------------------------------------------------------

def search_knowledge_base(question, top_k=3):
    """
    Search the knowledge base and return the most relevant
    documents.

    This is a simple keyword-based retriever.
    """

    knowledge_base = load_knowledge_base()

    # Handle different possible JSON structures
    if isinstance(knowledge_base, dict):

        if "documents" in knowledge_base:
            documents = knowledge_base["documents"]

        elif "data" in knowledge_base:
            documents = knowledge_base["data"]

        elif "chunks" in knowledge_base:
            documents = knowledge_base["chunks"]

        else:
            documents = [knowledge_base]

    elif isinstance(knowledge_base, list):
        documents = knowledge_base

    else:
        documents = []

    question_words = set(
        question.lower()
        .strip()
        .split()
    )

    scored_documents = []

    for document in documents:

        # -------------------------------------------------
        # Extract text from different possible formats
        # -------------------------------------------------

        if isinstance(document, str):
            text = document

        elif isinstance(document, dict):

            text = (
                document.get("text")
                or document.get("content")
                or document.get("chunk")
                or document.get("page_content")
                or ""
            )

        else:
            continue

        if not text:
            continue

        # -------------------------------------------------
        # Calculate simple keyword score
        # -------------------------------------------------

        text_lower = text.lower()

        score = 0

        for word in question_words:

            if len(word) > 2 and word in text_lower:
                score += 1

        scored_documents.append(
            {
                "text": text,
                "score": score,
                "document": document,
            }
        )

    # -----------------------------------------------------
    # Sort by relevance
    # -----------------------------------------------------

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # -----------------------------------------------------
    # Return top results
    # -----------------------------------------------------

    results = []

    for item in scored_documents[:top_k]:

        original_document = item["document"]

        if isinstance(original_document, dict):

            result = {
                "text": item["text"],
                "score": item["score"],
            }

            # Preserve useful metadata if available
            if "source" in original_document:
                result["source"] = original_document["source"]

            if "title" in original_document:
                result["title"] = original_document["title"]

            results.append(result)

        else:

            results.append(
                {
                    "text": item["text"],
                    "score": item["score"],
                }
            )

    return results