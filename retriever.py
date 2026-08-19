from pathlib import Path
import json
import re


# ============================================================
# KNOWLEDGE BASE PATH
# ============================================================

# retriever.py ke folder ko base directory maanenge
BASE_DIR = Path(__file__).resolve().parent

# Expected structure:
#
# backend/
# ├── main.py
# ├── rag.py
# ├── retriever.py
# └── data/
#     └── knowledge_base.json

KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.json"


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():
    """
    Load knowledge_base.json from the backend/data folder.
    """

    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at: {KNOWLEDGE_BASE_PATH}"
        )

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        knowledge_base = json.load(file)

    return knowledge_base


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Convert text into a simple searchable format.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# EXTRACT DOCUMENTS
# ============================================================

def extract_documents(knowledge_base):
    """
    Convert different possible knowledge-base JSON structures
    into a simple list of documents.
    """

    documents = []

    # --------------------------------------------------------
    # Case 1:
    # [
    #   {"text": "...", "source": "..."},
    #   {"text": "...", "source": "..."}
    # ]
    # --------------------------------------------------------

    if isinstance(knowledge_base, list):

        for item in knowledge_base:

            if isinstance(item, dict):

                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("document")
                    or item.get("page_content")
                    or ""
                )

                if text:
                    documents.append({
                        "text": str(text),
                        "source": item.get("source", ""),
                        "title": item.get("title", ""),
                        "document": item
                    })

            elif isinstance(item, str):

                documents.append({
                    "text": item,
                    "source": "",
                    "title": "",
                    "document": item
                })

    # --------------------------------------------------------
    # Case 2:
    # {
    #   "documents": [...]
    # }
    # --------------------------------------------------------

    elif isinstance(knowledge_base, dict):

        if isinstance(knowledge_base.get("documents"), list):

            for item in knowledge_base["documents"]:

                if isinstance(item, dict):

                    text = (
                        item.get("text")
                        or item.get("content")
                        or item.get("document")
                        or item.get("page_content")
                        or ""
                    )

                    if text:
                        documents.append({
                            "text": str(text),
                            "source": item.get("source", ""),
                            "title": item.get("title", ""),
                            "document": item
                        })

                elif isinstance(item, str):

                    documents.append({
                        "text": item,
                        "source": "",
                        "title": "",
                        "document": item
                    })

        # ----------------------------------------------------
        # Case 3:
        # {
        #   "chunks": [...]
        # }
        # ----------------------------------------------------

        elif isinstance(knowledge_base.get("chunks"), list):

            for item in knowledge_base["chunks"]:

                if isinstance(item, dict):

                    text = (
                        item.get("text")
                        or item.get("content")
                        or item.get("document")
                        or item.get("page_content")
                        or ""
                    )

                    if text:
                        documents.append({
                            "text": str(text),
                            "source": item.get("source", ""),
                            "title": item.get("title", ""),
                            "document": item
                        })

                elif isinstance(item, str):

                    documents.append({
                        "text": item,
                        "source": "",
                        "title": "",
                        "document": item
                    })

        # ----------------------------------------------------
        # Case 4:
        # Dictionary containing data directly
        # ----------------------------------------------------

        else:

            for key, value in knowledge_base.items():

                if isinstance(value, str):

                    documents.append({
                        "text": value,
                        "source": "",
                        "title": str(key),
                        "document": {
                            "key": key,
                            "text": value
                        }
                    })

                elif isinstance(value, dict):

                    text = (
                        value.get("text")
                        or value.get("content")
                        or value.get("document")
                        or value.get("page_content")
                        or ""
                    )

                    if text:

                        documents.append({
                            "text": str(text),
                            "source": value.get("source", ""),
                            "title": value.get("title", str(key)),
                            "document": value
                        })

    return documents


# ============================================================
# KEYWORD SEARCH
# ============================================================

def calculate_score(question, document_text):
    """
    Simple keyword-based relevance score.
    """

    question = normalize_text(question)
    document_text = normalize_text(document_text)

    if not question or not document_text:
        return 0.0

    # Extract words
    question_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", question)
    )

    document_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", document_text)
    )

    if not question_words:
        return 0.0

    # Number of matching words
    matched_words = question_words.intersection(document_words)

    score = len(matched_words) / len(question_words)

    # Exact question match gets a boost
    if question in document_text:
        score += 1.0

    return score


# ============================================================
# SEARCH KNOWLEDGE BASE
# ============================================================

def search_knowledge_base(question, top_k=3):
    """
    Search the knowledge base and return the most relevant
    documents.
    """

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question or not str(question).strip():

        return []

    # --------------------------------------------------------
    # Load knowledge base
    # --------------------------------------------------------

    knowledge_base = load_knowledge_base()

    # --------------------------------------------------------
    # Convert KB into documents
    # --------------------------------------------------------

    documents = extract_documents(knowledge_base)

    if not documents:
        return []

    # --------------------------------------------------------
    # Score every document
    # --------------------------------------------------------

    scored_documents = []

    for document in documents:

        text = document["text"]

        score = calculate_score(
            question,
            text
        )

        scored_documents.append({
            "document": document.get("document", document),
            "text": text,
            "score": score,
            "source": document.get("source", ""),
            "title": document.get("title", "")
        })

    # --------------------------------------------------------
    # Sort by score
    # --------------------------------------------------------

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Return top results
    # --------------------------------------------------------

    results = []

    for item in scored_documents[:top_k]:

        original_document = item["document"]

        result = {
            "text": item["text"],
            "score": item["score"]
        }

        # Preserve source
        if item.get("source"):
            result["source"] = item["source"]

        elif isinstance(original_document, dict):
            if "source" in original_document:
                result["source"] = original_document["source"]

        # Preserve title
        if item.get("title"):
            result["title"] = item["title"]

        elif isinstance(original_document, dict):
            if "title" in original_document:
                result["title"] = original_document["title"]

        results.append(result)

    return results


# ============================================================
# TEST FUNCTION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Voice RAG - Knowledge Base Retriever")
    print("=" * 60)

    print()
    print("Knowledge base path:")
    print(KNOWLEDGE_BASE_PATH)

    print()

    if KNOWLEDGE_BASE_PATH.exists():

        print("Knowledge base found!")

        try:

            knowledge_base = load_knowledge_base()

            print("Knowledge base loaded successfully.")

            documents = extract_documents(
                knowledge_base
            )

            print(
                f"Documents found: {len(documents)}"
            )

            print()

            question = input(
                "Enter your question: "
            )

            results = search_knowledge_base(
                question,
                top_k=3
            )

            print()
            print("Search results:")
            print("-" * 60)

            for index, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"\nResult {index}"
                )

                print(
                    f"Score: {result['score']}"
                )

                print(
                    f"Text: {result['text'][:500]}"
                )

                if "source" in result:
                    print(
                        f"Source: {result['source']}"
                    )

                if "title" in result:
                    print(
                        f"Title: {result['title']}"
                    )

        except Exception as error:

            print()
            print(
                "ERROR:",
                error
            )

    else:

        print(
            "ERROR: knowledge_base.json not found!"
        )

        print()
        print(
            "Expected location:"
        )

        print(
            KNOWLEDGE_BASE_PATH
        )