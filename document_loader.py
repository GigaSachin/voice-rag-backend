def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_into_chunks(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    file_path = "../data/sample.txt"

    text = load_document(file_path)

    print("\n--- DOCUMENT LOADED ---")
    print(f"Total characters: {len(text)}")

    chunks = split_into_chunks(text)

    print(f"Total chunks: {len(chunks)}")

    print("\n--- FIRST CHUNK ---")
    print(chunks[0])