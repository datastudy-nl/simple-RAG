"""
main.py
Main entry point - uses rag.py and Ollama API for RAG.
"""

import requests

import fileutils
import rag

# ---------------------------
# Ollama API constants
# ---------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3"


def query_ollama(model: str, prompt: str) -> str:
    """
    Send prompt to Ollama API and return response.
    Args:
        model (str): Model name.
        prompt (str): Prompt string.
    Returns:
        str: LLM response.
    """
    payload = {"model": model, "prompt": prompt}
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=30)
    output = ""
    for line in resp.iter_lines():
        if line:
            data = line.decode("utf-8")
            if '"response":"' in data:
                part = data.split('"response":"')[1].split('"')[0]
                output += part
    return output.strip()


# ---------------------------
# Main Flow
# ---------------------------


def main():
    """
    Main RAG pipeline flow.
    """
    # Step 1: Load knowledge base
    texts = fileutils.load_knowledge_base("knowledge_base")
    all_chunks = []
    for t in texts:
        all_chunks.extend(rag.split_text(t))

    # Step 2: Build embeddings + FAISS index
    store = rag.VectorStore()
    store.build_index(all_chunks)

    # Step 3: Interactive Q&A
    print(f"RAG pipeline ready. Using Ollama (model: '{OLLAMA_MODEL}').")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.strip().lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break

        # Retrieve relevant docs
        retrieved = store.search(query, k=3)
        context = "\n\n".join([doc for doc, _ in retrieved])

        # Build RAG prompt
        prompt = (
            "You are a helpful assistant.\n"
            "Use the following context to answer the question.\n"
            "If it is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        # Call Ollama
        answer = query_ollama(OLLAMA_MODEL, prompt)

        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources ---")
        for doc, score in retrieved:
            print("-", doc[:200].replace("\n", " "), f"... (score: {score:.4f})")


if __name__ == "__main__":
    main()
