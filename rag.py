
# rag.py

"""
rag.py
RAG pipeline utilities - pure Python (no LangChain).
Handles:
- Text splitting
- Embeddings
- FAISS vector store
- Retrieval
"""


from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fileutils


# ---------------------------
# Text Processing
# ---------------------------


# Default chunk size for text splitting
CHUNK_SIZE = 1000
# Number of overlapping characters between chunks
CHUNK_OVERLAP = 100

def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    # Splits text into chunks with specified overlap for better context retrieval
    """
    Split text into chunks with overlap.
    Args:
        text (str): Input text.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        List[str]: List of text chunks.
    """
    # Validate chunk size and overlap
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    chunks = []
    start = 0
    # Loop through text and create overlapping chunks
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks



# ---------------------------
# Embeddings + FAISS
# ---------------------------


# Default model for sentence embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Default path for FAISS index file
FAISS_INDEX_PATH = "faiss.index"
# Default path for document chunks file
DOCS_PATH = "docs.txt"

class VectorStore:
    """
    Handles embedding, indexing, and retrieval using FAISS.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        # Initialize the sentence transformer model and storage for index/docs
        self.model = SentenceTransformer(model_name)
        self.index = None  # FAISS index object
        self.docs = []     # List of document chunks

    def build_index(self, documents: List[str]):
        """
        Create FAISS index from document chunks.
        Args:
            documents (List[str]): List of document chunks.
        """
        self.docs = documents
        # Generate embeddings for all document chunks
        embeddings = self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]  # Dimensionality of embeddings
        self.index = faiss.IndexFlatL2(dim)  # Create FAISS index
        self.index.add(embeddings)           # Add embeddings to index

    def save(self, path: str = FAISS_INDEX_PATH, docs_path: str = DOCS_PATH):
        """
        Save FAISS index and document chunks to disk.
        Args:
            path (str): Path to save FAISS index.
            docs_path (str): Path to save document chunks.
        """
        faiss.write_index(self.index, path)
        # Save document chunks, separated by delimiter
        with open(docs_path, "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(d.replace("\n", " ") + "\n====\n")

    def load(self, path: str = FAISS_INDEX_PATH, docs_path: str = DOCS_PATH):
        """
        Load FAISS index and document chunks from disk.
        Args:
            path (str): Path to load FAISS index.
            docs_path (str): Path to load document chunks.
        """
        self.index = faiss.read_index(path)
        # Load document chunks and split by delimiter
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = [d.strip() for d in f.read().split("\n====\n") if d.strip()]

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents with scores.
        Args:
            query (str): Query string.
            k (int): Number of results.
        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples.
        """
        if self.index is None:
            raise ValueError("FAISS index not built or loaded.")
        # Encode query to embedding
        q_emb = self.model.encode([query], convert_to_numpy=True)
        # Search FAISS index for top-k results
        D, I = self.index.search(q_emb, k)
        # Return list of (document, score) tuples
        return [(self.docs[i], float(D[0][j])) for j, i in enumerate(I[0])]
