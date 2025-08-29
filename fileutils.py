"""
fileutils.py
Utility functions for file handling in the RAG pipeline.
Supports PDF, text, and markdown file loading.
"""

import os
from typing import List
from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_text(path: str) -> str:
    """Read a plain text or markdown file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_knowledge_base(folder: str = "knowledge_base") -> List[str]:
    """
    Load all supported documents from a folder.
    Supports: .pdf, .txt, .md
    Returns a list of raw texts.
    """
    texts = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.lower().endswith(".pdf"):
            texts.append(load_pdf(path))
        elif file.lower().endswith((".txt", ".md")):
            texts.append(load_text(path))
    return texts
