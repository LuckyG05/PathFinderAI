import re
from typing import BinaryIO, List, Union
from pypdf import PdfReader


def extract_text_from_pdf(file_source: Union[str, BinaryIO]) -> str:
    """
    Extracts and cleans text from a PDF file path or an uploaded file stream (e.g., Streamlit uploader).
    Tags each non-empty page with [Page N] so downstream LLM responses can cite source pages.
    """
    if hasattr(file_source, "seek"):
        file_source.seek(0)

    reader = PdfReader(file_source)
    pages_text: List[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        if cleaned:
            pages_text.append(f"[Page {page_num}] {cleaned}")

    return "\n\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Splits long document text into overlapping character chunks so sentences/concepts
    are not cut off at chunk boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text or not text.strip():
        return []

    cleaned_text = text.strip()
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: List[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(cleaned_text[start:end].strip())
        if end == text_length:
            break
        start += chunk_size - overlap

    return chunks


def _tokenize(text: str) -> set[str]:
    """Extracts lowercase alphanumeric words (3+ chars) for lexical relevance scoring."""
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "have", "has", "had", "not", "but", "what", "all", "can",
        "who", "how", "why", "into", "about", "your", "you",
    }
    words = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    return {w for w in words if w not in stop_words}


def retrieve_relevant_context(
    source_text: str,
    query: str,
    top_k: int = 3,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> str:
    """
    Splits the source document into chunks and retrieves the top_k most relevant chunks
    for a given topic or student question.
    """
    if top_k <= 0 or not source_text or not source_text.strip():
        return ""

    chunks = chunk_text(source_text, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) <= top_k:
        return "\n\n---\n\n".join(chunks)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return "\n\n---\n\n".join(chunks[:top_k])

    scored_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_tokens = _tokenize(chunk)
        overlap_score = len(query_tokens.intersection(chunk_tokens))
        scored_chunks.append((overlap_score, -idx, chunk))

    scored_chunks.sort(reverse=True)
    selected = [chunk for _, _, chunk in scored_chunks[:top_k]]
    return "\n\n---\n\n".join(selected)
