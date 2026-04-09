"""
document_loader.py
------------------
Handles extraction and preprocessing of text from multiple document formats:
  - PDF  (.pdf)  → pdfplumber (primary) with pypdf fallback
  - Word (.docx) → python-docx
  - Plain text   (.txt)

Each loader returns a list of Page objects:
    {"page_num": int, "text": str, "source": str}
"""

import re
import io
from pathlib import Path
from typing import List, Dict, Any

# ── PDF support ──────────────────────────────────────────────────────────────
try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

# ── DOCX support ──────────────────────────────────────────────────────────────
try:
    from docx import Document as DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      1. Collapse consecutive whitespace / blank lines.
      2. Strip leading/trailing whitespace.
      3. Remove null bytes and control characters.
    """
    if not text:
        return ""
    # Remove null bytes and other control characters (keep newlines & tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces into one (but keep newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(file_obj, source_name: str) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF file.

    Args:
        file_obj  : File-like object (BytesIO) or path string.
        source_name: Human-readable name shown in source citations.

    Returns:
        List of page dicts: [{"page_num": 1, "text": "...", "source": "..."}]
    """
    pages: List[Dict[str, Any]] = []

    # ── Try pdfplumber first (better table / layout handling) ────────────────
    if _PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(file_obj) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    raw = page.extract_text() or ""
                    text = clean_text(raw)
                    if text:
                        pages.append({"page_num": i, "text": text, "source": source_name})
            if pages:
                return pages
        except Exception:
            pass  # fall through to PyPDF2

    # ── Fallback: pypdf ─────────────────────────────────────────────────────
    if _PYPDF_AVAILABLE:
        try:
            # Reset stream position if possible
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            reader = PdfReader(file_obj)
            for i, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                text = clean_text(raw)
                if text:
                    pages.append({"page_num": i, "text": text, "source": source_name})
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF '{source_name}': {e}") from e

    return pages


# ─────────────────────────────────────────────────────────────────────────────
# DOCX Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_docx(file_obj, source_name: str) -> List[Dict[str, Any]]:
    """
    Extract text from a Word (.docx) document.
    DOCX files have no native page concept, so we group paragraphs into
    ~500-word virtual 'pages' to keep the data structure consistent.
    """
    if not _DOCX_AVAILABLE:
        raise ImportError("python-docx is not installed. Run: pip install python-docx")

    try:
        doc = DocxDocument(file_obj)
    except Exception as e:
        raise RuntimeError(f"Cannot open DOCX '{source_name}': {e}") from e

    # Collect non-empty paragraph texts
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Group into virtual pages of ~500 words
    pages: List[Dict[str, Any]] = []
    current_words: List[str] = []
    page_num = 1

    for para in paragraphs:
        current_words.append(para)
        word_count = sum(len(p.split()) for p in current_words)
        if word_count >= 500:
            text = clean_text("\n".join(current_words))
            if text:
                pages.append({"page_num": page_num, "text": text, "source": source_name})
            page_num += 1
            current_words = []

    # Flush remaining text
    if current_words:
        text = clean_text("\n".join(current_words))
        if text:
            pages.append({"page_num": page_num, "text": text, "source": source_name})

    return pages


# ─────────────────────────────────────────────────────────────────────────────
# TXT Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_txt(file_obj, source_name: str) -> List[Dict[str, Any]]:
    """
    Extract text from a plain-text file.
    Splits into virtual pages of ~500 words.
    """
    try:
        raw = file_obj.read()
        # Decode bytes if necessary
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"Cannot read TXT '{source_name}': {e}") from e

    # Split into ~500-word virtual pages
    words = raw.split()
    pages: List[Dict[str, Any]] = []
    page_size = 500

    for i in range(0, len(words), page_size):
        chunk_words = words[i : i + page_size]
        text = clean_text(" ".join(chunk_words))
        if text:
            pages.append({
                "page_num": (i // page_size) + 1,
                "text": text,
                "source": source_name,
            })

    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_document(file_obj, file_name: str) -> List[Dict[str, Any]]:
    """
    Detect file type from its extension and dispatch to the correct loader.

    Args:
        file_obj  : File-like object (e.g., Streamlit's UploadedFile or BytesIO).
        file_name : Original filename including extension.

    Returns:
        List of page dicts with keys: page_num, text, source.

    Raises:
        ValueError if the file extension is not supported.
    """
    ext = Path(file_name).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_obj, file_name)
    elif ext == ".docx":
        return load_docx(file_obj, file_name)
    elif ext in (".txt", ".md"):
        return load_txt(file_obj, file_name)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            "Supported formats: .pdf, .docx, .txt, .md"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly: python document_loader.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path-to-file>")
        sys.exit(1)

    path = Path(sys.argv[1])
    with open(path, "rb") as f:
        pages = load_document(f, path.name)

    print(f"Loaded {len(pages)} page(s) from '{path.name}'")
    for p in pages[:3]:
        preview = p["text"][:200].replace("\n", " ")
        print(f"  Page {p['page_num']}: {preview}…")
