"""
rag_pipeline.py
---------------
Orchestrates the full Retrieval-Augmented Generation (RAG) pipeline:

  1. Document ingestion  → load pages, split into chunks, embed, index.
  2. Question answering  → embed query, retrieve chunks, call LLM.
  3. Summarisation       → gather all chunks for a doc, call LLM.

LLM Backend
-----------
The system calls the local Ollama model (llama3) by default.
You can swap it for any other Ollama model by changing `_call_llm`.
Ensure Ollama is installed and the model is pulled:
    ollama pull llama3
"""

from __future__ import annotations

import logging
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from document_loader import load_document
from embeddings import get_embedder
from vector_store import FAISSVectorStore, chunk_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE    = 400   # words per chunk
DEFAULT_CHUNK_OVERLAP = 60    # overlap words
DEFAULT_TOP_K         = 5     # chunks retrieved per query
MAX_CONTEXT_CHARS     = 6000  # hard cap on context sent to LLM
LLM_MODEL             = "llama3"   # local Ollama model

# How many results to retrieve from FAISS before applying MMR
MMR_CANDIDATES        = 20
MMR_LAMBDA            = 0.7

# How many conversation turns to keep in history before summarizing
MAX_HISTORY_TURNS = 10  # 5 question-answer pairs
SUMMARY_HISTORY_PROMPT = textwrap.dedent("""
    Summarize the following conversation history concisely, capturing:
    - Key questions asked by the user
    - Important information provided in answers
    - Any evolving context or topics of interest

    Keep the summary under 200 words. Focus on factual information.

    Conversation:
    {history}

    Summary:
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM caller
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """
    Check if Ollama server is running and accessible.

    Returns:
        bool: True if Ollama is running, False otherwise.
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """
    Send a prompt to the local Ollama model and return the text response.

    Raises:
        RuntimeError if the call fails.
    """
    # Check if Ollama is running first
    if not check_ollama_running():
        raise RuntimeError(
            "Ollama server is not running. Please start Ollama and ensure llama3 model is available:\n"
            "1. Install Ollama: https://ollama.ai\n"
            "2. Start Ollama: ollama serve\n"
            "3. Pull model: ollama pull llama3"
        )

    try:
        import ollama
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"num_predict": max_tokens}
        )
        return response["message"]["content"].strip()
    except ImportError as exc:
        raise ImportError(
            "ollama SDK not installed. Run: pip install ollama"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def _call_llm_stream(system_prompt: str, user_prompt: str, max_tokens: int = 1024):
    """
    Send a prompt to the local Ollama model and yield streaming text response.

    Yields:
        str: Incremental chunks of the response text.

    Raises:
        RuntimeError if the call fails.
    """
    # Check if Ollama is running first
    if not check_ollama_running():
        raise RuntimeError(
            "Ollama server is not running. Please start Ollama and ensure llama3 model is available:\n"
            "1. Install Ollama: https://ollama.ai\n"
            "2. Start Ollama: ollama serve\n"
            "3. Pull model: ollama pull llama3"
        )

    try:
        import ollama
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"num_predict": max_tokens},
            stream=True
        )
        for chunk in stream:
            if chunk and chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]
    except ImportError as exc:
        raise ImportError(
            "ollama SDK not installed. Run: pip install ollama"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"LLM streaming call failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = textwrap.dedent("""
    You are DocAssist, a precise document question-answering assistant.

    Rules:
    1. Answer ONLY from the provided document context below.
    2. If the answer cannot be found in the context, reply exactly:
       "I could not find an answer in the uploaded documents."
    3. Be concise but complete. Use bullet points when listing items.
    4. Never fabricate information.
""").strip()

QA_USER_TEMPLATE = textwrap.dedent("""
    === DOCUMENT CONTEXT ===
    {context}

    === CONVERSATION HISTORY ===
    {history}

    === QUESTION ===
    {question}

    Answer:
""").strip()

SUMMARY_SYSTEM_PROMPT = textwrap.dedent("""
    You are a professional document summariser.
    Produce a clear, structured summary that captures:
      • The main topic / purpose of the document.
      • Key points, findings, or arguments (use bullet points).
      • Any important conclusions or recommendations.
    Keep the summary under 300 words.
""").strip()

SUMMARY_USER_TEMPLATE = textwrap.dedent("""
    Summarise the following document excerpts from "{source}":

    {context}

    Summary:
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Typical usage::

        pipeline = RAGPipeline()
        pipeline.ingest_file(file_bytes, "report.pdf")
        answer, sources, contexts = pipeline.ask("What is the main finding?")
    """

    def __init__(
        self,
        chunk_size: int    = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int         = DEFAULT_TOP_K,
        load_existing_index: bool = False,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k         = top_k

        # Load the embedding model (singleton – shared across calls)
        self.embedder  = get_embedder()

        # Initialise an empty FAISS index
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedder.embedding_dim
        )

        # Conversation history  [(role, text), …]
        self.history: List[Tuple[str, str]] = []

        # Track which files have been ingested
        self.ingested_files: List[str] = []

        # Try to load existing index from disk if requested
        if load_existing_index:
            self.load_index()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(self, file_obj: Any, file_name: str) -> int:
        """
        Process a single file and add its chunks to the vector index.

        Steps:
          1. Extract text pages via document_loader.
          2. Split each page into overlapping word chunks.
          3. Generate embeddings for all chunks.
          4. Add to FAISS index.

        Args:
            file_obj  : File-like object (BytesIO or Streamlit UploadedFile).
            file_name : Original filename (determines parser).

        Returns:
            Number of chunks indexed.
        """
        logger.info("Ingesting: %s", file_name)

        # Step 1 – extract pages
        pages = load_document(file_obj, file_name)
        if not pages:
            logger.warning("No text extracted from '%s'", file_name)
            return 0

        # Step 2 – chunk each page
        all_chunks: List[Dict] = []
        for page in pages:
            sub_chunks = chunk_text(
                page["text"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            for sub in sub_chunks:
                all_chunks.append({
                    "text"    : sub,
                    "source"  : page["source"],
                    "page_num": page["page_num"],
                })

        if not all_chunks:
            return 0

        # Step 3 – embed
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)

        # Step 4 – index
        self.vector_store.add_chunks(all_chunks, embeddings)

        if file_name not in self.ingested_files:
            self.ingested_files.append(file_name)

        logger.info("Ingested %d chunks from '%s'", len(all_chunks), file_name)

        # Save index to disk after ingestion
        self.save_index()

        return len(all_chunks)

    # ── Question Answering ────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Tuple[str, List[Dict], List[str]]:
        """
        Answer a question using retrieved document context (RAG).

        Args:
            question : User's question string.
            top_k    : Override default number of retrieved chunks.

        Returns:
            Tuple of:
              answer   (str)       – LLM-generated answer
              sources  (List[Dict])– list of source metadata dicts with scores
              contexts (List[str]) – raw retrieved chunk texts
        """
        if self.vector_store.total_chunks == 0:
            return (
                "No documents have been uploaded yet. "
                "Please upload at least one document first.",
                [],
                [],
            )

        k = top_k or self.top_k

        # Step 1 – embed the question
        query_vec = self.embedder.encode_query(question)

        # Step 2 – retrieve top-K chunks with MMR for diversity
        retrieved = self.vector_store.search(
            query_vec,
            top_k=k,
            use_mmr=True,
            mmr_lambda=MMR_LAMBDA,
            mmr_candidates=MMR_CANDIDATES
        )

        # Step 3 – build context string (trim to MAX_CONTEXT_CHARS)
        context_parts: List[str] = []
        total_chars = 0
        for chunk in retrieved:
            snippet = (
                f"[Source: {chunk['source']} | Page {chunk['page_num']}]\n"
                f"{chunk['text']}"
            )
            if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(snippet)
            total_chars += len(snippet)

        context = "\n\n---\n\n".join(context_parts)

        # Build conversation history string (last 6 turns)
        history_str = self._format_history(last_n=6)

        # Step 4 – call LLM
        user_prompt = QA_USER_TEMPLATE.format(
            context=context,
            history=history_str or "(no prior conversation)",
            question=question,
        )
        answer = _call_llm(QA_SYSTEM_PROMPT, user_prompt, max_tokens=1024)

        # Apply citation highlighting
        contexts_for_highlighting = [r["text"] for r in retrieved]
        answer = self._highlight_citations(answer, contexts_for_highlighting)

        # Save to history
        self.history.append(("user", question))
        self.history.append(("assistant", answer))

        # Summarize history if it gets too long
        self._summarize_history()

        # Format source attribution
        sources = [
            {
                "source"  : r["source"],
                "page_num": r["page_num"],
                "score"   : r["score"],
                "chunk_id": r["chunk_id"],
            }
            for r in retrieved
        ]
        contexts = [r["text"] for r in retrieved]

        return answer, sources, contexts

    def ask_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
    ):
        """
        Answer a question using retrieved document context (RAG) with streaming response.

        Args:
            question : User's question string.
            top_k    : Override default number of retrieved chunks.

        Yields:
            Tuple of:
              chunk    (str)       – Incremental chunk of LLM-generated answer
              sources  (List[Dict])– list of source metadata dicts with scores (first yield only)
              contexts (List[str]) – raw retrieved chunk texts (first yield only)
        """
        if self.vector_store.total_chunks == 0:
            yield (
                "No documents have been uploaded yet. "
                "Please upload at least one document first.",
                [],
                [],
            )
            return

        k = top_k or self.top_k

        # Step 1 – embed the question
        query_vec = self.embedder.encode_query(question)

        # Step 2 – retrieve top-K chunks with MMR for diversity
        retrieved = self.vector_store.search(
            query_vec,
            top_k=k,
            use_mmr=True,
            mmr_lambda=MMR_LAMBDA,
            mmr_candidates=MMR_CANDIDATES
        )

        # Step 3 – build context string (trim to MAX_CONTEXT_CHARS)
        context_parts: List[str] = []
        total_chars = 0
        for chunk in retrieved:
            snippet = (
                f"[Source: {chunk['source']} | Page {chunk['page_num']}]\n"
                f"{chunk['text']}"
            )
            if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(snippet)
            total_chars += len(snippet)

        context = "\n\n---\n\n".join(context_parts)

        # Build conversation history string (last 6 turns)
        history_str = self._format_history(last_n=6)

        # Step 4 – call LLM with streaming
        user_prompt = QA_USER_TEMPLATE.format(
            context=context,
            history=history_str or "(no prior conversation)",
            question=question,
        )

        # Yield sources and contexts first
        sources = [
            {
                "source"  : r["source"],
                "page_num": r["page_num"],
                "score"   : r["score"],
                "chunk_id": r["chunk_id"],
            }
            for r in retrieved
        ]
        contexts = [r["text"] for r in retrieved]
        yield "", sources, contexts  # Empty chunk first to provide metadata

        # Then yield streaming answer chunks
        full_answer = ""
        for chunk in _call_llm_stream(QA_SYSTEM_PROMPT, user_prompt, max_tokens=1024):
            full_answer += chunk
            yield chunk, [], []  # Empty sources/contexts for subsequent yields

        # Apply citation highlighting to complete answer
        full_answer = self._highlight_citations(full_answer, contexts)

        # Save to history after streaming completes
        self.history.append(("user", question))
        self.history.append(("assistant", full_answer))

        # Summarize history if it gets too long
        self._summarize_history()

    # ── Summarisation ─────────────────────────────────────────────────────────

    def summarise(self, source_name: str) -> str:
        """
        Generate a summary for a specific document by combining its chunks.

        Args:
            source_name : The exact filename as stored in the vector index.

        Returns:
            Summary string produced by the LLM.
        """
        chunks = self.vector_store.get_chunks_by_source(source_name)
        if not chunks:
            return f"No content found for document: {source_name}"

        # Use first N chars to stay within LLM context limits
        combined = "\n\n".join(c["text"] for c in chunks)
        combined = combined[:MAX_CONTEXT_CHARS]

        user_prompt = SUMMARY_USER_TEMPLATE.format(
            source=source_name,
            context=combined,
        )
        return _call_llm(SUMMARY_SYSTEM_PROMPT, user_prompt, max_tokens=512)

    # ── History helpers ───────────────────────────────────────────────────────

    def _format_history(self, last_n: int = 6) -> str:
        """Return the last `last_n` turns as a readable string."""
        recent = self.history[-(last_n * 2):]   # 2 entries per turn
        lines  = []
        for role, text in recent:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    def clear_history(self) -> None:
        """Reset conversation memory (documents remain indexed)."""
        self.history.clear()

    def reset(self) -> None:
        """Clear both the vector store AND conversation history."""
        self.vector_store.clear()
        self.history.clear()
        self.ingested_files.clear()
        # Delete saved index from disk
        import shutil
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
            logger.info("Deleted faiss_index directory")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, directory: str = "faiss_index") -> None:
        """
        Save the FAISS index and ingested files list to disk.

        Args:
            directory: Directory path to save the index files.
        """
        try:
            self.vector_store.save(directory)
            # Save ingested files list
            import json
            files_path = os.path.join(directory, "ingested_files.json")
            with open(files_path, "w") as f:
                json.dump(self.ingested_files, f)
            logger.info("Index saved to '%s'", directory)
        except Exception as exc:
            logger.error("Failed to save index: %s", exc)
            raise RuntimeError(f"Index save failed: {exc}") from exc

    def load_index(self, directory: str = "faiss_index") -> bool:
        """
        Load the FAISS index and ingested files list from disk.

        Args:
            directory: Directory path to load the index files from.

        Returns:
            True if index was loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(directory):
                logger.info("Index directory '%s' does not exist", directory)
                return False

            self.vector_store.load(directory)

            # Load ingested files list
            files_path = os.path.join(directory, "ingested_files.json")
            if os.path.exists(files_path):
                import json
                with open(files_path, "r") as f:
                    self.ingested_files = json.load(f)

            logger.info("Index loaded from '%s' (%d chunks)", directory, self.total_chunks)
            return True
        except Exception as exc:
            logger.error("Failed to load index: %s", exc)
            return False

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def indexed_sources(self) -> List[str]:
        """List of document names currently in the vector index."""
        return self.vector_store.get_all_sources()

        return self.vector_store.total_chunks

    # ── Citation highlighting ──────────────────────────────────────────────────

    def _highlight_citations(self, answer: str, contexts: List[str]) -> str:
        """
        Highlight parts of the answer that match text from retrieved contexts.

        Args:
            answer: The LLM-generated answer text.
            contexts: List of retrieved chunk texts.

        Returns:
            Answer with citations highlighted using markdown bold.
        """
        import re
        from difflib import SequenceMatcher

        highlighted_answer = answer

        # For each context chunk, find matching phrases in the answer
        for context in contexts:
            # Split context into sentences/phrases
            phrases = re.split(r'[.!?]+', context)
            phrases = [p.strip() for p in phrases if p.strip() and len(p.strip()) > 10]

            for phrase in phrases:
                # Find similar sequences in the answer
                answer_lower = answer.lower()
                phrase_lower = phrase.lower()

                # Use sequence matcher to find best matches
                matcher = SequenceMatcher(None, answer_lower, phrase_lower)
                match = matcher.find_longest_match(0, len(answer_lower), 0, len(phrase_lower))

                if match.size > 15:  # Only highlight if match is substantial
                    # Get the actual text from answer that matches
                    matched_text = answer[match.a:match.a + match.size]

                    # Skip if already highlighted
                    if f"**{matched_text}**" in highlighted_answer:
                        continue

                    # Highlight the matched text
                    highlighted_answer = highlighted_answer.replace(
                        matched_text,
                        f"**{matched_text}**"
                    )

        return highlighted_answer

    # ── History management ─────────────────────────────────────────────────────

    def _summarize_history(self) -> None:
        """
        Summarize old conversation history to manage context length.
        Keeps the most recent turns and replaces older ones with a summary.
        """
        if len(self.history) <= MAX_HISTORY_TURNS * 2:  # 2 entries per turn (user + assistant)
            return

        # Keep the last few turns as-is
        recent_turns = self.history[-(MAX_HISTORY_TURNS * 2):]

        # Summarize the older turns
        older_history = self.history[:-(MAX_HISTORY_TURNS * 2)]
        if older_history:
            # Format older history as readable text
            history_lines = []
            for role, text in older_history:
                prefix = "User" if role == "user" else "Assistant"
                history_lines.append(f"{prefix}: {text}")
            history_text = "\n".join(history_lines)

            # Generate summary using LLM
            try:
                summary = _call_llm(
                    system_prompt="You are a conversation summarizer.",
                    user_prompt=SUMMARY_HISTORY_PROMPT.format(history=history_text),
                    max_tokens=256
                )
                # Replace old history with summary
                self.history = [("assistant", f"Summary of earlier conversation: {summary.strip()}")] + recent_turns
            except Exception as exc:
                logger.warning("Failed to summarize history: %s", exc)
                # Keep all history if summarization fails
                self.history = self.history

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def indexed_sources(self) -> List[str]:
        """List of document names currently in the vector index."""
        return self.vector_store.get_all_sources()

    @property
    def total_chunks(self) -> int:
        """Total number of chunks across all indexed documents."""
        return self.vector_store.total_chunks
