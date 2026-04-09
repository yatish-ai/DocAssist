"""
app.py
------
DocAssist – AI-Powered Document Q&A System
Streamlit web interface.

Run:
    streamlit run app.py

The system uses Ollama with Llama3 locally for question answering.
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import List

import streamlit as st

from rag_pipeline import RAGPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocAssist – AI Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global fonts & background ── */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .header-banner h1 { margin: 0; font-size: 2.2rem; }
    .header-banner p  { margin: .4rem 0 0; opacity: .85; font-size: 1.05rem; }

    /* ── Chat bubbles ── */
    .chat-user {
        background: #e8f4fd;
        border-left: 4px solid #1a73e8;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin: 8px 0;
    }
    .chat-assistant {
        background: #f0f9ff;
        border-left: 4px solid #34a853;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin: 8px 0;
    }

    /* ── Source card ── */
    .source-card {
        background: #fffbea;
        border: 1px solid #f6c90e;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: .88rem;
    }

    /* ── Context snippet ── */
    .context-snippet {
        background: #f8f9fa;
        border-left: 3px solid #aaa;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        font-size: .85rem;
        color: #444;
        margin: 4px 0;
    }

    /* ── Stat box ── */
    .stat-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 12px 18px;
        text-align: center;
    }
    .stat-box .num { font-size: 1.8rem; font-weight: 700; color: #1a73e8; }
    .stat-box .lbl { font-size: .8rem; color: #666; }

    /* ── Upload zone feedback ── */
    .success-badge {
        background: #d4edda;
        color: #155724;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_chat_bubble(role: str, content: str) -> None:
    css_class = "chat-user" if role == "user" else "chat-assistant"
    icon      = "🧑" if role == "user" else "🤖"
    st.markdown(
        f'<div class="{css_class}"><b>{icon} {"You" if role=="user" else "DocAssist"}</b><br>{content}</div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: List[dict]) -> None:
    if not sources:
        return
    with st.expander("📎 Sources", expanded=False):
        seen = set()
        for s in sources:
            key = (s["source"], s["page_num"])
            if key in seen:
                continue
            seen.add(key)
            score_pct = int(s["score"] * 100)
            st.markdown(
                f'<div class="source-card">📄 <b>{s["source"]}</b> &nbsp;|&nbsp; '
                f'Page {s["page_num"]} &nbsp;|&nbsp; '
                f'Relevance: <b>{score_pct}%</b></div>',
                unsafe_allow_html=True,
            )


def render_contexts(contexts: List[str]) -> None:
    if not contexts:
        return
    with st.expander("🔍 Retrieved Context Chunks", expanded=False):
        for i, ctx in enumerate(contexts, 1):
            st.markdown(
                f'<div class="context-snippet"><b>Chunk {i}:</b><br>{ctx[:500]}{"…" if len(ctx)>500 else ""}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Ollama Status Check ─────────────────────────────────────────────────
    from rag_pipeline import check_ollama_running
    if not check_ollama_running():
        st.error(
            "🚨 **Ollama Not Running**\n\n"
            "DocAssist requires Ollama to be running locally.\n\n"
            "**To fix:**\n"
            "1. Install Ollama: https://ollama.ai\n"
            "2. Start server: `ollama serve`\n"
            "3. Pull model: `ollama pull llama3`\n\n"
            "Then refresh this page."
        )
    else:
        st.success("✅ Ollama is running")

    st.divider()

    # ── Fresh Start Toggle ──────────────────────────────────────────────────
    fresh_start = st.checkbox(
        "🗑️ Fresh Start (delete old index)",
        value=True,
        help="Start with a clean index. Uncheck to load existing index from disk.",
    )

    # Handle fresh start toggle change
    if "fresh_start_prev" not in st.session_state or st.session_state.fresh_start_prev != fresh_start:
        if fresh_start:
            # Delete old index if exists
            import shutil
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
                st.info("🗑️ Deleted old FAISS index for fresh start.")
        # Reinitialize pipeline
        st.session_state.pipeline = RAGPipeline(load_existing_index=not fresh_start)
        st.session_state.fresh_start_prev = fresh_start
        # Clear chat and processed files
        st.session_state.chat_messages = []
        st.session_state.processed_files = set()

    st.divider()

    # ── Document Upload ──────────────────────────────────────────────────────
    st.markdown("## 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload one or more documents to query.",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.processed_files
        ]
        if new_files:
            if st.button(f"⚡ Process {len(new_files)} file(s)", use_container_width=True, type="primary"):
                progress = st.progress(0, text="Processing documents…")
                for idx, file in enumerate(new_files):
                    with st.spinner(f"Indexing {file.name}…"):
                        file_bytes = io.BytesIO(file.read())
                        n_chunks = st.session_state.pipeline.ingest_file(
                            file_bytes, file.name
                        )
                    st.session_state.processed_files.add(file.name)
                    progress.progress(
                        (idx + 1) / len(new_files),
                        text=f"Processed: {file.name} ({n_chunks} chunks)",
                    )
                    time.sleep(0.1)
                progress.empty()
                st.success(f"✅ {len(new_files)} document(s) indexed!")
                st.rerun()
        else:
            st.markdown('<div class="success-badge">✅ All files already processed.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Indexed documents list ────────────────────────────────────────────────
    sources = st.session_state.pipeline.indexed_sources
    st.markdown(f"## 📚 Indexed Documents ({len(sources)})")
    if sources:
        for src in sources:
            st.markdown(f"• `{src}`")

        # ── Per-doc summarisation ─────────────────────────────────────────────
        st.divider()
        st.markdown("## 📝 Summarise a Document")
        selected_doc = st.selectbox("Choose document", sources, label_visibility="collapsed")
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Summarising…"):
                summary = st.session_state.pipeline.summarise(selected_doc)
            st.info(summary)
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-box"><div class="num">{len(sources)}</div>'
            f'<div class="lbl">Documents</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-box"><div class="num">{st.session_state.pipeline.total_chunks}</div>'
            f'<div class="lbl">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Reset ─────────────────────────────────────────────────────────────────
    if st.button("🗑️ Reset Everything", use_container_width=True):
        st.session_state.pipeline.reset()
        st.session_state.chat_messages.clear()
        st.session_state.processed_files.clear()
        st.rerun()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.pipeline.clear_history()
        st.session_state.chat_messages.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN AREA ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ── Header banner ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-banner">
        <h1>📚 DocAssist</h1>
        <p>AI-Powered Document Q&amp;A — upload documents, ask questions, get cited answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Empty-state prompt ────────────────────────────────────────────────────────
if not st.session_state.pipeline.indexed_sources:
    st.info(
        "👈 **Get started:** upload one or more documents in the sidebar, "
        "click **Process**, then ask questions below."
    )
    st.markdown(
        """
        ### How DocAssist works
        1. **Upload** – PDF, DOCX, or TXT files
        2. **Index** – Text is chunked and embedded into a FAISS vector store
        3. **Ask** – Your question is matched against relevant chunks (RAG)
        4. **Answer** – An LLM generates a grounded answer with source citations
        """
    )

# ── Render existing chat messages ─────────────────────────────────────────────
for msg in st.session_state.chat_messages:
    render_chat_bubble(msg["role"], msg["content"])
    if msg["role"] == "assistant":
        render_sources(msg.get("sources", []))
        render_contexts(msg.get("contexts", []))

# ── Chat input ────────────────────────────────────────────────────────────────
st.divider()

question = st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.pipeline.indexed_sources,
)

if question:
    # Show user bubble immediately
    st.session_state.chat_messages.append(
        {"role": "user", "content": question, "sources": [], "contexts": []}
    )
    render_chat_bubble("user", question)

    # Create a placeholder for the streaming response
    response_placeholder = st.empty()
    sources_placeholder = st.empty()
    contexts_placeholder = st.empty()

    # Generate answer with streaming
    try:
        full_answer = ""
        sources = []
        contexts = []

        for chunk, chunk_sources, chunk_contexts in st.session_state.pipeline.ask_stream(question):
            if chunk_sources and chunk_contexts:  # First yield with metadata
                sources = chunk_sources
                contexts = chunk_contexts
            else:  # Subsequent yields with answer chunks
                full_answer += chunk
                # Update the response in real-time
                response_placeholder.markdown(f"**Assistant:** {full_answer}▊")

        # Final update without cursor
        response_placeholder.markdown(f"**Assistant:** {full_answer}")

    except RuntimeError as exc:
        full_answer = f"❌ Error: {exc}"
        sources = []
        contexts = []
        response_placeholder.markdown(f"**Assistant:** {full_answer}")

    # Clear placeholders and render final response properly
    response_placeholder.empty()
    sources_placeholder.empty()
    contexts_placeholder.empty()

    # Store & render assistant response
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": full_answer, "sources": sources, "contexts": contexts}
    )
    render_chat_bubble("assistant", full_answer)
    render_sources(sources)
    render_contexts(contexts)
    st.rerun()
