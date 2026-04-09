# 📚 DocAssist – AI-Powered Document Question Answering System

> **Retrieval-Augmented Generation (RAG) in a clean, local-first Streamlit app.**

---

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [How DocAssist Works](#how-docassist-works)
3. [Architecture Diagram](#architecture-diagram)
4. [Features](#features)
5. [Technology Stack](#technology-stack)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
8. [Running the App](#running-the-app)
9. [Configuration](#configuration)
10. [Example Usage](#example-usage)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a hybrid AI architecture that combines:

- 🔍 **Retrieval** – searching a knowledge base (your documents) for relevant information
- 🤖 **Generation** – using a Large Language Model (LLM) to generate a fluent, accurate answer *grounded in* the retrieved information

### Why RAG?
| Problem with plain LLMs | RAG Solution |
|--------------------------|--------------|
| Hallucinate facts | Answers grounded in real documents |
| Knowledge cutoff date | Knowledge updated by uploading new docs |
| No citations | Every answer cites its source document & page |
| Can't use your private data | Your documents stay local |

---

## How DocAssist Works

### Step-by-step flow

```
1. UPLOAD        User uploads PDF / DOCX / TXT files
        ↓
2. EXTRACT       Text extracted page-by-page from each file
        ↓
3. CHUNK         Text split into overlapping ~400-word chunks
                 (overlap preserves context at boundaries)
        ↓
4. EMBED         Each chunk encoded into a 384-dim vector
                 via SentenceTransformers (all-MiniLM-L6-v2)
        ↓
5. INDEX         Vectors stored in FAISS (flat inner-product index)
        ↓
──────────────────────────── At query time ────────────────────────────
        ↓
6. EMBED QUERY   User's question converted to a vector embedding
        ↓
7. RETRIEVE      FAISS returns the top-5 most semantically similar chunks
        ↓
8. PROMPT        Retrieved chunks injected into an LLM prompt as context
        ↓
9. GENERATE      Claude (claude-haiku-4-5) generates a grounded answer
        ↓
10. DISPLAY      Answer + source citations + raw context shown in UI
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI  (app.py)                   │
│                                                                   │
│   ┌──────────────┐    ┌─────────────────────────────────────┐   │
│   │   Sidebar    │    │           Main Chat Area             │   │
│   │              │    │                                     │   │
│   │ 📤 Upload    │    │  💬 Chat messages                   │   │
│   │ 📚 Doc list  │    │  📎 Source citations                │   │
│   │ 📝 Summarise │    │  🔍 Context snippets                │   │
│   │ ⚙️  Settings  │    │                                     │   │
│   └──────────────┘    └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE  (rag_pipeline.py)              │
│                                                                   │
│   ┌──────────────────┐         ┌──────────────────────────────┐ │
│   │  INGESTION PATH  │         │       QUERY PATH             │ │
│   │                  │         │                              │ │
│   │ document_loader  │         │  1. embed_query()            │ │
│   │  ├─ load_pdf()   │         │  2. vector_store.search()    │ │
│   │  ├─ load_docx()  │         │  3. build prompt + history   │ │
│   │  └─ load_txt()   │         │  4. _call_llm() → answer     │ │
│   │       ↓          │         │                              │ │
│   │   chunk_text()   │         └──────────────────────────────┘ │
│   │       ↓          │                                           │
│   │   embedder       │                                           │
│   │  .encode()       │                                           │
│   │       ↓          │                                           │
│   │  vector_store    │                                           │
│   │  .add_chunks()   │                                           │
│   └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
┌────────────────────┐                  ┌──────────────────────────┐
│  EMBEDDINGS        │                  │  VECTOR STORE            │
│  (embeddings.py)   │                  │  (vector_store.py)       │
│                    │                  │                          │
│ SentenceTransformer│                  │  FAISS IndexFlatIP       │
│ all-MiniLM-L6-v2   │                  │  (cosine similarity)     │
│ dim = 384          │                  │  + metadata list         │
└────────────────────┘                  └──────────────────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │  OLLAMA LOCAL         │
                                         │  llama3               │
                                         │  (LLM generation)     │
                                         └──────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| 📄 Multi-format upload | PDF, DOCX, TXT, Markdown |
| ⚡ Fast semantic search | FAISS flat index with MMR diversity, sub-second retrieval |
| 💬 Conversational memory | Last 6 turns sent as context, auto-summarized for long conversations |
| 📎 Source attribution | Document name + page number per answer |
| 🔍 Context transparency | Raw retrieved chunks displayed on demand |
| 📝 One-click summarisation | Full-document summary per file |
| 🗑️ Session reset | Clear docs + history with one click |
| 🔑 Local-first | No API keys needed, everything runs locally |
| ⚡ Streaming responses | ChatGPT-like token-by-token answer generation |
| 🎯 Citation highlighting | Key phrases from sources **bolded** in answers |
| 💾 Persistent storage | Document index survives app restarts |
| 🚀 Embedding caching | Repeated queries use cached embeddings for speed |
| 🎪 MMR retrieval | Diverse, relevant chunks instead of just similar ones |

---

## Technology Stack

| Layer | Tool |
|-------|------|
| UI | [Streamlit](https://streamlit.io) with streaming placeholders |
| Embeddings | [SentenceTransformers](https://www.sbert.net) – `all-MiniLM-L6-v2` with LRU caching |
| Vector DB | [FAISS](https://github.com/facebookresearch/faiss) – CPU build with MMR & persistence |
| LLM | [Ollama](https://ollama.ai) – `llama3` with streaming API |
| Document Processing | pypdf, python-docx, text parsing |
| Memory Management | Automatic conversation summarization |
| Citation Analysis | Sequence matching for answer highlighting |
| PDF parsing | [pypdf](https://pypi.org/project/pypdf/) |
| DOCX parsing | [python-docx](https://python-docx.readthedocs.io) |

---

## Project Structure

```
docassist/
│
├── app.py               # Streamlit UI – main entry point
├── rag_pipeline.py      # RAG orchestration (ingest, ask, summarise)
├── document_loader.py   # PDF / DOCX / TXT text extraction
├── embeddings.py        # SentenceTransformer embedding wrapper
├── vector_store.py      # FAISS index + chunk text splitter
│
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
└── sample_documents/    # Example documents for testing
    ├── MachineLearningNotes.txt
    └── ClimateReport2024.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- **Ollama** installed → [ollama.ai](https://ollama.ai)

### 1 – Install Ollama

Follow the instructions at [ollama.ai](https://ollama.ai) to install Ollama for your platform.

### 2 – Start Ollama server

In a separate terminal, start the Ollama server:

```bash
ollama serve
```

The server will run on `http://localhost:11434`.

### 3 – Pull the Llama3 model

```bash
ollama pull llama3
```

### 4 – Clone / download the project

```bash
git clone https://github.com/your-username/docassist.git
cd docassist
```

### 4 – Create a virtual environment (recommended)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 5 – Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first install downloads PyTorch + SentenceTransformers (~1.5 GB).
> Subsequent runs use the local cache.

### 6 – Run the app

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## Configuration

All tunable constants are at the top of each module:

| File | Constant | Default | Description |
|------|----------|---------|-------------|
| `rag_pipeline.py` | `DEFAULT_CHUNK_SIZE` | 400 | Words per chunk |
| `rag_pipeline.py` | `DEFAULT_CHUNK_OVERLAP` | 60 | Overlap words |
| `rag_pipeline.py` | `DEFAULT_TOP_K` | 5 | Chunks retrieved |
| `rag_pipeline.py` | `LLM_MODEL` | `llama3` | Local Ollama model |
| `rag_pipeline.py` | `MMR_CANDIDATES` | 20 | Candidates for MMR selection |
| `rag_pipeline.py` | `MMR_LAMBDA` | 0.7 | MMR relevance vs diversity balance |
| `rag_pipeline.py` | `MAX_HISTORY_TURNS` | 10 | Conversation turns before summarization |
| `embeddings.py` | `DEFAULT_MODEL_NAME` | `all-MiniLM-L6-v2` | HF embedding model |
| `embeddings.py` | `EMBEDDING_CACHE_SIZE` | 1000 | Max cached embeddings |

---

## Example Usage

1. Launch the app: `streamlit run app.py`
2. Upload `sample_documents/MachineLearningNotes.txt`
3. Click **Process 1 file(s)**
4. Ask: *"What is gradient descent?"*
5. DocAssist responds with a grounded answer and cites the source page.
6. Ask a follow-up: *"What are the variants of gradient descent?"*
7. Click **📎 Sources** to see which page the answer came from.
8. Click **Generate Summary** in the sidebar for a full document overview.

### Example output
```
Answer:
Gradient descent is an iterative optimisation algorithm used to minimise
a loss function. It moves in the direction opposite to the gradient to find
the minimum. Common variants include Batch GD, SGD, and Mini-Batch GD.

Source:
📄 MachineLearningNotes.txt | Page 1 | Relevance: 94%
```

---

## Advanced Features

### Streaming Responses
DocAssist now provides ChatGPT-like streaming responses, showing answers token-by-token as they're generated. This creates a more engaging user experience and allows you to see responses in real-time.

### MMR (Maximal Marginal Relevance)
Instead of returning only the most similar chunks, DocAssist uses MMR to balance **relevance** and **diversity**. This ensures retrieved chunks are both relevant to your question and cover different aspects of the topic, leading to more comprehensive answers.

### Citation Highlighting
Key phrases from source documents are automatically **bolded** in answers, making it easy to see which parts of the response come directly from your documents.

### Persistent Storage
Document indexes are automatically saved to disk and reloaded on app restart. No need to re-process documents every time you use DocAssist.

### Embedding Caching
Frequently used text embeddings are cached in memory, speeding up repeated queries and document processing.

### Smart Memory Management
Long conversations are automatically summarized to prevent context overflow, while maintaining the most relevant information for future questions.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama not running` | Start Ollama and ensure llama3 is pulled |
| `faiss-cpu` install fails | Try `pip install faiss-cpu --no-cache-dir` |
| PDF text is empty | Some scanned PDFs need OCR; try `pytesseract` |
| Slow first run | Model download (~90 MB) happens once; subsequent runs are fast |
| Port already in use | `streamlit run app.py --server.port 8502` |

---

*Built with ❤️ using Ollama, FAISS, SentenceTransformers, and Streamlit.*
