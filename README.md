# 🧠 WPIBot Architecture

## Overview
WPIBot is a campus assistant chatbot built using Retrieval-Augmented Generation (RAG) to answer queries related to Worcester Polytechnic Institute (WPI). It is deployed via Streamlit and utilizes both local and cloud-based LLMs.

---

## 🏗️ Architecture Components

### 1. 🗃️ Data Ingestion
- WPI-related webpage (PDFs, HTML, etc.)
- Processed and chunked into smaller passages
- Embedded using `sentence-transformers` (`all-MiniLM-L6-v2`)

### 2. 🧠 Vector Store
- **FAISS** used to store and index document embeddings
- Supports fast retrieval using vector similarity search

### 3. 📚 RAG Pipeline
- **Query Input** → Embed the query
- **Retriever** → Fetch top-k relevant chunks from FAISS
- **LLM Generator** → Generate a response using:
  - Groq-hosted models (e.g., `LLaMA-3-70B`, `Mixtral-8x7B`)
  - Local models (via Ollama; e.g., `Mistral-7B`, `Gemma`, `TinyLLaMA`)

### 4. 🖥️ Frontend
- Built with **Streamlit**
- UI allows:
  - Asking questions
  - Viewing retrieved chunks
  - Comparing responses across models

### 5. ☁️ Deployment
- **Primary**: Streamlit Cloud
- **Optional Scale-up**: AWS (for Groq API keys, local model hosting)

---

## 🔧 Optional Evaluation
- **BERTScore** for semantic similarity of answers
- **Latency comparison** between Groq and local models

---

## 🔄 Workflow

```mermaid
graph TD
    A[User Query] --> B[Embed Query]
    B --> C[Retrieve Top-k Chunks from FAISS]
    C --> D[Pass Context + Query to LLM]
    D --> E[Generate Response]
    E --> F[Streamlit UI]
