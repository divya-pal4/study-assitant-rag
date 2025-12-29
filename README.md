# Study Assistant (RAG-based)

A local Retrieval-Augmented Generation (RAG) study assistant that allows users to upload PDFs and ask questions grounded in their content.

## Features
- PDF upload & chunking
- Local embeddings using Sentence Transformers
- FAISS vector search
- FastAPI retrieval service
- Local LLM inference using Ollama (llama3.3:3b)
- Node.js backend

## Tech Stack
- Node.js + Express
- Python + FastAPI
- FAISS
- Sentence Transformers
- Ollama (llama3.2:3b)

## Architecture
PDF → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer

## Chat UI
- Implemented a basic UI
- Having an option to upload pdf and to chat where you can ask question related to the pdf.

## Upcoming
- Improvement in UI
- Multi-PDF support
- Source citations
- Deployment

---

Built with ❤️ while learning RAG from scratch.
