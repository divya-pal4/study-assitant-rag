from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import subprocess
import ollama
import os
import time
import faiss
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Study Assistant RAG API")


MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = "faiss_index/index.faiss"
DEFAULT_CHUNKS_PATH = "faiss_index/chunks.pkl"

model = SentenceTransformer(MODEL_NAME)

def load_index_and_chunks(pdf_id: Optional[str]):
    if pdf_id:
        idx_path = f"faiss_index/{pdf_id}/index.faiss"
        chunks_path = f"faiss_index/{pdf_id}/chunks.pkl"
    else:
        idx_path = DEFAULT_INDEX_PATH
        chunks_path = DEFAULT_CHUNKS_PATH

    if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Index or chunks not found for pdf_id={pdf_id}")

    idx = faiss.read_index(idx_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return idx, chunks


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    pdf_id: Optional[str] = None

def build_prompt(question: str, top_k: int, retrieved_chunks: list[str] = None):
    context = ""

    if retrieved_chunks:
        context = "\n\n".join(retrieved_chunks)
        context = context[:800]   # keep context small for speed

    prompt = f"""
You are a helpful study assistant.
Answer the question using the context below.
If the answer is not in the context, then answer accordingly to you knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()


@app.post("/ask_llm")
def ask_llm(req: QueryRequest):
    # load the appropriate index + chunks (per-pdf if requested)
    try:
        idx, chunks = load_index_and_chunks(req.pdf_id)
    except FileNotFoundError:
        return {"answer": "Index for requested PDF is not ready yet. Please try later.", "sources": []}

    # embed the question and retrieve top_k chunks
    q_emb = model.encode([req.question], convert_to_numpy=True).astype('float32')
    D, I = idx.search(q_emb, req.top_k)
    retrieved = []
    for i in I[0]:
        if i < len(chunks):
            retrieved.append(chunks[i])

    prompt = build_prompt(req.question, req.top_k, retrieved)

    response = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt,
        stream=False
    )

    answer = response["response"]
    answer = " ".join(answer.split())

    return {
        "answer": answer,
        "sources": retrieved
    }


@app.get("/")
def health():
    return {"status": "RAG API running"}
