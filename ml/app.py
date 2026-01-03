from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import subprocess
import ollama
import time
import faiss
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Study Assistant RAG API")


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f" Loaded FAISS with {index.ntotal} chunks")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

def build_prompt(question: str, top_k: int, retrieved_chunks: list[str] = None):
    context = ""

    if retrieved_chunks:
        context = "\n\n".join(retrieved_chunks)
        context = context[:800]   # keep context small for speed

    prompt = f"""
You are a helpful study assistant.
Answer the question using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()


@app.post("/ask_llm")
def ask_llm(req: QueryRequest):

    prompt = build_prompt(req.question, req.top_k)

    response = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt,
        stream=False
    )

    answer = response["response"]
    answer = " ".join(answer.split())
    
    return {
        "answer": answer,
        "sources": []
    }


@app.get("/")
def health():
    return {"status": "RAG API running"}
