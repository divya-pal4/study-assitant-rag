from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import faiss
import pickle
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


@app.post("/ask_llm")
def ask_llm(request: QueryRequest):
    query_embedding = model.encode(
        [request.question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, request.top_k)
    context = "\n\n".join(
        [chunks[i][:800] for i in indices[0]]
    )

    
    prompt = f"""
You are a helpful study assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{request.question}

Answer:
"""

    
    result = subprocess.run(
        ["ollama", "run", "llama3.2:3b"],
        input=prompt,
        text=True,
        capture_output=True
    )

    return {
        "question": request.question,
        "answer": result.stdout.strip(),
        "sources": [chunks[i] for i in indices[0]]
    }


@app.get("/")
def health():
    return {"status": "RAG API running"}
