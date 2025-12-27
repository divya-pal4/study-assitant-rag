import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# Paths
# -------------------------------
CHUNKS_FILE = "chunks.txt"
FAISS_DIR = "faiss_index"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_DIR, "chunks.pkl")

os.makedirs(FAISS_DIR, exist_ok=True)

# -------------------------------
# Load chunks
# -------------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded {len(chunks)} chunks")

# -------------------------------
# Load embedding model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Generate embeddings
# -------------------------------
embeddings = model.encode(
    chunks,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

# -------------------------------
# Create FAISS index
# -------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"✅ FAISS index created with {index.ntotal} vectors")

# -------------------------------
# Save index + chunks
# -------------------------------
faiss.write_index(index, INDEX_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("💾 FAISS index and chunks saved successfully")
