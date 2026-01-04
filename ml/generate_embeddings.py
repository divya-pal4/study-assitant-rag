import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import sys
import argparse

# parse CLI args so backend can request per-pdf indexing
parser = argparse.ArgumentParser(description="Generate FAISS index from chunks file")
parser.add_argument("--chunks-file", dest="chunks_file", default="chunks.txt")
parser.add_argument("--faiss-dir", dest="faiss_dir", default="faiss_index")
args = parser.parse_args()

CHUNKS_FILE = args.chunks_file
FAISS_DIR = args.faiss_dir
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_DIR, "chunks.pkl")

os.makedirs(FAISS_DIR, exist_ok=True)

# loading chunks
if not os.path.exists(CHUNKS_FILE):
    print(f"Error: {CHUNKS_FILE} not found. Please upload a PDF via the frontend or create the file.")
    sys.exit(1)

with open(CHUNKS_FILE, "r") as f:
    raw = f.read()

# support two formats:
# 1) backend writes one JSON object per line: {"pdf":..., "chunk_id":..., "text":...}\n
# 2) legacy `--- CHUNK ` separator used by earlier tools
chunks = []

lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
if not lines:
    print(f"Error: {CHUNKS_FILE} is empty")
    sys.exit(1)

# try JSON-lines first
try:
    for ln in lines:
        obj = json.loads(ln)
        # prefer `text` key, fallback to entire line
        if isinstance(obj, dict) and "text" in obj:
            chunks.append(obj["text"].strip())
        else:
            chunks.append(str(obj).strip())
except Exception:
    # fallback to separator-based splitting
    if "--- CHUNK " in raw:
        chunks = [c.strip() for c in raw.split("--- CHUNK ") if c.strip()]
    else:
        # fallback: treat each non-empty line as a chunk
        chunks = lines

print(f"✅ Loaded {len(chunks)} chunks")

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#generating embeddings
embeddings = model.encode(
    chunks,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

#creating FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index created with {index.ntotal} vectors")


faiss.write_index(index, INDEX_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index and chunks saved successfully")
