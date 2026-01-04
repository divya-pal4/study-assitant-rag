import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import sys
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parse CLI args
parser = argparse.ArgumentParser(description="Generate FAISS index from chunks file")
parser.add_argument("--chunks-file", dest="chunks_file", required=True,
                    help="Path to chunks file (JSON-lines format)")
parser.add_argument("--faiss-dir", dest="faiss_dir", required=True,
                    help="Directory to save FAISS index")
args = parser.parse_args()

CHUNKS_FILE = args.chunks_file
FAISS_DIR = args.faiss_dir
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_DIR, "chunks.pkl")

# Ensure output directory exists
os.makedirs(FAISS_DIR, exist_ok=True)

logger.info(f"Starting embedding generation for: {CHUNKS_FILE}")
logger.info(f"Output directory: {FAISS_DIR}")

# Load chunks
if not os.path.exists(CHUNKS_FILE):
    logger.error(f"Chunks file not found: {CHUNKS_FILE}")
    sys.exit(1)

try:
    with open(CHUNKS_FILE, "r", encoding='utf-8') as f:
        raw = f.read()
except Exception as e:
    logger.error(f"Failed to read chunks file: {e}")
    sys.exit(1)

# Parse chunks (support JSON-lines format)
chunks = []
lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

if not lines:
    logger.error(f"Chunks file is empty: {CHUNKS_FILE}")
    sys.exit(1)

logger.info(f"Parsing {len(lines)} lines from chunks file...")

# Try JSON-lines format first
parsed_count = 0
for i, ln in enumerate(lines):
    try:
        obj = json.loads(ln)
        if isinstance(obj, dict) and "text" in obj:
            text = obj["text"].strip()
            if text:
                chunks.append(text)
                parsed_count += 1
        else:
            logger.warning(f"Line {i+1}: No 'text' field in JSON object")
    except json.JSONDecodeError as e:
        logger.warning(f"Line {i+1}: Invalid JSON - {e}")
        # Fallback: treat line as plain text
        if ln:
            chunks.append(ln)
            parsed_count += 1

if not chunks:
    logger.error("No valid chunks found after parsing")
    sys.exit(1)

logger.info(f"✅ Successfully parsed {parsed_count} chunks")

# Load embedding model
logger.info("Loading embedding model: all-MiniLM-L6-v2")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("✅ Embedding model loaded")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    sys.exit(1)

# Generate embeddings
logger.info(f"Generating embeddings for {len(chunks)} chunks...")
try:
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32  # Process in batches for efficiency
    ).astype("float32")
    
    logger.info(f"✅ Generated embeddings with shape: {embeddings.shape}")
except Exception as e:
    logger.error(f"Failed to generate embeddings: {e}")
    sys.exit(1)

# Normalize embeddings for better similarity search (optional but recommended)
logger.info("Normalizing embeddings...")
faiss.normalize_L2(embeddings)

# Create FAISS index
logger.info("Creating FAISS index...")
try:
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for cosine similarity (after normalization)
    # This is faster and more accurate than L2 for normalized vectors
    index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine for normalized vectors
    
    index.add(embeddings)
    logger.info(f"✅ FAISS index created with {index.ntotal} vectors")
except Exception as e:
    logger.error(f"Failed to create FAISS index: {e}")
    sys.exit(1)

# Save index and chunks
logger.info(f"Saving FAISS index to: {INDEX_PATH}")
try:
    faiss.write_index(index, INDEX_PATH)
    logger.info("✅ FAISS index saved")
except Exception as e:
    logger.error(f"Failed to save FAISS index: {e}")
    sys.exit(1)

logger.info(f"Saving chunks to: {CHUNKS_PATH}")
try:
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info("✅ Chunks saved")
except Exception as e:
    logger.error(f"Failed to save chunks: {e}")
    sys.exit(1)

logger.info("=" * 50)
logger.info("✅ Embedding generation completed successfully!")
logger.info(f"   - Chunks: {len(chunks)}")
logger.info(f"   - Embeddings: {embeddings.shape}")
logger.info(f"   - Index vectors: {index.ntotal}")
logger.info(f"   - Output: {FAISS_DIR}")
logger.info("=" * 50)

sys.exit(0)