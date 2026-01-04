const axios = require("axios");
const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const { spawn } = require('child_process');
const crypto = require('crypto');

// Generate UUID without external package
function generateUUID() {
  return crypto.randomUUID();
}

const app = express();

app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });

// Store PDF metadata in-memory (in production, use a database)
const pdfMetadata = new Map();

// IMPROVED: Add chunk overlap to avoid losing context at boundaries
function chunkText(text, wordsPerChunk = 500, overlap = 50) {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const chunks = [];
  
  for (let i = 0; i < words.length; i += (wordsPerChunk - overlap)) {
    const chunk = words.slice(i, i + wordsPerChunk).join(" ");
    if (chunk.trim().length > 0) {
      chunks.push(chunk);
    }
  }
  
  return chunks;
}

app.get("/", (req, res) => {
  res.send("Study Assistant Backend is running");
});

// IMPROVED: Better error handling and metadata storage
app.post("/upload", upload.single("pdf"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No PDF file provided" });
    }

    const dataBuffer = fs.readFileSync(req.file.path);
    const pdfData = await pdfParse(dataBuffer);

    const text = pdfData.text;
    
    if (!text || text.trim().length === 0) {
      return res.status(400).json({ error: "PDF contains no extractable text" });
    }

    // IMPROVED: Use overlap for better context retention
    const chunks = chunkText(text, 500, 50);
    const pdfId = generateUUID();

    // Store metadata
    pdfMetadata.set(pdfId, {
      filename: req.file.originalname,
      uploadTime: new Date().toISOString(),
      totalChunks: chunks.length,
      indexStatus: 'processing'
    });

    // Create organized directory structure
    const mlDir = path.join(__dirname, "..", "ml");
    const chunksDir = path.join(mlDir, "chunks");
    const faissOutDir = path.join(mlDir, "faiss_index", pdfId);
    
    fs.mkdirSync(chunksDir, { recursive: true });
    fs.mkdirSync(faissOutDir, { recursive: true });
    
    const chunksFilePath = path.join(chunksDir, `${pdfId}.txt`);

    // Write chunks with metadata
    chunks.forEach((chunk, index) => {
      const chunkObj = {
        pdf: req.file.originalname,
        pdf_id: pdfId,
        chunk_id: index + 1,
        text: chunk
      };
      fs.appendFileSync(chunksFilePath, JSON.stringify(chunkObj) + "\n");
    });

    // IMPROVED: Better process monitoring
    // Use Python from virtual environment
    const pythonPath = path.join(__dirname, '..', 'ml', 'venv', 'bin', 'python3');
    const scriptPath = path.join(__dirname, '..', 'ml', 'generate_embeddings.py');
    
    console.log(`Starting embedding generation for PDF: ${req.file.originalname}`);
    
    const py = spawn(pythonPath, [
      scriptPath,
      '--chunks-file', chunksFilePath,
      '--faiss-dir', faissOutDir
    ], { 
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false  // Keep process attached for better error handling
    });

    // Log output for debugging
    py.stdout.on('data', (data) => {
      console.log(`[Embeddings] ${data.toString().trim()}`);
    });

    py.stderr.on('data', (data) => {
      console.error(`[Embeddings Error] ${data.toString().trim()}`);
    });

    py.on('close', (code) => {
      if (code === 0) {
        console.log(`✅ Embeddings generated successfully for ${req.file.originalname}`);
        if (pdfMetadata.has(pdfId)) {
          pdfMetadata.get(pdfId).indexStatus = 'ready';
        }
      } else {
        console.error(`❌ Embedding generation failed with code ${code}`);
        if (pdfMetadata.has(pdfId)) {
          pdfMetadata.get(pdfId).indexStatus = 'failed';
        }
      }
    });

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({
      message: "PDF uploaded and processing started",
      pdf_id: pdfId,
      filename: req.file.originalname,
      totalChunks: chunks.length,
      preview: chunks[0].substring(0, 300)
    });

  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ 
      error: "Failed to process PDF",
      details: error.message 
    });
  }
});

// IMPROVED: Better timeout handling and error messages
app.post("/ask", async (req, res) => {
  try {
    const { question, pdf_id, top_k } = req.body;

    if (!question || question.trim().length === 0) {
      return res.status(400).json({ error: "Question cannot be empty" });
    }

    if (pdf_id && !pdfMetadata.has(pdf_id)) {
      return res.status(404).json({ error: "Selected PDF not found" });
    }

    const payload = { 
      question: question.trim(), 
      top_k: top_k || 3 
    };
    
    if (pdf_id) {
      payload.pdf_id = pdf_id;
    }

    const response = await axios.post(
      "http://localhost:8000/ask_llm",
      payload,
      { 
        timeout: 120000  // 2 minute timeout (better than 0)
      }
    );

    res.json({
      question: question,
      answer: response.data.answer,
      sources: response.data.sources,
      pdf_name: pdf_id && pdfMetadata.has(pdf_id) 
        ? pdfMetadata.get(pdf_id).filename 
        : null
    });

  } catch (error) {
    console.error("ASK ERROR:", error.message);
    
    if (error.code === 'ECONNABORTED') {
      return res.status(504).json({ 
        error: "Request timeout - LLM took too long to respond" 
      });
    }
    
    if (error.response) {
      return res.status(error.response.status).json({ 
        error: error.response.data.detail || "Failed to get answer" 
      });
    }
    
    res.status(500).json({ 
      error: "Failed to get answer",
      details: error.message 
    });
  }
});

// IMPROVED: Return more detailed status
app.get('/index_status/:pdfId', (req, res) => {
  try {
    const pdfId = req.params.pdfId;
    const idxPath = path.join(__dirname, '..', 'ml', 'faiss_index', pdfId, 'index.faiss');
    const ready = fs.existsSync(idxPath);
    
    const metadata = pdfMetadata.get(pdfId);
    
    res.json({ 
      ready,
      status: ready ? 'ready' : (metadata?.indexStatus || 'processing'),
      filename: metadata?.filename,
      totalChunks: metadata?.totalChunks
    });
  } catch (err) {
    console.error('index_status error', err);
    res.status(500).json({ 
      ready: false, 
      status: 'error',
      error: err.message 
    });
  }
});

// NEW: Get all uploaded PDFs
app.get('/pdfs', (req, res) => {
  try {
    const pdfs = Array.from(pdfMetadata.entries()).map(([id, meta]) => ({
      pdf_id: id,
      ...meta
    }));
    res.json({ pdfs });
  } catch (err) {
    console.error('Error fetching PDFs:', err);
    res.status(500).json({ error: 'Failed to fetch PDFs' });
  }
});

// NEW: Delete a PDF and its index
app.delete('/pdf/:pdfId', (req, res) => {
  try {
    const pdfId = req.params.pdfId;
    
    // Delete chunks file from organized directory
    const chunksPath = path.join(__dirname, '..', 'ml', 'chunks', `${pdfId}.txt`);
    if (fs.existsSync(chunksPath)) {
      fs.unlinkSync(chunksPath);
    }
    
    // Delete FAISS index directory
    const faissDir = path.join(__dirname, '..', 'ml', 'faiss_index', pdfId);
    if (fs.existsSync(faissDir)) {
      fs.rmSync(faissDir, { recursive: true, force: true });
    }
    
    // Remove from metadata
    pdfMetadata.delete(pdfId);
    
    res.json({ message: 'PDF deleted successfully' });
  } catch (err) {
    console.error('Error deleting PDF:', err);
    res.status(500).json({ error: 'Failed to delete PDF' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});