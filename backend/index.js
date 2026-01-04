const axios = require("axios");
const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });


function chunkText(text, wordsPerChunk = 500) {
  const words = text.split(/\s+/);
  const chunks = [];

  for (let i = 0; i < words.length; i += wordsPerChunk) {
    chunks.push(words.slice(i, i + wordsPerChunk).join(" "));
  }

  return chunks;
}


app.get("/", (req, res) => {
  res.send("Study Assistant Backend is running");
});


app.post("/upload", upload.single("pdf"), async (req, res) => {
  try {
    const dataBuffer = fs.readFileSync(req.file.path);
    const pdfData = await pdfParse(dataBuffer);

    const text = pdfData.text;
    const chunks = chunkText(text, 500);

    // ensure we write to the project's top-level `ml/` folder (not backend/ml)
    const { v4: uuidv4 } = require('uuid');
    const pdfId = uuidv4();

    const chunksDir = path.join(__dirname, "..", "ml");
    fs.mkdirSync(chunksDir, { recursive: true });
    // write per-pdf chunks file (JSON-lines)
    const chunksFilePath = path.join(chunksDir, `chunks_${pdfId}.txt`);
    fs.writeFileSync(chunksFilePath, "");

    // create a per-pdf faiss output dir
    const faissOutDir = path.join(chunksDir, "faiss_index", pdfId);
    fs.mkdirSync(faissOutDir, { recursive: true });

    chunks.forEach((chunk, index) => {
      const chunkObj = {
        pdf: req.file.originalname,
        chunk_id: index + 1,
        text: chunk
      };

      fs.appendFileSync(
        chunksFilePath,
        JSON.stringify(chunkObj) + "\n"
      );
    });


    // spawn embeddings generation in background
    const { spawn } = require('child_process');
    const py = spawn('python3', [
      path.join(__dirname, '..', 'ml', 'generate_embeddings.py'),
      '--chunks-file', chunksFilePath,
      '--faiss-dir', faissOutDir
    ], { stdio: 'ignore', detached: true });
    py.unref();

    res.json({
      message: "PDF uploaded, chunked & saved successfully",
      pdf_id: pdfId,
      totalChunks: chunks.length,
      preview: chunks[0].substring(0, 300)
    });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
});

app.post("/ask", async (req, res) => {
  try {
    const { question, pdf_id, top_k } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    const payload = { question: question, top_k: top_k || 3 };
    if (pdf_id) payload.pdf_id = pdf_id;

    const response = await axios.post(
      "http://localhost:8000/ask_llm",
      payload,
      { timeout: 0 } // IMPORTANT for long LLM calls
    );

    res.json({
      question: question,
      answer: response.data.answer,
      sources: response.data.sources
    });

  } catch (error) {
    console.error("ASK ERROR:", error.message);
    res.status(500).json({ error: "Failed to get answer" });
  }
});



app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
