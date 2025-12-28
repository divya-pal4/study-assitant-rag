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

    const chunksFilePath = path.join(__dirname, "ml", "chunks.txt");
    fs.writeFileSync(chunksFilePath, "");

    chunks.forEach((chunk, index) => {
      fs.appendFileSync(
        chunksFilePath,
        `--- CHUNK ${index + 1} ---\n${chunk}\n\n`
      );
    });

    res.json({
      message: "PDF uploaded, chunked & saved successfully",
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
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    
    const response = await axios.post(
    "http://localhost:8000/ask_llm",
    {
    question: question,
    top_k: 3
    }
    );

    res.json({
    question: question,
    answer: response.data.answer,
    sources: response.data.sources
  });


  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Failed to get answer from llm" });
  }
});


app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
