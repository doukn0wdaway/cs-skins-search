const { pipeline } = require("@xenova/transformers");
const fs = require("node:fs");
const express = require("express");

let embedderInstance = null;

async function getEmbedding(text) {
  console.log(`embedderInstance is ${embedderInstance}`);
  if (!embedderInstance) {
    embedderInstance = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2",
    );
  }
  const embeddings = await embedderInstance(text);
  return averaging(embeddings);
}
const app = express();

let embeddings = null;
function loadEmbeddings() {
  if (!embeddings) {
    embeddings = JSON.parse(fs.readFileSync("./embeddings.json", "utf-8"));
  }
  return embeddings;
}

app.get("/search", async (req, res) => {
  const { query } = req.query;
  if (!query) {
    return res.status(400).json({ error: "Query parameter is required." });
  }

  const startTime = performance.now();
  const searchQueryEmbedding = await getEmbedding(query);
  const embeddings = loadEmbeddings();

  const results = [];

  embeddings.forEach((item) => {
    const similarity = cosineSimilarity(
      searchQueryEmbedding,
      item.textEmbedding,
    );
    if (similarity > 0.4) {
      results.push({
        id: item.id,
        similarity,
      });
    }
  });

  results.sort((a, b) => b.similarity - a.similarity);

  const endTime = performance.now();
  const elapsed = endTime - startTime;

  res.json({
    query,
    elapsedTime: elapsed.toFixed(3),
    results: results.slice(0, 5),
  });
});

function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;
  return dotProduct / denominator;
}

function averaging(embeddings) {
  const dims = embeddings.dims; // dims[1] - количество токенов, dims[2] - размерность эмбеддинга
  const data = embeddings.data;

  if (dims[1] === 0) {
    throw new Error(
      "Количество токенов равно 0, невозможно вычислить усреднение.",
    );
  }

  const averaged = new Float32Array(dims[2]);

  for (let i = 0; i < dims[1]; i++) {
    for (let j = 0; j < dims[2]; j++) {
      averaged[j] += data[i * dims[2] + j];
    }
  }

  for (let j = 0; j < dims[2]; j++) {
    averaged[j] /= dims[1];
  }

  return averaged;
}
app.listen(3000, () => {
  console.log(`Server is running at http://localhost:${3000}`);
});
