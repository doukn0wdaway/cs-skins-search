const { pipeline } = require("@xenova/transformers");
const fs = require("node:fs");
const express = require("express");
const MiniSearch = require("minisearch");
const fuzzysort = require("fuzzysort");

let embedderInstance = null;

async function getEmbedding(text) {
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
function getEmbeddings() {
  if (!embeddings) {
    embeddings = JSON.parse(fs.readFileSync("./embeddings.json", "utf-8"));
  }
  return embeddings;
}

let minifiedData = null;
function getMinifiedData() {
  if (!minifiedData) {
    minifiedData = JSON.parse(fs.readFileSync("./only-names.json", "utf-8"));
  }
  return minifiedData;
}

app.get("/search", async (req, res) => {
  const { query } = req.query;
  if (!query) {
    return res.status(400).json({ error: "Query parameter is required." });
  }

  const searchQueryEmbedding = await getEmbedding(query);
  const embeddings = getEmbeddings();

  const results = [];

  const startTime = performance.now();
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
  const endTime = performance.now();
  const elapsed = endTime - startTime;

  results.sort((a, b) => b.similarity - a.similarity);

  res.json({
    query,
    elapsedTime: elapsed.toFixed(3),
    results: results.slice(0, 5),
  });
});

let minisearch = null;
function search(query) {
  if (!minisearch) {
    const minifiedData = getMinifiedData();

    minisearch = new MiniSearch({
      fields: ["name", "description"],
      idField: "id",
      fuzzy: 1,
      minTermLength: 1,
    });

    minisearch.addAll(minifiedData);
  }

  const results = minisearch.search(query);
  return results;
}

app.get("/search-minified", async (req, res) => {
  const { query } = req.query;
  if (!query) {
    return res.status(400).json({ error: "Query parameter is required." });
  }

  const startTime = performance.now();

  const results = search(query);

  const endTime = performance.now();
  const elapsed = endTime - startTime;

  res.json({
    query,
    elapsedTime: elapsed.toFixed(3),
    results: results.slice(0, 5),
  });
});

app.get("/search-fuzzy", async (req, res) => {
  const { query } = req.query;
  if (!query) {
    return res.status(400).json({ error: "Query parameter is required." });
  }

  const startTime = performance.now();

  const data = getMinifiedData();

  const results = fuzzysort.go(query, data, { keys: ["name"] });

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
