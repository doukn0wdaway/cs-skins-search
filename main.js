const { pipeline } = require("@xenova/transformers");
const fs = require("node:fs");

async function getEmbedding(text, embedder) {
  const embeddings = await embedder(text);
  return averaging(embeddings);
}

main();

function loadEmbeddings() {
  return JSON.parse(fs.readFileSync("./embeddings.json", "utf-8"));
}

async function main() {
  const searchQuery = "Desert eagle printstream field tested";
  const embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
  );

  const searchQueryEmbedding = await getEmbedding(searchQuery, embedder);
  const embeddings = loadEmbeddings();

  const results = [];
  const startTime = performance.now();
  embeddings.forEach((item) => {
    const similarity = cosineSimilarity(
      searchQueryEmbedding,
      item.textEmbedding,
    );
    if (similarity > 0.55) {
      results.push({
        id: item.id,
        similarity,
      });
    }
  });
  results.sort((a, b) => b.similarity - a.similarity);
  // results.sort((a, b) => a.similarity - b.similarity);

  const endTime = performance.now();
  const elapsed = endTime - startTime;

  console.log(`Elapsed time: ${elapsed.toFixed(3)} ms`);
  console.log(results.slice(0, 5));
}

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
  const data = embeddings.data; // данные тензора (все эмбеддинги)

  if (dims[1] === 0) {
    throw new Error(
      "Количество токенов равно 0, невозможно вычислить усреднение.",
    );
  }

  const averaged = new Float32Array(dims[2]); // dims[2] — это размерность каждого эмбеддинга

  // Усредняем значения для каждого измерения эмбеддинга
  for (let i = 0; i < dims[1]; i++) {
    // dims[1] — количество токенов
    for (let j = 0; j < dims[2]; j++) {
      // dims[2] — размерность эмбеддинга
      averaged[j] += data[i * dims[2] + j]; // Складываем значения
    }
  }

  // Делаем усреднение, делим на количество токенов
  for (let j = 0; j < dims[2]; j++) {
    averaged[j] /= dims[1]; // Делим каждое значение на количество токенов
  }

  if (averaged.length !== 384) {
    // console.log(averaged.length);
  }
  return averaged;
}
