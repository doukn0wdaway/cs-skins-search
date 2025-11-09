const { pipeline } = require("@xenova/transformers");
const fs = require("node:fs");

generateEmbeddings();

async function getEmbedding(text, embedder) {
  const embeddings = await embedder(text);
  return averaging(embeddings);
}

function loadDataset() {
  const dataset = JSON.parse(fs.readFileSync("./dataset.json", "utf-8"));

  const keys = Object.keys(dataset);
  const transformedDataset = [];

  keys.forEach((key) => transformedDataset.push(dataset[key]));

  return transformedDataset;
}

function saveEmbeddings(embeddings) {
  return fs.writeFileSync(
    "./embeddings.json",
    JSON.stringify(embeddings),
    "utf-8",
  );
}

async function generateEmbeddings() {
  const embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
  );

  const dataset = loadDataset();
  const datasetLength = dataset.length;
  const results = [];
  for (let i = 0; i < datasetLength; i++) {
    const item = dataset[i];
    const text = `${item?.name} ${item?.description} ${item?.team?.name} ${item?.rarity?.name} ${item?.wear?.name}`; // TODO :ADD COLLECTIONS?

    try {
      const textEmbedding = await getEmbedding(text, embedder);

      results.push({ id: item.id, textEmbedding });
      await sleep(15);
    } catch (e) {
      console.warn(e);
    }

    if (i % 100 == 0) console.log(`Progress: ${i} out of ${datasetLength}`);
  }

  saveEmbeddings(results);
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
