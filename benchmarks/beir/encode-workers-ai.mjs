#!/usr/bin/env node
// Encode a BEIR dataset's corpus + test queries the way `embedding.mode:
// "workers-ai"` does — BGE-small-en-v1.5 through transformers.js — so the
// CLI's DEFAULT mode gets a relevance number on the same ladder as the
// inline MiniLM (B-F) and Arctic-XS (G-H) rows.
//
// Why this is a separate script rather than a flag on encode-pikelet.mjs:
// workers-ai is a SPLIT-ENCODER mode. The corpus is embedded locally at
// build time with transformers.js (Xenova/bge-small-en-v1.5, the hfModel in
// MODEL_MAP), while queries are embedded at request time by Cloudflare's
// AI binding (@cf/baai/bge-small-en-v1.5, the workersAiModel). Both halves
// are the same upstream BGE-small-en-v1.5, which is the premise the whole
// mode rests on and what `assertEncoderManifest` pins by name.
//
// WHAT THIS MEASURES, AND WHAT IT DOES NOT. This encodes BOTH halves
// locally with the Xenova ONNX export. That measures the MODEL, which is
// what the ladder is for. It does NOT exercise a real `env.AI.run` call, so
// it cannot detect drift between Cloudflare's hosted copy and this ONNX
// export (different quantization, runtime, possibly a different snapshot) —
// spec risk R1, which `assertEncoderManifest` guards by NAME, not by value.
// Confirming the two agree numerically needs a live Workers AI account; see
// the README's "what config I does not cover" note. Do not read these rows
// as validating the Cloudflare call path.
//
// The prefix policy is the contract (PIKELET_CLI_SPEC.md §Stage 3): BGE
// v1.5 retrieval depends on an asymmetric query-side instruction prefix and
// raw passages. DEFAULT_CONFIG.embedding.prefixPolicy in the CLI is
// { passage: '', query: DEFAULT_PREFIX } and encoder.workers-ai.js applies
// manifest.prefixPolicy.query before calling the AI binding, so this script
// applies exactly the same split. --no-prefix ablates it, the same way
// ablate-query-prefix.mjs does for Arctic-XS.
//
// Usage: node benchmarks/beir/encode-workers-ai.mjs <dataset> [--no-prefix] [--batch 16]
// Reads:  work/<dataset>/records.jsonl, cache/<dataset>/queries.jsonl,
//         cache/<dataset>/qrels/test.tsv
// Writes: work/<dataset>/vectors-corpus-workersai.f32
//         work/<dataset>/vectors-queries-workersai[-noprefix].f32 (+ .ids.json)
//         work/<dataset>/encode-workers-ai[-noprefix].json

import { readFileSync, writeFileSync, existsSync, createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { DEFAULT_PREFIX, MODEL_MAP } from '../../packages/pikelet/src/common.mjs';
import { tensorToVectors } from '../../packages/pikelet/src/embed.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const dataset = args[0];
const noPrefix = args.includes('--no-prefix');
if (!dataset || dataset.startsWith('--')) {
  console.error('usage: node benchmarks/beir/encode-workers-ai.mjs <dataset> [--no-prefix] [--batch 16]');
  process.exit(1);
}
// embed.mjs batches 16 at a time; keep the same default so this measures the
// shipped batching, and allow an override only for throughput experiments.
const batchArg = args.indexOf('--batch');
const BATCH = batchArg === -1 ? 16 : Number.parseInt(args[batchArg + 1], 10);
if (!Number.isInteger(BATCH) || BATCH < 1 || BATCH > 512) {
  console.error('--batch must be an integer between 1 and 512');
  process.exit(1);
}

// The build model workers-ai mode actually uses, read from the CLI's own
// MODEL_MAP rather than hardcoded, so this tracks the allowlist.
const BUILD_MODEL = 'bge-small-en-v1.5';
const model = MODEL_MAP[BUILD_MODEL];
if (!model) {
  console.error(`MODEL_MAP has no ${BUILD_MODEL}; the CLI's allowlist changed — update this script.`);
  process.exit(1);
}
const DIM = model.dims;

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
for (const p of [path.join(workDir, 'records.jsonl'), path.join(cacheDir, 'queries.jsonl'), path.join(cacheDir, 'qrels', 'test.tsv')]) {
  if (!existsSync(p)) {
    console.error(`missing ${p} — run download.py and convert.mjs for ${dataset} first`);
    process.exit(1);
  }
}

async function readJsonl(filePath) {
  const rows = [];
  const rl = createInterface({ input: createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) {
    const t = line.trim();
    if (t) rows.push(JSON.parse(t));
  }
  return rows;
}

function testQueryIds() {
  const text = readFileSync(path.join(cacheDir, 'qrels', 'test.tsv'), 'utf8');
  const ids = new Set();
  for (const line of text.split('\n').slice(1)) {
    const [qid] = line.split('\t');
    if (qid) ids.add(qid);
  }
  return ids;
}

const records = await readJsonl(path.join(workDir, 'records.jsonl')); // pikeletRow order
const queries = await readJsonl(path.join(cacheDir, 'queries.jsonl'));
const testIds = testQueryIds();
const testQueries = queries.filter((q) => testIds.has(q._id));

let transformers;
try {
  transformers = await import('@xenova/transformers');
} catch (error) {
  console.error(`Failed to load @xenova/transformers (workers-ai's build-time encoder): ${error.message}`);
  process.exit(1);
}

// Same pipeline construction as embed.mjs's workers-ai branch: the quantized
// ONNX export of the MODEL_MAP hfModel.
console.error(`loading ${model.hfModel} (quantized ONNX; first run downloads ~30 MB to the platform cache)...`);
const tLoad = performance.now();
const extractor = await transformers.pipeline('feature-extraction', model.hfModel, { quantized: true });
const loadMs = performance.now() - tLoad;
console.error(`encoder ready in ${(loadMs / 1000).toFixed(1)}s`);

// embed.mjs pools/normalizes from embeddingConfig; DEFAULT_CONFIG.embedding
// sets mean + normalize true for this mode.
const POOLING = 'mean';
const NORMALIZE = true;

async function embedTexts(texts, label) {
  const flat = new Float32Array(texts.length * DIM);
  const t0 = performance.now();
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH);
    const output = await extractor(batch, { pooling: POOLING, normalize: NORMALIZE });
    const vectors = tensorToVectors(output, DIM);
    if (vectors.length !== batch.length) {
      throw new Error(`${label}: expected ${batch.length} vectors from batch at ${i}, got ${vectors.length}`);
    }
    for (let j = 0; j < vectors.length; j++) {
      if (vectors[j].length !== DIM) throw new Error(`${label}: vector ${i + j} has ${vectors[j].length} dims, expected ${DIM}`);
      flat.set(vectors[j], (i + j) * DIM);
    }
    const done = Math.min(i + BATCH, texts.length);
    if (i % (BATCH * 25) === 0 || done === texts.length) {
      const rate = done / ((performance.now() - t0) / 1000);
      console.error(`${label} ${done}/${texts.length} (${rate.toFixed(1)}/s)`);
    }
  }
  return { flat, ms: performance.now() - t0 };
}

// Passages raw, queries prefixed — the asymmetry is the contract.
const passagePrefix = '';
const queryPrefix = noPrefix ? '' : DEFAULT_PREFIX;

console.error(`encoding ${records.length} corpus records (batch ${BATCH}, passage prefix ${passagePrefix ? 'ENABLED' : 'none'})...`);
const corpus = await embedTexts(records.map((r) => `${passagePrefix}${r.text}`), 'corpus');

console.error(`encoding ${testQueries.length} test queries (query prefix ${noPrefix ? 'DISABLED (ablation)' : 'ENABLED'})...`);
const q = await embedTexts(testQueries.map((t) => `${queryPrefix}${t.text}`), 'queries');

const suffix = noPrefix ? '-noprefix' : '';
writeFileSync(path.join(workDir, 'vectors-corpus-workersai.f32'), Buffer.from(corpus.flat.buffer, corpus.flat.byteOffset, corpus.flat.byteLength));
writeFileSync(path.join(workDir, `vectors-queries-workersai${suffix}.f32`), Buffer.from(q.flat.buffer, q.flat.byteOffset, q.flat.byteLength));
writeFileSync(path.join(workDir, `vectors-queries-workersai${suffix}.ids.json`), JSON.stringify({ ids: testQueries.map((t) => t._id), dim: DIM, count: testQueries.length }));
writeFileSync(path.join(workDir, `encode-workers-ai${suffix}.json`), JSON.stringify({
  dataset,
  mode: 'workers-ai',
  build_model: BUILD_MODEL,
  hf_model: model.hfModel,          // what encoded these vectors, locally
  workers_ai_model: model.workersAiModel, // what would encode queries in production
  measures_cloudflare_call: false,  // see the header note: model only, not the AI binding
  pooling: POOLING,
  normalize: NORMALIZE,
  batch: BATCH,
  passage_prefix: passagePrefix,
  query_prefix: queryPrefix,
  encoder_load_ms: loadMs,
  corpus_count: records.length,
  corpus_encode_ms: corpus.ms,
  query_count: testQueries.length,
  query_encode_ms: q.ms,
  cores: typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length,
  node: process.version,
  platform: process.platform,
}, null, 2));

console.log(`${dataset} workers-ai${suffix}: corpus ${records.length} in ${(corpus.ms / 1000).toFixed(1)}s `
  + `(${(records.length / (corpus.ms / 1000)).toFixed(1)}/s), queries ${testQueries.length} in ${(q.ms / 1000).toFixed(1)}s`);
