#!/usr/bin/env node
'use strict';
/**
 * Distinct-vector churn benchmark on a real dataset (NYTimes-256, cosine).
 *
 * Same protocol as churn_scale.js — a fixed live population is replaced
 * completely for several rounds, recall@k against exact brute force over the
 * live set is measured after every turnover, after compact(), and against a
 * fresh build of the final population — but every generation is a disjoint
 * slice of real vectors instead of synthetic clusters, and the HNSW build
 * seed is a parameter so the run can be repeated.
 *
 * Usage:
 *   node benchmarks/engine/churn_real.js [--data nytimes] [--population 48000]
 *     [--rounds 5] [--queries 1000] [--k 10] [--m 12] [--ef-construction 75]
 *     [--ef-search 200] [--seed 1] [--output <json>]
 *
 * population * (rounds + 1) must fit the base set (NYTimes: 290,000 rows, so
 * 48,000 x 6 generations).
 */
const fs = require('fs');
const path = require('path');
const Pikelet = require('pikelet-wasm');

const args = process.argv.slice(2);
function intArg(name, fallback) {
  const index = args.indexOf(`--${name}`);
  if (index < 0) return fallback;
  const value = Number.parseInt(args[index + 1], 10);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`--${name} must be a positive integer`);
  return value;
}
function stringArg(name, fallback) {
  const index = args.indexOf(`--${name}`);
  return index < 0 ? fallback : args[index + 1];
}
const DATA_DIR = path.resolve(stringArg('data', path.join(__dirname, '..', '..', 'nytimes')));
const POPULATION = intArg('population', 48_000);
const ROUNDS = intArg('rounds', 5);
const QUERY_COUNT = intArg('queries', 1000);
const K = intArg('k', 10);
const M = intArg('m', 12);
const EF_CONSTRUCTION = intArg('ef-construction', 75);
const EF_SEARCH = intArg('ef-search', 200);
const SEED = intArg('seed', 1);
const BATCH_SIZE = intArg('batch-size', 1000);
const OUTPUT = stringArg('output', path.join(
  __dirname, '..', 'results', 'raw',
  `churn_real_nytimes_seed${SEED}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`,
));

function readFvecs(filePath) {
  const buf = fs.readFileSync(filePath);
  const dim = buf.readInt32LE(0);
  const stride = 4 + dim * 4;
  const count = Math.floor(buf.length / stride);
  const vectors = new Float32Array(count * dim);
  for (let i = 0; i < count; i++) {
    const at = i * stride + 4;
    for (let d = 0; d < dim; d++) vectors[i * dim + d] = buf.readFloatLE(at + d * 4);
  }
  return { vectors, dim, count };
}

// Unit-normalize every row and drop zero rows (NYTimes has a few all-zero
// vectors, which a cosine index rejects). Returns the kept rows, packed.
function normalizeAll(flat, dim) {
  const total = flat.length / dim;
  const kept = new Float32Array(flat.length);
  let out = 0;
  for (let row = 0; row < total; row++) {
    const off = row * dim;
    let normSq = 0;
    for (let d = 0; d < dim; d++) normSq += flat[off + d] * flat[off + d];
    if (!(normSq > 0)) continue;
    const inv = 1 / Math.sqrt(normSq);
    for (let d = 0; d < dim; d++) kept[out * dim + d] = flat[off + d] * inv;
    out++;
  }
  return { vectors: kept.subarray(0, out * dim), count: out, dropped: total - out };
}

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A fixed shuffle of the base rows (independent of --seed, so every seed
// sees the same generations and only the HNSW build differs).
function shuffledOrder(count) {
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  const random = mulberry32(0x4e595449);
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }
  return order;
}

let DIMS = 0;
function generation(base, order, round) {
  const vectors = new Float32Array(POPULATION * DIMS);
  for (let row = 0; row < POPULATION; row++) {
    const src = order[round * POPULATION + row] * DIMS;
    vectors.set(base.subarray(src, src + DIMS), row * DIMS);
  }
  return vectors;
}

function rows(flat, start, end) {
  const result = new Array(end - start);
  for (let row = start; row < end; row++) result[row - start] = flat.subarray(row * DIMS, (row + 1) * DIMS);
  return result;
}

function exactTopK(vectors, ids, query) {
  const best = [];
  for (let row = 0; row < ids.length; row++) {
    const offset = row * DIMS;
    let dot = 0;
    for (let d = 0; d < DIMS; d++) dot += vectors[offset + d] * query[d];
    const candidate = { id: ids[row], distance: 1 - dot };
    if (best.length < K) {
      best.push(candidate);
      best.sort((a, b) => a.distance - b.distance);
    } else if (candidate.distance < best[K - 1].distance) {
      best[K - 1] = candidate;
      best.sort((a, b) => a.distance - b.distance);
    }
  }
  return new Set(best.map((entry) => entry.id));
}

function measure(index, vectors, ids, queries) {
  let recall = 0;
  let resultCount = 0;
  const latencies = [];
  for (const query of queries) {
    const truth = exactTopK(vectors, ids, query);
    const start = performance.now();
    const results = index.search(query, K, { efSearch: EF_SEARCH });
    latencies.push(performance.now() - start);
    resultCount += results.length;
    let hits = 0;
    for (const result of results) if (truth.has(result.id)) hits++;
    recall += hits / K;
  }
  latencies.sort((a, b) => a - b);
  return {
    recall: recall / queries.length,
    averageResultCount: resultCount / queries.length,
    p50Ms: latencies[Math.floor((latencies.length - 1) * 0.50)],
    p99Ms: latencies[Math.floor((latencies.length - 1) * 0.99)],
    count: index.count,
    liveCount: index.liveCount,
    deletedCount: index.deletedCount,
    deletedRatio: index.deletedRatio,
    logicalIndexBytes: index.memoryUsage.logicalIndexBytes,
    wasmHeapBytes: index.memoryUsage.wasmHeapBytes,
  };
}

async function insertPopulation(index, vectors) {
  const ids = [];
  for (let start = 0; start < POPULATION; start += BATCH_SIZE) {
    const end = Math.min(POPULATION, start + BATCH_SIZE);
    ids.push(...index.addBatch(rows(vectors, start, end)));
  }
  return ids;
}

async function replacePopulation(index, oldIds, vectors) {
  const nextIds = [];
  for (let start = 0; start < POPULATION; start += BATCH_SIZE) {
    const end = Math.min(POPULATION, start + BATCH_SIZE);
    for (let row = start; row < end; row++) index.delete(oldIds[row]);
    nextIds.push(...index.addBatch(rows(vectors, start, end)));
  }
  return nextIds;
}

async function main() {
  const base = readFvecs(path.join(DATA_DIR, 'nytimes_base.fvecs'));
  const queryFile = readFvecs(path.join(DATA_DIR, 'nytimes_query.fvecs'));
  DIMS = base.dim;
  if (POPULATION * (ROUNDS + 1) > base.count) {
    throw new Error(`population ${POPULATION} x ${ROUNDS + 1} generations exceeds the ${base.count} base rows`);
  }
  if (QUERY_COUNT > queryFile.count) throw new Error(`--queries exceeds the ${queryFile.count} query rows`);
  const baseKept = normalizeAll(base.vectors, DIMS);
  const queryKept = normalizeAll(queryFile.vectors, DIMS);
  if (baseKept.dropped || queryKept.dropped) console.log(`dropped zero rows: base ${baseKept.dropped}, queries ${queryKept.dropped}`);
  if (POPULATION * (ROUNDS + 1) > baseKept.count) throw new Error(`population x generations exceeds the ${baseKept.count} usable base rows`);
  const order = shuffledOrder(baseKept.count);
  const queries = rows(queryKept.vectors, 0, QUERY_COUNT);
  const createOptions = { dim: DIMS, metric: 'cosine', quantized: true, M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH, seed: SEED };
  console.log(`nytimes: ${base.count} base rows, dim ${DIMS}; population ${POPULATION} x ${ROUNDS + 1} generations, ${QUERY_COUNT} queries, k=${K}, M=${M}, efC=${EF_CONSTRUCTION}, efS=${EF_SEARCH}, seed ${SEED}`);

  const index = await Pikelet.create({ ...createOptions, maxElements: POPULATION * (ROUNDS + 1) });
  const measurements = [];
  try {
    let liveVectors = generation(baseKept.vectors, order, 0);
    const buildStart = performance.now();
    let liveIds = await insertPopulation(index, liveVectors);
    const buildMs = performance.now() - buildStart;
    const baseline = { stage: 'baseline', round: 0, buildMs, ...measure(index, liveVectors, liveIds, queries) };
    measurements.push(baseline);
    console.log(`baseline recall=${(baseline.recall * 100).toFixed(2)}% p50=${baseline.p50Ms.toFixed(2)}ms build=${(buildMs / 1000).toFixed(1)}s`);

    for (let round = 1; round <= ROUNDS; round++) {
      liveVectors = generation(baseKept.vectors, order, round);
      const t0 = performance.now();
      liveIds = await replacePopulation(index, liveIds, liveVectors);
      const turnoverMs = performance.now() - t0;
      const result = { stage: 'turnover', round, turnoverMs, ...measure(index, liveVectors, liveIds, queries) };
      measurements.push(result);
      console.log(`round=${round} deleted=${(result.deletedRatio * 100).toFixed(1)}% recall=${(result.recall * 100).toFixed(2)}% results=${result.averageResultCount.toFixed(1)} p50=${result.p50Ms.toFixed(2)}ms logicalMB=${(result.logicalIndexBytes / 1048576).toFixed(0)} turnover=${(turnoverMs / 1000).toFixed(1)}s`);
    }

    const compactStart = performance.now();
    index.compact();
    const compactMs = performance.now() - compactStart;
    const compacted = { stage: 'compacted', round: ROUNDS, compactMs, ...measure(index, liveVectors, liveIds, queries) };
    measurements.push(compacted);
    console.log(`compacted recall=${(compacted.recall * 100).toFixed(2)}% p50=${compacted.p50Ms.toFixed(2)}ms compact=${(compactMs / 1000).toFixed(1)}s logicalMB=${(compacted.logicalIndexBytes / 1048576).toFixed(0)}`);

    const fresh = await Pikelet.create({ ...createOptions, maxElements: POPULATION });
    try {
      const freshStart = performance.now();
      const freshIds = await insertPopulation(fresh, liveVectors);
      const freshBuildMs = performance.now() - freshStart;
      const ref = { stage: 'fresh-reference', round: ROUNDS, freshBuildMs, ...measure(fresh, liveVectors, freshIds, queries) };
      measurements.push(ref);
      console.log(`fresh-reference recall=${(ref.recall * 100).toFixed(2)}% p50=${ref.p50Ms.toFixed(2)}ms build=${(freshBuildMs / 1000).toFixed(1)}s`);
    } finally {
      fresh.dispose();
    }

    const output = {
      timestamp: new Date().toISOString(),
      dataset: 'nytimes-256-angular',
      params: { POPULATION, DIMS, ROUNDS, QUERY_COUNT, K, M, EF_CONSTRUCTION, EF_SEARCH, SEED },
      measurements,
    };
    fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
    fs.writeFileSync(OUTPUT, `${JSON.stringify(output, null, 2)}\n`);
    console.log(`output=${OUTPUT}`);
  } finally {
    index.dispose();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
