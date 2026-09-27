#!/usr/bin/env node
// LAYERED_PROFILE.md open question 2: how often the union bloom's one-sided
// known-token error changes an abstention verdict at the drift limit.
//
// Section 4.5's claim, which this measures rather than assumes:
//
//   "words that survive only in tombstoned records still read as known, so
//    the scorer's known-token fraction over-reads for a query about deleted
//    content, making abstention *less* likely there; it never reads a live
//    word as unknown."
//
// Nothing can be removed from a bloom, so a chain's vocabulary bloom is the
// OR of the base's and every layer's -- a bloom over every record EVER
// appended, live or tombstoned. A from-scratch compaction rebuilds it over
// live records only. The gap is the pollution.
//
// Method. For each drift level:
//   1. Tombstone a slice of the corpus (the records whose content is now
//      deleted) and append a slice of withheld records.
//   2. Build two blooms with the shipped geometry (calibrate.mjs's
//      buildVocabBloom: BLOOM_SEEDS = [0, 0x9e3779b9], minCount 3 at >= 200
//      records, bits doubling to 32x the kept-word count, capped 2^21):
//        POLLUTED -- every record ever present: live + tombstoned + appended.
//                    This is exactly the union a chain serves.
//        CLEAN    -- live records only. This is what compaction restores.
//   3. Score three probe families through the SAME fit, changing only which
//      bloom the scorer holds:
//        deleted  -- queries about tombstoned content. The pollution case:
//                    their words are in POLLUTED but (mostly) not in CLEAN.
//        live     -- queries about records still present. Control: the two
//                    blooms should agree, since nothing was removed.
//        foreign  -- off-domain queries. Control for the floor.
//   4. Report known_frac under each bloom, the verdict distribution, and the
//      flip counts -- specifically abstain-under-CLEAN -> answer-under-
//      POLLUTED, which is the harm 4.5 describes.
//
// Only the bloom differs between the two scorings: same fit, same asset,
// same retrieval, same probe texts. A verdict difference is the pollution.
//
// Usage:
//   node benchmarks/beir/measure-bloom-pollution.mjs <dataset> [--foreign <ds>]
//     [--base-frac 0.5] [--drifts 0.05,0.1,0.2,0.4] [--probes 60] [--seed 1] [--json <f>]

import { readFileSync, writeFileSync, existsSync, createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import Pikelet from 'pikelet-wasm';
import { calibrateRetrievalAbstention } from '../../packages/pikelet/src/calibrate.mjs';
import { createAbstentionScorer } from '../../packages/pikelet-wasm/complete/retrieval-abstention.mjs';
import { buildLexicalSegment } from '../../packages/pikelet-wasm/complete/builder.mjs';
import { openLexicalIndex } from '../../packages/pikelet-wasm/complete/lexical.mjs';
import { inlineEncoderDeclaration } from '../../packages/pikelet/src/complete-build.mjs';
import { embedChunksWithInlineTransformer } from '../../packages/pikelet/src/embed.mjs';
import { createInlineTransformerEmbedder } from '../../packages/pikelet-wasm/complete/inline-transformer.mjs';
import createEncoder from '../../packages/pikelet-wasm/complete/encoder-kernels/encoder.node.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const dataset = args[0];
if (!dataset || dataset.startsWith('--')) {
  console.error('usage: node benchmarks/beir/measure-bloom-pollution.mjs <dataset> [--foreign <ds>] [--base-frac 0.5] [--drifts 0.05,0.1,0.2,0.4] [--probes 60] [--seed 1] [--json <f>]');
  process.exit(1);
}
const argAfter = (f, d) => { const i = args.indexOf(f); return i === -1 ? d : args[i + 1]; };
const FOREIGN = argAfter('--foreign', dataset === 'scifact' ? 'nfcorpus' : 'scifact');
const BASE_FRAC = Number.parseFloat(argAfter('--base-frac', '0.5'));
const DRIFTS = argAfter('--drifts', '0.05,0.1,0.2,0.4').split(',').map(Number).filter((x) => Number.isFinite(x) && x > 0);
const PROBES = Number.parseInt(argAfter('--probes', '60'), 10);
const SEED = Number.parseInt(argAfter('--seed', '1'), 10);
const jsonOut = argAfter('--json', null);

// calibrate.mjs's bloom geometry, replicated exactly (it is not exported).
const BLOOM_SEEDS = [0, 0x9e3779b9];
function fnv1a(str, seed, bits) {
  let h = 0x811c9dc5 ^ seed;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h >>> 0) % bits;
}
const tokenize = (s) => String(s || '').toLowerCase().match(/[a-z0-9']+/g) || [];
// Must mirror buildVocabBloom: same minCount rule, same bit sizing.
function buildBloomOver(texts, bitsOverride = null, minCountOverride = null) {
  const counts = new Map();
  for (const t of texts) for (const w of tokenize(t)) counts.set(w, (counts.get(w) || 0) + 1);
  const minCount = minCountOverride ?? (texts.length >= 200 ? 3 : 1);
  const kept = [...counts.entries()].filter(([, c]) => c >= minCount).map(([w]) => w);
  let bits = bitsOverride;
  if (!bits) { bits = 1 << 14; while (bits < kept.length * 32 && bits < (1 << 21)) bits <<= 1; }
  const bloom = new Uint8Array(bits / 8);
  for (const w of kept) for (const seed of BLOOM_SEEDS) { const bit = fnv1a(w, seed, bits); bloom[bit >> 3] |= 1 << (bit & 7); }
  return { bloom, bits, minCount, keptWords: kept.length };
}
const setBitCount = (u8) => { let n = 0; for (const b of u8) { let x = b; while (x) { n += x & 1; x >>= 1; } } return n; };

function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
async function readJsonl(f) {
  const rows = [];
  const rl = createInterface({ input: createReadStream(f), crlfDelay: Infinity });
  for await (const line of rl) { const t = line.trim(); if (t) rows.push(JSON.parse(t)); }
  return rows;
}

const encoderDir = path.resolve(__dirname, '..', '..', 'examples', 'one-file-search', 'encoder-spike', 'real');
for (const p of [path.join(__dirname, 'work', dataset, 'records.jsonl'), path.join(__dirname, 'work', FOREIGN, 'records.jsonl'),
  path.join(encoderDir, 'vocab.txt'), path.join(encoderDir, 'encoder-weights.bin')]) {
  if (!existsSync(p)) { console.error(`missing ${p}`); process.exit(1); }
}

const allRecords = await readJsonl(path.join(__dirname, 'work', dataset, 'records.jsonl'));
const foreignRecords = await readJsonl(path.join(__dirname, 'work', FOREIGN, 'records.jsonl'));
const rng = mulberry32(SEED);
const order = allRecords.map((_, i) => i);
for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
const baseCount = Math.round(order.length * BASE_FRAC);
const baseIdx = order.slice(0, baseCount);
const withheldIdx = order.slice(baseCount);
const chunkOf = (r, id) => ({ id, title: (allRecords[r].title || `record ${r}`).trim() || `record ${r}`, text: allRecords[r].text, sourcePath: `r${r}.md` });
const baseChunks = baseIdx.map((r, i) => chunkOf(r, i));

const config = {
  embedding: { dims: 384, mode: 'inline-transformer', pooling: 'mean', normalize: true, prefixPolicy: { query: '', passage: '' } },
  index: { metric: 'cosine', quantized: true, M: 16, efConstruction: 200, efSearch: 120 },
  runtime: { inlineEncoder: { vocabPath: path.join(encoderDir, 'vocab.txt'), weightsPath: path.join(encoderDir, 'encoder-weights.bin') } },
};
const declaration = inlineEncoderDeclaration(config, config.runtime.inlineEncoder);
const embedder = await createInlineTransformerEmbedder({
  declaration, vocabText: readFileSync(path.join(encoderDir, 'vocab.txt'), 'utf8'),
  blob: readFileSync(path.join(encoderDir, 'encoder-weights.bin')), createEncoder, verify: false,
});
const embedQuery = async (t) => (await embedder.embed(t)).vector;

async function embedAll(texts, label) {
  const t0 = performance.now();
  const v = await embedChunksWithInlineTransformer(texts.map((text) => ({ text })), config, () => {}, __dirname);
  console.error(`  ${label}: ${texts.length} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return v;
}

console.error(`base ${baseChunks.length} / withheld ${withheldIdx.length} / foreign ${foreignRecords.length}`);
console.error('embedding base...');
const baseVectors = await embedAll(baseChunks.map((c) => c.text), 'base');

console.error('fitting the base calibrator...');
const baseLexical = openLexicalIndex(buildLexicalSegment(baseChunks.map((c) => c.text)).bytes);
const fit = await calibrateRetrievalAbstention({
  Pikelet, chunks: baseChunks, vectors: baseVectors, config,
  embedQuery, embedWordVecs: async (t) => embedder.embedWords(t), lexicalIndex: baseLexical,
  log: () => {}, projectDir: __dirname,
});
if (!fit) { console.error('calibration returned null; raise --base-frac'); process.exit(1); }
const asset = fit.calibrationJson.asset;
const baseBloomBytes = Buffer.from(fit.calibrationJson.vocabBloomBase64, 'base64');
// The shipped bloom's geometry is what every layer's bloom must match (4.5),
// so both of our blooms are built at exactly these bits/minCount.
const SHIPPED_BITS = baseBloomBytes.length * 8;
const SHIPPED_MINCOUNT = baseChunks.length >= 200 ? 3 : 1;
console.error(`fit ready: features ${asset.features.join(',')}; bloom ${SHIPPED_BITS} bits, minCount ${SHIPPED_MINCOUNT}`);
if (!asset.features.includes('known_frac')) {
  console.error('this fit does not use known_frac; bloom pollution cannot change its verdicts');
}

const maxAppend = Math.min(withheldIdx.length, Math.ceil(baseChunks.length * Math.max(...DRIFTS) / 2));
const appendVecs = new Map();
if (maxAppend > 0) {
  const slice = withheldIdx.slice(0, maxAppend);
  const v = await embedAll(slice.map((r) => allRecords[r].text), 'withheld');
  slice.forEach((r, i) => appendVecs.set(r, v[i]));
}

const qFromRecord = (r) => String(r.title || r.text || '').replace(/\s+/g, ' ').trim().split(' ').slice(0, 14).join(' ');
const foreignProbes = [];
for (const r of foreignRecords.slice(0, PROBES)) {
  const text = qFromRecord(r);
  foreignProbes.push({ text, vector: await embedQuery(text) });
}
const probeVecCache = new Map();
async function probeFor(recIdx) {
  if (!probeVecCache.has(recIdx)) {
    const text = qFromRecord(allRecords[recIdx]);
    probeVecCache.set(recIdx, { text, vector: await embedQuery(text) });
  }
  return probeVecCache.get(recIdx);
}

const splitHeadingBody = (t) => { const s = String(t || ''); const nl = s.indexOf('\n'); return nl === -1 ? { heading: s, body: '' } : { heading: s.slice(0, nl), body: s.slice(nl + 1) }; };

async function scoreFamily(scorer, probeList, liveChunks, liveVectors) {
  const dim = 384;
  const rows = [];
  for (const p of probeList) {
    const scored = [];
    for (let i = 0; i < liveVectors.length; i++) {
      let dot = 0; const v = liveVectors[i];
      for (let d = 0; d < dim; d++) dot += p.vector[d] * v[d];
      scored.push({ i, distance: 1 - dot });
    }
    scored.sort((a, b) => a.distance - b.distance || a.i - b.i);
    const top = scored.slice(0, 10);
    const need = scorer.usesPassage ? (scorer.passagesNeeded || 1) : 0;
    const out = await scorer.score(
      p.text,
      top.map((h) => ({ id: h.i, distance: h.distance })),
      top.slice(0, need).map((h) => splitHeadingBody(liveChunks[h.i].text)),
    );
    rows.push({ p: out.p, verdict: out.verdict, known: out.signals ? out.signals.known_frac : null });
  }
  return rows;
}

const results = [];
for (const drift of DRIFTS) {
  const change = Math.round(baseChunks.length * drift);
  const nAppend = Math.ceil(change / 2);
  const nTomb = change - nAppend;
  const r2 = mulberry32(SEED + 17);
  const tombstoned = new Set();
  while (tombstoned.size < Math.min(nTomb, baseChunks.length - 2)) tombstoned.add(Math.floor(r2() * baseChunks.length));
  const appendRecs = withheldIdx.slice(0, Math.min(nAppend, withheldIdx.length));

  const liveChunks = [];
  const liveVectors = [];
  for (let i = 0; i < baseChunks.length; i++) if (!tombstoned.has(i)) { liveChunks.push(baseChunks[i]); liveVectors.push(baseVectors[i]); }
  for (const r of appendRecs) { liveChunks.push(chunkOf(r, liveChunks.length)); liveVectors.push(appendVecs.get(r)); }

  // POLLUTED: every record ever present in the chain. CLEAN: live only.
  const everTexts = [...baseChunks.map((c) => c.text), ...appendRecs.map((r) => allRecords[r].text)];
  const polluted = buildBloomOver(everTexts, SHIPPED_BITS, SHIPPED_MINCOUNT);
  const clean = buildBloomOver(liveChunks.map((c) => c.text), SHIPPED_BITS, SHIPPED_MINCOUNT);

  const scorerPolluted = createAbstentionScorer(asset, polluted.bloom);
  const scorerClean = createAbstentionScorer(asset, clean.bloom);

  // deleted: probes about tombstoned records -- the pollution case.
  const deletedProbes = [];
  for (const i of [...tombstoned].slice(0, PROBES)) deletedProbes.push(await probeFor(baseIdx[i]));
  // live: probes about records still present -- control.
  const liveProbes = [];
  for (let i = 0; i < baseChunks.length && liveProbes.length < PROBES; i++) if (!tombstoned.has(i)) liveProbes.push(await probeFor(baseIdx[i]));

  const fams = { deleted: deletedProbes, live: liveProbes, foreign: foreignProbes };
  const row = { drift, change, tombstoned: tombstoned.size, appended: appendRecs.length, live: liveChunks.length,
    bloom: { bits: SHIPPED_BITS, polluted_set_bits: setBitCount(polluted.bloom), clean_set_bits: setBitCount(clean.bloom),
      polluted_kept: polluted.keptWords, clean_kept: clean.keptWords }, families: {} };

  for (const [fam, list] of Object.entries(fams)) {
    if (!list.length) { row.families[fam] = { n: 0 }; continue; }
    const P = await scoreFamily(scorerPolluted, list, liveChunks, liveVectors);
    const C = await scoreFamily(scorerClean, list, liveChunks, liveVectors);
    const mean = (a, f) => a.reduce((s, x) => s + (Number.isFinite(f(x)) ? f(x) : 0), 0) / a.length;
    let cleanAbstainPollutedAnswer = 0, cleanAnswerPollutedAbstain = 0, anyFlip = 0;
    for (let i = 0; i < P.length; i++) {
      if (P[i].verdict !== C[i].verdict) anyFlip++;
      if (C[i].verdict === 'abstain' && P[i].verdict !== 'abstain') cleanAbstainPollutedAnswer++;
      if (C[i].verdict !== 'abstain' && P[i].verdict === 'abstain') cleanAnswerPollutedAbstain++;
    }
    row.families[fam] = {
      n: P.length,
      known_frac_polluted: mean(P, (x) => x.known), known_frac_clean: mean(C, (x) => x.known),
      known_frac_delta: mean(P, (x) => x.known) - mean(C, (x) => x.known),
      mean_p_polluted: mean(P, (x) => x.p), mean_p_clean: mean(C, (x) => x.p),
      abstain_polluted: P.filter((x) => x.verdict === 'abstain').length / P.length,
      abstain_clean: C.filter((x) => x.verdict === 'abstain').length / C.length,
      verdict_flips: anyFlip,
      // The harm 4.5 describes: the clean bloom would have abstained, the
      // polluted union answers instead.
      flips_clean_abstain_to_polluted_answer: cleanAbstainPollutedAnswer,
      // The direction 4.5 says cannot happen (a live word read as unknown).
      flips_clean_answer_to_polluted_abstain: cleanAnswerPollutedAbstain,
    };
  }
  const d = row.families.deleted;
  console.error(`drift ${drift}: -${row.tombstoned}/+${row.appended} | deleted-probe known_frac ${d.known_frac_clean?.toFixed(3)} -> ${d.known_frac_polluted?.toFixed(3)} `
    + `| flips ${d.flips_clean_abstain_to_polluted_answer}/${d.n} clean-abstain->polluted-answer`);
  results.push(row);
}

embedder.dispose();
const out = {
  dataset, foreign: FOREIGN, base_frac: BASE_FRAC, seed: SEED, probes: PROBES,
  base_records: baseChunks.length, fit_features: asset.features,
  // The sign of known_frac's weight decides which way pollution pushes: the
  // smoke run's polluted bloom LOWERED mean p on deleted probes, which only
  // happens if this weight is negative -- the opposite of 4.5's predicted
  // direction. Recorded so the direction is read off the fit, not inferred.
  fit_weights: Object.fromEntries((asset.features || []).map((f, i) => [f, asset.weights[i]])),
  fit_thresholds: asset.thresholds,
  fit_standardize: asset.standardize,
  note: 'Only the bloom differs between the two scorings: same fit, same asset, same retrieval, same probe texts. '
    + 'POLLUTED is built over every record ever present (live + tombstoned + appended), which is the union a chain serves; '
    + 'CLEAN is live records only, which is what compaction restores. The family that matters is "deleted": probes about '
    + 'tombstoned content, whose words survive in POLLUTED but not CLEAN. flips_clean_answer_to_polluted_abstain should be 0 '
    + 'in every row -- a bloom can only gain bits, so a live word can never read as unknown; a non-zero value is a harness bug.',
  results,
};
console.log(JSON.stringify(out, null, 2));
if (jsonOut) { writeFileSync(jsonOut, `${JSON.stringify(out, null, 2)}\n`); console.error(`-> ${jsonOut}`); }
