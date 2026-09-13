#!/usr/bin/env node
// Measurement for spec/COMPLETE_PROFILE.md section 8, question 1: corpus
// compression (per-record or segment/page-level) — record-granular range
// reads argue for per-record; measure before deciding.
//
// Method: take the real wiki corpus segment (456,153 records, raw UTF-8
// JSON, corpus.bin + corpus-offsets.u32 — exactly section 3.5 layout v1's
// on-disk shape) and compare three schemes against the uncompressed
// baseline:
//   none        today's format: raw record bytes, one range read hydrates one record
//   per-record  each record gzip/brotli-compressed independently; one range
//               read still hydrates exactly one record, but small records
//               carry a fixed per-stream header/dictionary tax
//   paged       records grouped into fixed-size pages (mirrors the existing
//               layout-v2 page grouping used for digests, PAGE_SIZES below)
//               and each page compressed as one stream; hydrating one record
//               means fetching and decompressing its whole page
//
// Hydration cost is modeled against the real access pattern: the 200-query
// eval ground truth (eval-gt.json), whose ids cluster the way real corpus
// hydration does (article chunks are id-adjacent — same locality noted in
// the row-commitments measurement), plus a uniform-random control to
// separate corpus locality from luck.
//
//   node measure-corpus-compression.mjs
//
// Output: summary tables and corpus-compression-results.json.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(here, '..', '..', '..', 'examples', '04-static-wiki-pack', 'data-full');
const CORPUS_BIN = path.join(DATA, 'corpus.bin');
const OFFSETS_PATH = path.join(DATA, 'corpus-offsets.u32');
const EVAL_GT_PATH = path.join(DATA, 'eval-gt.json');

const PAGE_SIZES = [16, 32, 64, 128, 256, 512];
const CODECS = {
    gzip: { compress: (b) => zlib.gzipSync(b, { level: 9 }), decompress: (b) => zlib.gunzipSync(b) },
    brotli: {
        compress: (b) => zlib.brotliCompressSync(b, { params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 9 } }),
        decompress: (b) => zlib.brotliDecompressSync(b),
    },
};

function loadCorpus() {
    const offsBuf = fs.readFileSync(OFFSETS_PATH);
    const offsets = new Uint32Array(offsBuf.buffer, offsBuf.byteOffset, offsBuf.byteLength / 4);
    const bin = fs.readFileSync(CORPUS_BIN);
    return { offsets, bin, count: offsets.length - 1 };
}

function recordBytes(corpus, i) {
    return corpus.bin.subarray(corpus.offsets[i], corpus.offsets[i + 1]);
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
function quantile(sorted, q) {
    return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

// --- Scheme 1: no compression (today's format) ---
function measureBaseline(corpus) {
    return { totalBytes: corpus.bin.length, perRecord: null };
}

// --- Scheme 2: per-record compression ---
function measurePerRecord(corpus, codecName) {
    const codec = CODECS[codecName];
    let totalCompressed = 0;
    const sizes = new Array(corpus.count);
    let worseCount = 0;
    for (let i = 0; i < corpus.count; i++) {
        const raw = recordBytes(corpus, i);
        const compressed = codec.compress(raw);
        sizes[i] = compressed.length;
        totalCompressed += compressed.length;
        if (compressed.length >= raw.length) worseCount++;
    }
    return { totalCompressed, sizes, worseCount, worsePct: (100 * worseCount) / corpus.count };
}

// --- Scheme 3: paged compression ---
function measurePaged(corpus, codecName, pageRecords) {
    const codec = CODECS[codecName];
    const pages = Math.ceil(corpus.count / pageRecords);
    let totalCompressed = 0;
    const pageCompressedBytes = new Array(pages);
    const pageRawBytes = new Array(pages);
    for (let p = 0; p < pages; p++) {
        const start = p * pageRecords;
        const end = Math.min(corpus.count, start + pageRecords);
        const raw = corpus.bin.subarray(corpus.offsets[start], corpus.offsets[end]);
        const compressed = codec.compress(raw);
        pageCompressedBytes[p] = compressed.length;
        pageRawBytes[p] = raw.length;
        totalCompressed += compressed.length;
    }
    return { totalCompressed, pages, pageCompressedBytes, pageRawBytes };
}

// Hydration cost for paged: fetching record i means fetching+decompressing
// its whole page. Bytes-over-the-wire is the page's compressed size; extra
// bytes vs. the ideal (fetch just this record) is page bytes minus record
// bytes. Model against a real access pattern (id sets), not per-record in
// isolation, so cross-record reuse within a query (two ground-truth ids in
// the same page) is counted once, matching how a reader with a page cache
// would actually behave within one query.
function hydrationCost(corpus, idSets, pageRecords, pageCompressedBytes) {
    const perQueryWireBytes = [];
    const perQueryIdealBytes = [];
    const perQueryPagesTouched = [];
    for (const ids of idSets) {
        const touchedPages = new Set();
        let idealBytes = 0;
        for (const id of ids) {
            touchedPages.add(Math.floor(id / pageRecords));
            idealBytes += corpus.offsets[id + 1] - corpus.offsets[id];
        }
        let wireBytes = 0;
        for (const p of touchedPages) wireBytes += pageCompressedBytes[p];
        perQueryWireBytes.push(wireBytes);
        perQueryIdealBytes.push(idealBytes);
        perQueryPagesTouched.push(touchedPages.size);
    }
    return {
        meanWireBytes: mean(perQueryWireBytes),
        meanIdealBytes: mean(perQueryIdealBytes),
        meanPagesTouched: mean(perQueryPagesTouched),
        amplification: mean(perQueryWireBytes) / mean(perQueryIdealBytes),
    };
}

function makeUniformIdSets(idSets, corpusCount, seed) {
    let s = seed >>> 0;
    const rand = () => {
        s ^= (s << 13) >>> 0; s ^= s >>> 17; s ^= (s << 5) >>> 0;
        return (s >>> 0) / 0xffffffff;
    };
    return idSets.map((ids) => {
        const out = new Set();
        while (out.size < ids.length) out.add(Math.floor(rand() * corpusCount));
        return [...out];
    });
}

function main() {
    console.log('Loading corpus (456,153 records, raw UTF-8 JSON)...');
    const corpus = loadCorpus();
    const evalGt = JSON.parse(fs.readFileSync(EVAL_GT_PATH, 'utf8'));
    const realIdSets = evalGt;
    const uniformIdSets = makeUniformIdSets(realIdSets, corpus.count, 1234);

    const baseline = measureBaseline(corpus);
    console.log(`\nBaseline (uncompressed, today's format): ${(baseline.totalBytes / 1e6).toFixed(1)} MB\n`);

    const results = { corpusRecords: corpus.count, baselineBytes: baseline.totalBytes, perRecord: {}, paged: {} };

    console.log('=== Per-record compression ===');
    console.log('codec   | total bytes | ratio | records that GREW (%)');
    for (const codecName of Object.keys(CODECS)) {
        const r = measurePerRecord(corpus, codecName);
        const ratio = r.totalCompressed / baseline.totalBytes;
        console.log(`${codecName.padEnd(7)} | ${(r.totalCompressed / 1e6).toFixed(1).padStart(8)} MB | ${ratio.toFixed(3)} | ${r.worsePct.toFixed(1)}%`);
        results.perRecord[codecName] = { totalCompressed: r.totalCompressed, ratio, worsePct: r.worsePct };
    }

    console.log('\n=== Paged compression: corpus size ===');
    console.log('codec   | page  | total bytes | ratio vs baseline');
    for (const codecName of Object.keys(CODECS)) {
        results.paged[codecName] = {};
        for (const pageRecords of PAGE_SIZES) {
            const r = measurePaged(corpus, codecName, pageRecords);
            const ratio = r.totalCompressed / baseline.totalBytes;
            console.log(`${codecName.padEnd(7)} | ${String(pageRecords).padStart(5)} | ${(r.totalCompressed / 1e6).toFixed(1).padStart(8)} MB | ${ratio.toFixed(3)}`);
            results.paged[codecName][pageRecords] = { totalCompressed: r.totalCompressed, ratio, pages: r.pages, pageCompressedBytes: r.pageCompressedBytes };
        }
    }

    console.log('\n=== Hydration cost per query (mean over 200-query eval set) ===');
    console.log('workload  | codec   | page  | wire bytes | ideal bytes | amplification | pages touched');
    results.hydration = { real: {}, uniform: {} };
    for (const [workloadName, idSets] of [['real', realIdSets], ['uniform', uniformIdSets]]) {
        results.hydration[workloadName] = {};
        for (const codecName of Object.keys(CODECS)) {
            results.hydration[workloadName][codecName] = {};
            for (const pageRecords of PAGE_SIZES) {
                const pageCompressedBytes = results.paged[codecName][pageRecords].pageCompressedBytes;
                const h = hydrationCost(corpus, idSets, pageRecords, pageCompressedBytes);
                console.log(
                    `${workloadName.padEnd(9)} | ${codecName.padEnd(7)} | ${String(pageRecords).padStart(5)} | ` +
                    `${h.meanWireBytes.toFixed(0).padStart(10)} | ${h.meanIdealBytes.toFixed(0).padStart(11)} | ` +
                    `${h.amplification.toFixed(2).padStart(6)}x | ${h.meanPagesTouched.toFixed(1)}`
                );
                results.hydration[workloadName][codecName][pageRecords] = h;
            }
        }
    }

    // Per-record hydration cost is trivially ideal==wire (no page fetched):
    // report per-query wire bytes for direct comparison against the paged
    // table above.
    console.log('\n=== Per-record scheme: hydration bytes per query (no page amplification, by construction) ===');
    results.perRecordHydration = { real: {}, uniform: {} };
    for (const [workloadName, idSets] of [['real', realIdSets], ['uniform', uniformIdSets]]) {
        for (const codecName of Object.keys(CODECS)) {
            const codec = CODECS[codecName];
            const perQuery = idSets.map((ids) => ids.reduce((s, id) => s + codec.compress(recordBytes(corpus, id)).length, 0));
            const idealPerQuery = idSets.map((ids) => ids.reduce((s, id) => s + (corpus.offsets[id + 1] - corpus.offsets[id]), 0));
            const h = { meanWireBytes: mean(perQuery), meanIdealBytes: mean(idealPerQuery), ratio: mean(perQuery) / mean(idealPerQuery) };
            results.perRecordHydration[workloadName][codecName] = h;
            console.log(`${workloadName.padEnd(9)} | ${codecName.padEnd(7)} | per-record | wire=${h.meanWireBytes.toFixed(0)} ideal(raw)=${h.meanIdealBytes.toFixed(0)} ratio=${h.ratio.toFixed(3)}`);
        }
    }

    fs.writeFileSync(path.join(here, 'corpus-compression-results.json'), JSON.stringify(results, null, 2));
    console.log('\nWrote corpus-compression-results.json');
}

main();
