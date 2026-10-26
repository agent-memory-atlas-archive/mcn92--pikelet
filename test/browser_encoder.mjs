// The inline-encoder kernel's browser build (encoder.mjs / encoder.wasm) in
// Chromium, Firefox and WebKit, against the Node build (encoder.node.wasm):
//
//   - packs compiled before the integer kernel still verify: the f32
//     kernel's test vectors (test/fixtures/minilm-f32-kernel-test-vectors.json)
//     pass verifyInlineTestVectors at their 1e-3 tolerance in every engine
//   - every engine embeds bit-identically to Node, at full width and with a
//     narrow maxTokens that forces multi-window pooling, so a pack's
//     embeddings never depend on where it was opened
//
// Nothing else ran the encoder in a browser; test:browser covers the engine.
// Under CI every engine is required; locally an engine that cannot launch is
// skipped, as in test/browser_smoke.js.

import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const { chromium, firefox, webkit } = require('playwright');
const { createInlineTransformerEmbedder } = await import('../packages/pikelet-wasm/complete/inline-transformer.mjs');
const { default: createNodeEncoder } = await import('../packages/pikelet-wasm/complete/encoder-kernels/encoder.node.mjs');

const TEXTS = [
  'how do volcanoes form and why do they erupt',
  'which treaty ended the first world war',
  'What is ownership in Rust?',
  'Hydraulic fracturing: risks, benefits & regulation (2019–2024)',
  'café, naïve, Zürich — résumé of the coöperative’s façade',
  'a',
  'The water cycle describes how water moves between the oceans, the atmosphere, and the land. '.repeat(12),
];
const DECLARATION = {
  kind: 'inline-transformer-v1',
  model: 'sentence-transformers/all-MiniLM-L6-v2',
  dim: 384,
  pooling: 'mean',
  normalized: true,
  maxTokens: 512,
  longInputs: 'windowed-mean-pool',
  prefixPolicy: { passage: '', query: '' },
  layout: { V: 30522, P: 512, T: 2, D: 384, F: 1536, L: 6, B: 64, H: 12 },
};
const NARROW = { ...DECLARATION, maxTokens: 32 };
const ENCODER_DIR = '/packages/pikelet-wasm/complete/';
const ASSET_DIR = '/packages/pikelet/src/inline-encoder/';
const TYPES = { '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.txt': 'text/plain', '.bin': 'application/octet-stream' };

const vocabText = readFileSync(path.join(ROOT, ASSET_DIR, 'vocab.txt'), 'utf8');
const blob = readFileSync(path.join(ROOT, ASSET_DIR, 'encoder-weights.bin'));
const { testVectors: f32Vectors } = JSON.parse(readFileSync(path.join(ROOT, 'test/fixtures/minilm-f32-kernel-test-vectors.json'), 'utf8'));

async function embedAll(createEncoder, declaration) {
  const e = await createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder, verify: false });
  const out = [];
  for (const t of TEXTS) out.push(Array.from((await e.embed(t)).vector));
  e.dispose();
  return out;
}
const reference = { wide: await embedAll(createNodeEncoder, DECLARATION), narrow: await embedAll(createNodeEncoder, NARROW) };

const server = http.createServer(async (req, res) => {
  const p = new URL(req.url, 'http://x').pathname;
  if (p === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>encoder</title>'); return; }
  if (p.includes('..') || !(p.startsWith(ENCODER_DIR) || p.startsWith(ASSET_DIR))) { res.writeHead(404); res.end(); return; }
  try {
    const body = await readFile(path.join(ROOT, p));
    res.writeHead(200, { 'content-type': TYPES[path.extname(p)] || 'application/octet-stream' });
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;

let failed = 0;
const skipped = [];
for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  let browser;
  try {
    browser = await engine.launch({ headless: true });
  } catch (err) {
    if (process.env.CI) { console.log(`  ${name}: FAIL cannot launch (${err.message.split('\n')[0]})`); failed++; continue; }
    console.log(`  ${name}: skipped (${err.message.split('\n')[0]})`);
    skipped.push(name);
    continue;
  }
  try {
    const page = await browser.newPage();
    await page.goto(origin);
    const got = await page.evaluate(async ({ texts, declaration, narrow, f32Vectors, encoderDir, assetDir }) => {
      const { createInlineTransformerEmbedder, verifyInlineTestVectors } = await import(`${encoderDir}inline-transformer.mjs`);
      const { default: createEncoder } = await import(`${encoderDir}encoder-kernels/encoder.mjs`);
      const vocabText = await (await fetch(`${assetDir}vocab.txt`)).text();
      const blob = new Uint8Array(await (await fetch(`${assetDir}encoder-weights.bin`)).arrayBuffer());
      const old = await createInlineTransformerEmbedder({ declaration: { ...declaration, testVectors: f32Vectors }, vocabText, blob, createEncoder, verify: false });
      await verifyInlineTestVectors(old);  // throws past the 1e-3 tolerance
      old.dispose();
      const out = {};
      for (const [key, decl] of [['wide', declaration], ['narrow', narrow]]) {
        const e = await createInlineTransformerEmbedder({ declaration: decl, vocabText, blob, createEncoder, verify: false });
        out[key] = [];
        for (const t of texts) out[key].push(Array.from((await e.embed(t)).vector));
        e.dispose();
      }
      return out;
    }, { texts: TEXTS, declaration: DECLARATION, narrow: NARROW, f32Vectors, encoderDir: ENCODER_DIR, assetDir: ASSET_DIR });
    const mismatches = [];
    for (const key of ['wide', 'narrow']) {
      got[key].forEach((v, i) => {
        if (v.length !== reference[key][i].length || v.some((x, k) => x !== reference[key][i][k])) mismatches.push(`${key} "${TEXTS[i].slice(0, 24)}"`);
      });
    }
    if (mismatches.length) {
      console.log(`  ${name}: FAIL not bit-identical to Node for ${mismatches.join(', ')}`);
      failed++;
    } else {
      console.log(`  ${name}: passed (f32-kernel test vectors verify; ${TEXTS.length * 2} embeddings bit-identical to Node)`);
    }
  } catch (err) {
    console.log(`  ${name}: FAIL ${err.message.split('\n')[0]}`);
    failed++;
  } finally {
    await browser.close();
  }
}
server.close();
if (failed) {
  console.log(`Browser encoder test failed (${failed} engine(s))`);
  process.exit(1);
}
console.log(skipped.length ? `Browser encoder test passed (skipped ${skipped.join(', ')})` : 'Browser encoder test passed (all 3 engines)');
