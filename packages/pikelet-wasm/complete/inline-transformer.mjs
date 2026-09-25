import { createWordPiece } from './wordpiece.mjs';

const decoder = new TextDecoder();

// The wasm kernel compiles these in (encoder-spike/encoder.cpp `constexpr`
// block) and walks the weight blob with a bare running cursor — it never
// reads the declaration. Declaration/blob skew is therefore only catchable
// here, host-side, before the first forward.
export const KERNEL_LAYOUT = { V: 30522, P: 512, T: 2, D: 384, F: 1536, L: 6, B: 64, H: 12 };
const KERNEL_MAX_SEQ = 512;
// Smallest window that still frames interior tokens between [CLS] and [SEP]
// with room to make progress; see the maxTokens check in
// parseInlineTransformerEncoder().
const MIN_DECLARED_MAX_TOKENS = 8;

// Byte size the kernel's fill_layout() will consume for a given layout:
// each quantized matrix is rows*cols u8 plus per-block f32 scales and
// offsets; biases and layer norms are plain f32.
export function expectedBlobBytes(layout = KERNEL_LAYOUT) {
  const { V, P, T, D, F, L, B } = layout;
  const quant = (rows, cols) => rows * cols + rows * (cols / B) * 4 * 2;
  const layer = 4 * quant(D, D) + 4 * (D * 4)   // wq,wk,wv,wo + their biases
    + 2 * (D * 4)                                // ln1 gamma/beta
    + quant(F, D) + F * 4                        // wu + bu
    + quant(D, F) + D * 4                        // wd + bd
    + 2 * (D * 4);                               // ln2 gamma/beta
  return quant(V, D) + quant(P, D) + quant(T, D) + 2 * (D * 4) + L * layer;
}

export function parseInlineTransformerEncoder(encoderBytes) {
  const view = new DataView(encoderBytes.buffer, encoderBytes.byteOffset, encoderBytes.byteLength);
  const declLen = view.getUint32(0, true);
  const vocabLen = view.getUint32(4, true);
  const blobLen = view.getUint32(8, true);
  if (12 + declLen + vocabLen + blobLen !== encoderBytes.length) {
    throw new Error('.pikelet inline-encoder layout is inconsistent');
  }
  const declaration = JSON.parse(decoder.decode(encoderBytes.subarray(12, 12 + declLen)));
  const blob = encoderBytes.subarray(12 + declLen + vocabLen);
  const layout = declaration.layout || {};
  for (const key of Object.keys(KERNEL_LAYOUT)) {
    if (layout[key] !== KERNEL_LAYOUT[key]) {
      throw new Error(`.pikelet inline-encoder declares layout ${key}=${layout[key]}; `
        + `the compiled kernel requires ${key}=${KERNEL_LAYOUT[key]}`);
    }
  }
  if (declaration.dim !== KERNEL_LAYOUT.D) {
    throw new Error(`.pikelet inline-encoder declares dim ${declaration.dim}; the kernel emits ${KERNEL_LAYOUT.D}`);
  }
  // maxTokens sizes the kernel's ids/hidden heap buffers, and the windowing
  // arithmetic (windowLen = maxSeq - 2) assumes it leaves room for [CLS] and
  // [SEP] plus interior tokens. A pack declaring 1 makes windowLen negative,
  // so interior.slice(w * windowLen, (w + 1) * windowLen) returns nearly the
  // whole sequence and the forward pass writes far past both allocations; 2
  // makes windowLen 0, so the window count is Infinity. Reject the whole
  // range here, at parse, so nothing is allocated from an unusable value.
  // The floor is well under any value the toolchain emits (the builder
  // declares 512; the BEIR encoder's own --max-tokens floor is 16).
  if (declaration.maxTokens !== undefined && declaration.maxTokens !== null) {
    const declared = declaration.maxTokens;
    if (!Number.isInteger(declared) || declared < MIN_DECLARED_MAX_TOKENS || declared > KERNEL_MAX_SEQ) {
      throw new Error(`.pikelet inline-encoder declares maxTokens ${declared}; `
        + `must be an integer in [${MIN_DECLARED_MAX_TOKENS}, ${KERNEL_MAX_SEQ}]`);
    }
  }
  const expected = expectedBlobBytes(layout);
  if (blob.length !== expected) {
    throw new Error(`.pikelet inline-encoder blob is ${blob.length} bytes but its declared layout implies ${expected}`);
  }
  return {
    declaration,
    vocabText: decoder.decode(encoderBytes.subarray(12 + declLen, 12 + declLen + vocabLen)),
    blob,
  };
}

export async function createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder, verify = true }) {
  const vocabLines = vocabText.split('\n');
  const vocabSize = vocabLines[vocabLines.length - 1] === '' ? vocabLines.length - 1 : vocabLines.length;
  if (vocabSize > KERNEL_LAYOUT.V) {
    throw new Error(`inline-encoder vocab has ${vocabSize} entries; the kernel's embedding table holds ${KERNEL_LAYOUT.V}`);
  }
  const tokenizer = createWordPiece(vocabText);
  const EM = await createEncoder();
  const dim = declaration.dim;
  // Clamped on BOTH sides: parseInlineTransformerEncoder() rejects an
  // out-of-range declaration, but this function is also called directly (the
  // BEIR ladder, encoder ablations), so the buffer sizing must not depend on
  // that check having run.
  const maxSeq = Math.min(
    Math.max(MIN_DECLARED_MAX_TOKENS, Math.trunc(declaration.maxTokens) || KERNEL_MAX_SEQ),
    KERNEL_MAX_SEQ,
  );
  const blobPtr = EM._malloc(blob.length);
  EM.HEAPU8.set(blob, blobPtr);
  const idsPtr = EM._malloc(maxSeq * 4);
  const hiddenPtr = EM._malloc(maxSeq * dim * 4);

  const usesClsPooling = declaration.pooling === 'cls';

  async function embed(text) {
    const allIds = tokenizer.encode(String(text || ''));
    // Inputs longer than maxSeq are windowed across [CLS]…[SEP]-framed
    // chunks instead of truncated, so long chunks keep their tail content.
    const interior = allIds.slice(1, -1);
    const windowLen = maxSeq - 2;
    const windows = Math.max(1, Math.ceil(interior.length / windowLen));
    const pooled = new Float32Array(dim);
    let pooledTokens = 0;
    // CLS pooling has no defined multi-window semantics (unlike mean
    // pooling, there's no natural way to combine multiple windows' [CLS]
    // tokens into one), so only the first window is encoded; long
    // documents lose tail content under this pooling mode — mean pooling
    // is the better choice for a corpus with chunks near or past maxSeq.
    const windowsToEncode = usesClsPooling ? 1 : windows;
    for (let w = 0; w < windowsToEncode; w++) {
      const tokenIds = [allIds[0], ...interior.slice(w * windowLen, (w + 1) * windowLen), allIds[allIds.length - 1]];
      // The windowing above must never produce more tokens than the buffers
      // were sized for; fail closed instead of writing past them.
      if (tokenIds.length > maxSeq) {
        throw new Error(`inline encoder window is ${tokenIds.length} tokens for a ${maxSeq}-token buffer`);
      }
      new Int32Array(EM.HEAP32.buffer, idsPtr, tokenIds.length).set(tokenIds);
      const rc = EM._encoder_forward(blobPtr, idsPtr, tokenIds.length, hiddenPtr, 0);
      if (rc !== tokenIds.length) throw new Error(`inline encoder failed: ${rc}`);
      const hidden = new Float32Array(EM.HEAPF32.buffer, hiddenPtr, tokenIds.length * dim);
      if (usesClsPooling) {
        for (let d = 0; d < dim; d++) pooled[d] = hidden[d];
        pooledTokens = 1;
        break;
      }
      for (let t = 0; t < tokenIds.length; t++) {
        for (let d = 0; d < dim; d++) pooled[d] += hidden[t * dim + d];
      }
      pooledTokens += tokenIds.length;
    }
    let norm = 0;
    for (let d = 0; d < dim; d++) {
      if (!usesClsPooling) pooled[d] /= pooledTokens;
      norm += pooled[d] ** 2;
    }
    norm = Math.sqrt(norm);
    for (let d = 0; d < dim; d++) pooled[d] /= norm;
    return { vector: pooled, text, windows };
  }

  // Per-word vectors from the same forward pass embed() already runs — the
  // kernel's outHidden buffer holds every token's final-layer hidden state
  // (encoder.cpp's encoder_forward does no pooling itself; JS pools it),
  // so this is not a second, slower embedding path: it is the same
  // forward pass, with the per-token buffer copied out and pooled per
  // WORD (via encodeWithWordMap's word alignment, merging WordPiece
  // sub-tokens like "quant"+"##ization" into one vector) instead of
  // discarded after sentence-level pooling. Used for MaxSim-style
  // token-level grounding (complete/retrieval-abstention.mjs, calibrate.
  // mjs), an alternative to coverage1's exact/stemmed word match that
  // measures semantic closeness instead of lexical identity — so a query
  // and passage that use different words for the same idea can still
  // ground each other, the case coverage1 structurally cannot handle.
  // Windowing/CLS-pooling limits mirror embed(): a word that falls past
  // the first window under CLS pooling, or past the last window under
  // mean pooling's cap, has no vector and is left out of the result.
  async function embedWords(text) {
    const { ids: allIds, wordOf: allWordOf, words } = tokenizer.encodeWithWordMap(String(text || ''));
    const interior = allIds.slice(1, -1);
    const interiorWordOf = allWordOf.slice(1, -1);
    const windowLen = maxSeq - 2;
    const windows = Math.max(1, Math.ceil(interior.length / windowLen));
    const windowsToEncode = usesClsPooling ? 1 : windows;
    // Accumulate per-word sums, then average — a word split into several
    // WordPiece pieces gets one vector, not several.
    const wordSums = words.map(() => new Float64Array(dim));
    const wordCounts = words.map(() => 0);
    for (let w = 0; w < windowsToEncode; w++) {
      const idSlice = interior.slice(w * windowLen, (w + 1) * windowLen);
      const wordOfSlice = interiorWordOf.slice(w * windowLen, (w + 1) * windowLen);
      const tokenIds = [allIds[0], ...idSlice, allIds[allIds.length - 1]];
      // The windowing above must never produce more tokens than the buffers
      // were sized for; fail closed instead of writing past them.
      if (tokenIds.length > maxSeq) {
        throw new Error(`inline encoder window is ${tokenIds.length} tokens for a ${maxSeq}-token buffer`);
      }
      new Int32Array(EM.HEAP32.buffer, idsPtr, tokenIds.length).set(tokenIds);
      const rc = EM._encoder_forward(blobPtr, idsPtr, tokenIds.length, hiddenPtr, 0);
      if (rc !== tokenIds.length) throw new Error(`inline encoder failed: ${rc}`);
      const hidden = new Float32Array(EM.HEAPF32.buffer, hiddenPtr, tokenIds.length * dim);
      // Offset by 1: tokenIds[0] is [CLS], so idSlice's tokens start at
      // hidden row 1, matching wordOfSlice's own indexing into idSlice.
      for (let t = 0; t < wordOfSlice.length; t++) {
        const wi = wordOfSlice[t];
        if (wi === -1) continue;
        const row = (t + 1) * dim;
        for (let d = 0; d < dim; d++) wordSums[wi][d] += hidden[row + d];
        wordCounts[wi]++;
      }
    }
    const vectors = new Map();
    for (let wi = 0; wi < words.length; wi++) {
      if (wordCounts[wi] === 0) continue;
      const v = new Float32Array(dim);
      let norm = 0;
      for (let d = 0; d < dim; d++) { v[d] = wordSums[wi][d] / wordCounts[wi]; norm += v[d] ** 2; }
      norm = Math.sqrt(norm) || 1;
      for (let d = 0; d < dim; d++) v[d] /= norm;
      vectors.set(words[wi], v);
    }
    return vectors;
  }

  const embedder = {
    declaration,
    dim,
    maxSeq,
    embed,
    embedWords,
    dispose() {
      EM._free(hiddenPtr);
      EM._free(idsPtr);
      EM._free(blobPtr);
    },
  };
  if (verify && Array.isArray(declaration.testVectors) && declaration.testVectors.length > 0) {
    try {
      await verifyInlineTestVectors(embedder);
    } catch (err) {
      embedder.dispose();
      throw err;
    }
  }
  return embedder;
}

// Conformance probes for kind-3 declarations (contract section 4.4 mode 1:
// an inline encoder carries verification vectors). The last probe exceeds
// the kernel window so the windowed mean-pool path stays pinned alongside
// the single-window path.
export const INLINE_TEST_VECTOR_TEXTS = [
  'how do volcanoes form and why do they erupt',
  'which treaty ended the first world war',
  'The water cycle describes how water moves between the oceans, the atmosphere, and the land. '
    + 'Heat from the sun evaporates water from the surface of the sea, and plants release more through their leaves. '
    + 'The rising vapor cools as it climbs, condensing into droplets that gather into clouds. '
    + 'When the droplets grow too heavy they fall as rain, snow, or hail, depending on the temperature of the air they pass through. '
    + 'Some of the water soaks into the soil and filters down to recharge underground aquifers, '
    + 'while the rest runs off into streams and rivers that carry it back toward the coast. '
    + 'Along the way it erodes rock, deposits sediment, and sustains wetlands, forests, and farmland. '
    + 'Glaciers and ice sheets hold a share of it frozen for centuries before releasing meltwater in spring. '
    + 'Eventually nearly every drop returns to the ocean, where the sun lifts it again and the whole journey repeats.',
];

export async function buildInlineTestVectors(embedder, texts = INLINE_TEST_VECTOR_TEXTS) {
  const out = [];
  for (const text of texts) {
    const { vector, windows } = await embedder.embed(text);
    out.push({
      text,
      windows,
      embedding: Array.from(vector, (v) => Number(v.toFixed(6))),
      tolerance: 1e-3,
    });
  }
  return out;
}

// A verification vector's expected embedding must itself be verifiable:
// exactly dim real, finite numbers. A malformed entry (string, null, NaN,
// missing component, wrong length) would otherwise make every comparison
// NaN, and `NaN > maxDiff` is false — the vector would "pass" without a
// single meaningful comparison. Shared by the kind-2 host verifier and the
// kind-3 inline verifier.
export function validateExpectedEmbedding(values, dim, label) {
  if (!Array.isArray(values) || values.length !== dim) {
    throw new Error(`${label}: expected embedding must be an array of ${dim} numbers, got ${Array.isArray(values) ? `length ${values.length}` : typeof values}`);
  }
  for (let d = 0; d < dim; d++) {
    if (typeof values[d] !== 'number' || !Number.isFinite(values[d])) {
      throw new Error(`${label}: expected embedding component ${d} is not a finite number`);
    }
  }
}

export function validateTestVectorTolerance(tolerance, label) {
  if (tolerance === undefined) return 1e-3;
  if (typeof tolerance !== 'number' || !Number.isFinite(tolerance) || tolerance <= 0 || tolerance >= 1) {
    throw new Error(`${label}: tolerance is implausible`);
  }
  return tolerance;
}

export async function verifyInlineTestVectors(embedder) {
  for (const tv of embedder.declaration.testVectors) {
    if (!tv || typeof tv.text !== 'string') {
      throw new Error('inline encoder verification vector is malformed (no text)');
    }
    const label = `inline encoder verification vector "${tv.text.slice(0, 40)}…"`;
    const { vector, windows } = await embedder.embed(tv.text);
    validateExpectedEmbedding(tv.embedding, vector.length, label);
    const tolerance = validateTestVectorTolerance(tv.tolerance, label);
    if (tv.windows !== undefined && windows !== tv.windows) {
      throw new Error(`inline encoder verification: "${tv.text.slice(0, 40)}…" pooled over ${windows} window(s); `
        + `the declaration recorded ${tv.windows}`);
    }
    let maxDiff = 0;
    for (let d = 0; d < vector.length; d++) {
      const diff = Math.abs(vector[d] - tv.embedding[d]);
      if (diff > maxDiff) maxDiff = diff;
    }
    if (!(maxDiff <= tolerance)) {
      throw new Error(`inline encoder failed its declaration's verification vector "${tv.text.slice(0, 40)}…": `
        + `max component diff ${maxDiff.toExponential(2)} exceeds tolerance ${tolerance}`);
    }
  }
  return { checked: embedder.declaration.testVectors.length };
}
