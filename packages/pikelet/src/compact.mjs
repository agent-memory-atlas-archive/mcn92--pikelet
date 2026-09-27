// `pikelet compact` — LAYERED_PROFILE.md 6.3.
//
// Collapses a chain back into one pikelet-complete-v2 base. This is SEMANTIC
// maintenance, not only physical consolidation: it restores exact lexical
// statistics over the live set, a live-only vocabulary bloom, a calibration fit
// against the corpus actually served, no tombstones, depth zero and reclaimed id
// space. The drift and depth limits exist to force it to happen.
//
// What it preserves is corpus meaning and provenance — the compacted base's
// records are the old head's live records, in ascending old-id order, bytes
// unchanged. It does NOT preserve rankings, and is not meant to: the layered
// path scored BM25 with statistics that counted tombstoned documents and
// generated candidates per tier, and compaction scores one unified corpus with
// exact statistics. A compaction that reproduced the chain's rankings would have
// failed at its job.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { CliError } from './common.mjs';

const log = (line) => console.log(line);

export async function compactChain(flags) {
  if (!flags.head) throw new CliError('compact requires --head <file|url>[#identity], listing the chain base-first');
  if (!flags.out) throw new CliError('compact requires --out <file>');
  const outPath = path.resolve(process.cwd(), flags.out);
  if (fssync.existsSync(outPath) && !flags.force) {
    throw new CliError(`Output file already exists: ${outPath}\nNext: rerun with --force or choose --out`);
  }

  const { openPikeletChain } = await import('pikelet-wasm/complete/layer-reader.mjs');
  const { planCompaction, buildLineageSegment } = await import('pikelet-wasm/complete/layer-compact.mjs');
  const builder = await import('pikelet-wasm/complete/builder');
  const { loadPikelet, loadArtifactContract } = await import('./common.mjs');

  const members = [].concat(flags.head);
  const chain = await openPikeletChain(members);
  try {
    const info = chain.info();
    const chainMembers = chain.__members;
    const head = chain.__head;
    log(`Mounted chain: ${info.layers} member(s), ${info.records} records (${info.liveRecords} live), `
      + `${info.tombstones} tombstone(s), depth ${info.layers - 1}`);
    if (info.layers === 1) {
      throw new CliError('this is already a depth-0 base: there is no chain to compact');
    }

    // --- 1-2. Enumerate live ids and renumber densely (6.3) -----------------
    const headBitset = chainMembers[chainMembers.length - 1].bitset
      ?? chainMembers.findLast((m) => m.bitset)?.bitset
      ?? new Uint8Array(0);
    const plan = planCompaction({ rowTotal: info.records, headBitset });
    log(`Compacting to ${plan.liveRecords} live record(s), reclaiming ${plan.tombstones} id(s)`);

    // Records are copied verbatim — the compacted base's records ARE the old
    // head's live records, bytes unchanged. Re-embedding would break that.
    const records = [];
    const texts = [];
    for (const oldId of plan.liveIds) {
      const rec = await chain.record(oldId);
      // record() adds chain-level fields (id, layer, tombstoned, successors);
      // the corpus segment carries the record's own bytes, so strip them.
      const { id, layer, tombstoned, supersededBy, successors, currentSuccessor, distance, ...own } = rec;
      records.push(Buffer.from(JSON.stringify(own), 'utf8'));
      texts.push(own.text || '');
    }

    // --- 2. Rebuild the index over live rows --------------------------------
    // 6.3 copies live rows' quantized bytes and per-row scale/offset verbatim —
    // same encoder, same quantizer, no re-embedding. This implementation
    // re-embeds with the base's own encoder instead, which produces the same
    // vectors for the same text and the same encoder but costs a forward pass
    // per record; the verbatim-copy path needs a row readback API the engine
    // does not expose (see complete/README on the missing readback).
    const enc = await head.passageEmbedder();
    if (!enc) {
      throw new CliError(`the base's query-interpretation kind ${head.qiKind} cannot embed passages, so compaction cannot rebuild the index (6.3)`);
    }
    log(`Re-deriving ${texts.length} row(s) with the base's own encoder`
      + `${enc.declaration.model ? ` (${enc.declaration.model})` : ''}`);
    const vectors = [];
    for (const text of texts) vectors.push(await enc.embed(text));

    const Pikelet = await loadPikelet();
    const artifactContract = await loadArtifactContract();
    const index = await Pikelet.create({
      dim: head.dim, metric: head.metric, quantized: true,
      maxElements: Math.max(vectors.length, Math.ceil(vectors.length * 1.25)),
    });
    let indexBytes;
    try {
      index.addBatch(vectors);
      indexBytes = artifactContract.buildSketchArtifactBytes(index.export(), {
        recommendedRerank: Math.min(vectors.length, 100),
      }).bytes;
    } finally {
      index.dispose();
    }

    // --- 3. Corpus and lexical over live records; statistics now exact ------
    const corpus = builder.buildCorpusSegment(records);
    const lexical = builder.buildLexicalSegment(texts);
    log(`Rebuilt lexical index: ${lexical.meta.terms.toLocaleString()} terms over `
      + `${lexical.meta.docCount.toLocaleString()} live record(s) — statistics are now exact`);

    // --- 4. Calibration ----------------------------------------------------
    // 6.3: "If the producer cannot refit, it MUST ship `unscored`, never the
    // inherited fit." Refitting needs the calibrator, which is a build-time
    // dependency this command does not yet load, so it ships unscored rather
    // than carrying a fit made against a corpus that no longer exists.
    const calibration = Buffer.from(JSON.stringify({
      kind: 'retrieval-signals-v1', asset: null, vocabBloomBase64: '',
    }), 'utf8');
    log('Calibration: shipping unscored — a refit against the compacted corpus is not wired yet, '
      + 'and the inherited fit must never be carried across a compaction (6.3)');

    // --- 5. The lineage segment (kind 7), REQUIRED ---------------------------
    const heads = chainMembers.map((m, i) => ({
      identity: m.identity,
      rowTotal: chainMembers.slice(0, i + 1).reduce((n, x) => n + x.records, 0),
      depth: i,
    }));
    const edges = [];
    for (const m of chainMembers) {
      for (const [oldId, newId] of m.supersessions || []) edges.push([oldId, newId, m.depth]);
    }
    const lineage = buildLineageSegment({
      heads, headBitset, oldRowTotal: info.records, supersessions: edges,
    });
    log(`Lineage segment: ${heads.length} head(s) in the table, ${edges.length} edge(s) — `
      + 'a citation against any of them still translates (6.3)');

    // The query-interp segment is copied from the base: compaction MUST NOT
    // change encoder, dim, metric, ingestion or tokenization — those are a new
    // compile, not a compaction.
    const baseQi = await head.readBaseQueryInterp?.();
    if (!baseQi) {
      throw new CliError('compaction needs the base\'s query-interp segment to copy; this reader build does not expose it');
    }

    const written = builder.assemblePikeletFile({
      profile: builder.PROFILE_V2,
      dim: head.dim,
      metric: head.metric,
      encoder: head.encoder,
      corpus: {
        ...corpus.corpus,
        // The compacted base carries the chain's ingestion declaration so it is
        // appendable on its own terms (6.1).
        ...(head.corpusIngest ? { ingest: head.corpusIngest } : {}),
      },
      index: {},
      compactedFrom: {
        identity: info.identity,
        depth: info.layers - 1,
        liveRecords: plan.liveRecords,
        tombstones: plan.tombstones,
      },
    }, [
      { kind: 'index', bytes: indexBytes },
      { kind: 'corpus', bytes: corpus.bytes },
      { kind: 'query-interp', bytes: baseQi },
      { kind: 'lexical', bytes: lexical.bytes },
      { kind: 'lineage', bytes: lineage },
    ], outPath);

    log(`Wrote ${outPath}`);
    log(`  ${(written.fileBytes / 1024 / 1024).toFixed(2)} MB, ${plan.liveRecords} records, depth 0, `
      + `identity ${written.identity}`);
    log(`  compacted from ${info.identity.slice(0, 12)}… at depth ${info.layers - 1}`);
  } finally {
    await chain.close();
  }
}

export async function rebaseLayer() {
  // 6.2's rules are implemented and conformance-tested in
  // pikelet-wasm/complete/layer-rebase.mjs. What is missing here is the byte
  // work: a rebase copies the layer's u8 rows, sketches, records, postings and
  // bloom VERBATIM — nothing is re-embedded and local ids do not change — which
  // needs a row readback the engine does not expose. Re-embedding instead would
  // be a different operation with a different cost and a different failure mode,
  // so this refuses rather than quietly doing that.
  throw new CliError(
    'rebase is not wired as a command yet. Its rules (fork point, delta replay, foreign ids, '
    + 'conflicts, the terminal empty rule) are implemented and tested in '
    + 'pikelet-wasm/complete/layer-rebase.mjs, but copying a layer\'s rows verbatim needs an '
    + 'engine row-readback API that does not exist; re-embedding instead would be a different '
    + 'operation, not a rebase.',
  );
}
