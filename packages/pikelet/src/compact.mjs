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
  const { loadArtifactContract } = await import('./common.mjs');

  const { resolveChainMembers } = await import('./common.mjs');
  const members = await resolveChainMembers([].concat(flags.head), log);
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
    // The live records' own fields, kept for a refit (6.3 step 4), which needs
    // titles as well as text to generate its verified positives.
    const liveChunks = [];
    for (const oldId of plan.liveIds) {
      const rec = await chain.record(oldId);
      // record() adds chain-level fields (id, layer, tombstoned, successors);
      // the corpus segment carries the record's own bytes, so strip them.
      const { id, layer, tombstoned, supersededBy, successors, currentSuccessor, distance, ...own } = rec;
      records.push(Buffer.from(JSON.stringify(own), 'utf8'));
      texts.push(own.text || '');
      liveChunks.push(own);
    }

    // --- 2. Rebuild the index over live rows --------------------------------
    // 6.3: copy each live row's quantized bytes and its per-row scale/offset
    // verbatim. Same encoder, same quantizer, no re-embedding — the compacted
    // base's vectors ARE the head's live vectors, exactly as its records are.
    //
    // This re-embedded every live record with the base's own encoder instead,
    // on the stated grounds that the engine exposed no row readback. It does:
    // the sketch reader's fetchRows() (format 2 verifies each row's digest as
    // it reads) with the resident scales/offsets, reached through the chain's
    // __fetchRows seam. Re-embedding cost a forward pass per record and only
    // reproduced the same bytes if the encoder was bit-identical, which is a
    // stronger assumption than a copy needs to make.
    const artifactContract = await loadArtifactContract();
    log(`Copying ${plan.liveIds.length} row(s) verbatim from the head's tiers`);
    const fetched = await chain.__fetchRows(plan.liveIds);
    const dim = head.dim;
    const qdata = new Uint8Array(plan.liveRecords * dim);
    const scales = new Float32Array(plan.liveRecords);
    const offsets = new Float32Array(plan.liveRecords);
    for (let newId = 0; newId < fetched.length; newId++) {
      const { row, scale, offset } = fetched[newId];
      if (row.length !== dim) {
        throw new CliError(`row for old id ${plan.liveIds[newId]} is ${row.length} bytes, expected ${dim}`);
      }
      qdata.set(row, newId * dim);
      scales[newId] = scale;
      offsets[newId] = offset;
    }
    const indexBytes = artifactContract.exportSketchArtifact(
      { dim, count: plan.liveRecords, metric: head.metricCode, qdata, scales, offsets },
      null,
      { recommendedRerank: Math.min(plan.liveRecords, 100) },
    ).bytes;

    // --- 3. Corpus and lexical over live records; statistics now exact ------
    const corpus = builder.buildCorpusSegment(records);
    const lexical = builder.buildLexicalSegment(texts);
    log(`Rebuilt lexical index: ${lexical.meta.terms.toLocaleString()} terms over `
      + `${lexical.meta.docCount.toLocaleString()} live record(s) — statistics are now exact`);

    // --- 4. Calibration ----------------------------------------------------
    // 6.3 step 4: "Refit calibration with the chain's encoder ... If the
    // producer cannot refit, it MUST ship `unscored`, never the inherited fit."
    // The refit is the normative path and `unscored` is the fallback, so this
    // refits when it can. It is opt-OUT rather than opt-in (--no-refit) because
    // shipping unscored silently turns every query on the compacted pack into
    // an unverdicted one.
    //
    // The cost is real and worth naming: the verbatim row copy above removed
    // the per-record forward pass, and a refit puts one back, because the
    // calibrator fits against float vectors and generated queries rather than
    // the quantized rows on disk. That is a build-time cost paid once per
    // compaction, not a serve-time one.
    let calibration = Buffer.from(JSON.stringify({
      kind: 'retrieval-signals-v1', asset: null, vocabBloomBase64: '',
    }), 'utf8');
    if (flags['no-refit']) {
      log('Calibration: shipping unscored at --no-refit; the compacted pack reports '
        + 'match_quality "unscored" for every query (6.3)');
    } else {
      const enc = await head.passageEmbedder();
      if (!enc) {
        log(`Calibration: shipping unscored — the base's query-interpretation kind ${head.qiKind} `
          + 'carries no encoder to refit with, and the inherited fit must never be carried '
          + 'across a compaction (6.3)');
      } else {
        const { loadPikelet } = await import('./common.mjs');
        const Pikelet = await loadPikelet();
        const { calibrateRetrievalAbstention } = await import('./calibrate.mjs');
        const { openLexicalIndex } = await import('pikelet-wasm/complete');
        log(`Refitting calibration over ${liveChunks.length} live record(s) with the base's own encoder`
          + `${enc.declaration.model ? ` (${enc.declaration.model})` : ''}`);
        // Float vectors for the fit. The index above copies quantized rows; a
        // fit needs the pre-quantization vectors, so these are embedded here
        // and thrown away with this process.
        const fitVectors = [];
        for (const text of texts) fitVectors.push(await enc.embed(text));
        const refit = await calibrateRetrievalAbstention({
          Pikelet,
          chunks: liveChunks,
          vectors: fitVectors,
          // The calibrator builds a scratch index for its probes, so it needs
          // the metric as the string form create() takes (head.metricCode is
          // the numeric one the artifact header carries) and the same
          // quantization the base used.
          config: {
            embedding: { dims: head.dim },
            index: { metric: head.metric, quantized: true },
          },
          embedQuery: enc.embedQuery,
          embedWordVecs: enc.embedWords,
          lexicalIndex: openLexicalIndex(lexical.bytes),
          log,
        });
        if (refit) {
          calibration = Buffer.from(JSON.stringify(refit.calibrationJson), 'utf8');
          const s = refit.summary || {};
          log(`Calibration: refit over the compacted corpus — 5-fold CV AUC ${s.cvAuc ?? 'n/a'} `
            + `(hard ${s.cvAucHard ?? 'n/a'}), bloom over live records only`);
        } else {
          log('Calibration: shipping unscored — the refit declined (the calibrator reports why '
            + 'above), and the inherited fit must never be carried across a compaction (6.3)');
        }
      }
    }

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

    // The ENCODER half of the query-interp segment is copied from the base:
    // compaction MUST NOT change encoder, dim, metric, ingestion or
    // tokenization — those are a new compile, not a compaction. The
    // CALIBRATION half is replaced, because 6.3 forbids carrying the
    // inherited fit across a compaction: the fit was made against a corpus
    // that no longer exists.
    //
    // This used to copy the whole segment, so the base's fit shipped verbatim
    // while the log said "shipping unscored". A compacted pack scored `strong`
    // with a confidence, from a fit whose corpus had changed underneath it —
    // exactly what step 4's MUST exists to prevent.
    const baseQi = await head.readBaseQueryInterp?.();
    if (!baseQi) {
      throw new CliError('compaction needs the base\'s query-interp segment to copy; this reader build does not expose it');
    }
    const qiView = new DataView(baseQi.buffer, baseQi.byteOffset, baseQi.byteLength);
    const qiVersion = qiView.getUint32(0, true);
    const qiKindRead = qiView.getUint32(4, true);
    const qiEncoderLen = qiView.getUint32(8, true);
    const qiCalibrationLen = qiView.getUint32(12, true);
    if (16 + qiEncoderLen + qiCalibrationLen !== baseQi.length) {
      throw new CliError(`the base's query-interp segment declares ${qiEncoderLen}+${qiCalibrationLen} bytes `
        + `in a ${baseQi.length}-byte segment; refusing to rebuild it`);
    }
    const encoderBytes = baseQi.subarray(16, 16 + qiEncoderLen);
    const queryInterp = builder.buildQueryInterpSegment(qiKindRead, encoderBytes, calibration);
    if (qiVersion !== 1) {
      throw new CliError(`the base's query-interp segment is version ${qiVersion}; this build writes version 1`);
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
      { kind: 'query-interp', bytes: queryInterp },
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
  // pikelet-wasm/complete/layer-rebase.mjs: fork point, delta replay, foreign
  // ids, the conflict policies and the terminal empty rule.
  //
  // The row readback this used to blame is no longer missing — `compact` copies
  // rows verbatim through the chain's __fetchRows seam, and a rebase would use
  // the same one. What is actually missing is the command: planRebase needs the
  // layer's ORIGINAL PARENT bitset, so rebase must mount three things (the old
  // chain the layer was cut against, the new head chain, and the layer itself)
  // and then apply the fork-point and conflict policy across them. None of that
  // has a flag surface yet — `rebase` is absent from printHelp, so --layer,
  // --onto, --on-foreign and --on-conflict would be designed here rather than
  // connected. That is a command to design, not a call to wire, and doing it
  // badly is worse than refusing.
  throw new CliError(
    'rebase is not wired as a command yet. Its rules (fork point, delta replay, foreign ids, '
    + 'conflicts, the terminal empty rule) are implemented and tested in '
    + 'pikelet-wasm/complete/layer-rebase.mjs, and the verbatim row copy it needs now exists '
    + '(the same readback `compact` uses). What is missing is the command around them: a rebase '
    + 'mounts the layer\'s original parent as well as the new head, and that flag surface is '
    + 'not designed yet.',
  );
}
