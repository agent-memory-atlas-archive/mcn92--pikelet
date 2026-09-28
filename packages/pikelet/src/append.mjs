// `pikelet append` — LAYERED_PROFILE.md 6.1.
//
// Compiles one layer on top of a mounted chain. The planning and assembly rules
// live in pikelet-wasm/complete/layer-append.mjs (and are conformance-tested
// there); this module is the operator-facing wiring: resolve the chain, load the
// base's own encoder, ingest and embed the new records, then write the file.
//
// The encoder comes from the BASE, not from this CLI's packaged weights. That is
// the point of 6.1's "a pack is sufficient to compile its own successors": a
// kind-3 base carries its teacher in its own query-interp segment, so appending
// needs no model download and cannot silently substitute a different encoder.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { CliError, loadCompleteModules, DEFAULT_CONFIG } from './common.mjs';
import { ingestFolder, ingestUrl, chunkDocs, dedupeChunks } from './ingest.mjs';

const log = (line) => console.log(line);

/**
 * @param {object} flags parsed CLI flags
 */
export async function appendLayer(flags) {
  if (!flags.parent) throw new CliError('append requires --parent <file|url>[#identity]');
  if (!flags.out) throw new CliError('append requires --out <file>');
  const outPath = path.resolve(process.cwd(), flags.out);
  if (fssync.existsSync(outPath) && !flags.force) {
    throw new CliError(`Output file already exists: ${outPath}\nNext: rerun with --force or choose --out`);
  }

  const removeIds = toIdList(flags.remove, '--remove');
  const supersedePairs = toSupersedeList(flags.supersede);
  const sources = flags.source ? [].concat(flags.source) : [];
  if (!sources.length && !removeIds.length && !supersedePairs.length) {
    // 6.1's "a layer must do something", caught before any work: the planner
    // enforces it too, but failing here costs no mount and no embedding.
    throw new CliError('append needs at least one of --source, --remove or --supersede: a layer that introduces neither a record nor a tombstone is refused (6.1)');
  }

  const { reader } = await loadCompleteModules();
  const {
    planAppendability, planLayer, buildLayerSegments, checkAppendDrift,
  } = await import('pikelet-wasm/complete/layer-append.mjs');
  const { openPikeletChain } = await import('pikelet-wasm/complete/layer-reader.mjs');

  // --- 1. Mount the parent chain (6.1 step 1) ---------------------------------
  // The parent may itself be a chain; the caller lists its members base-first,
  // which is also what --lineage would supply. A bare --parent is a base.
  const members = [].concat(flags.parent);
  const chain = await openPikeletChain(members, {
    ...(flags['reader-drift-limit'] ? { readerDriftLimit: Number(flags['reader-drift-limit']) } : {}),
  });
  try {
    const info = chain.info();
    log(`Mounted chain: ${info.layers} member(s), ${info.records} records (${info.liveRecords} live), depth ${info.layers - 1}`);

    const head = chain.__head;
    if (!head) {
      throw new CliError('this reader build does not expose the chain head needed to append');
    }

    // --- 2. Appendability and the ingestion declaration (6.1) ----------------
    const assertIngest = flags['assert-ingest']
      ? JSON.parse(await fs.readFile(path.resolve(process.cwd(), flags['assert-ingest']), 'utf8'))
      : null;
    const plan0 = planAppendability({
      qiKind: head.qiKind,
      corpusIngest: head.corpusIngest,
      assertIngest,
    });
    log(`Base is appendable: encoder from the ${plan0.encoderSource === 'pack' ? 'pack itself' : 'host'}`
      + `${plan0.ingestAsserted ? ', ingestion declaration asserted by the operator' : ''}`);

    // --- 3. Ingest and chunk under the chain's declaration (6.1 step 2) ------
    let chunks = [];
    if (sources.length) {
      const docs = [];
      for (const src of sources) {
        const isUrl = /^https?:\/\//i.test(src);
        if (isUrl) {
          docs.push(...await ingestUrl({ type: 'url', url: src, maxPages: Number(flags['max-pages'] || 500) }, log));
        } else {
          docs.push(...await ingestFolder(path.resolve(process.cwd(), src), {
            include: flags.include || ['**/*.{md,mdx,html,txt}'],
            exclude: flags.exclude || ['**/node_modules/**', '**/.git/**'],
          }, log));
        }
      }
      // Records are NOT deduplicated against the chain (decision 9): a producer
      // that wants "append only what is new" compares candidate digests against
      // the chain's live recordSha256 values as its own policy.
      // Chunk under the CHAIN's declaration, not this CLI's defaults: record
      // granularity is part of what the calibration saw and must not change
      // mid-chain (6.1). plan0.ingest is the chain's declaration — the base's
      // own when it carries one, the operator's assertion otherwise.
      const chunking = {
        targetTokens: plan0.ingest.targetTokens ?? DEFAULT_CONFIG.chunking.targetTokens,
        overlapPercent: plan0.ingest.overlapPercent ?? DEFAULT_CONFIG.chunking.overlapPercent,
      };
      chunks = dedupeChunks(chunkDocs(docs, chunking));
      log(`Ingested ${docs.length} doc(s) -> ${chunks.length} chunk(s)`);
      if (!chunks.length && !removeIds.length && !supersedePairs.length) {
        throw new CliError('the sources produced no chunks, and no --remove or --supersede was given: nothing to append (6.1)');
      }
    }

    // Resolve each --supersede path to the local index of the chunk it
    // produced. The path was parsed and then thrown away, so supersession
    // edges were assigned by the order the pairs happened to appear rather
    // than by which file replaced which record. A path that ingested to
    // exactly one chunk resolves to it; anything else is refused rather than
    // guessed, because a silently wrong edge points readers at the wrong
    // successor.
    const resolvedSupersede = supersedePairs.map(([oldId, target]) => {
      if (!sources.length) {
        throw new CliError(`--supersede ${oldId}=${target} needs a --source: the path names which new record replaces id ${oldId}`);
      }
      const wanted = path.resolve(process.cwd(), target);
      const matches = [];
      for (let i = 0; i < chunks.length; i++) {
        const sp = chunks[i].sourcePath;
        if (!sp) continue;
        if (sp === target || path.resolve(process.cwd(), sp) === wanted) matches.push(i);
      }
      if (!matches.length) {
        throw new CliError(`--supersede ${oldId}=${target} matched no ingested record; `
          + `the path must be one the --source ingested (${chunks.length} chunk(s) produced)`);
      }
      if (matches.length > 1) {
        throw new CliError(`--supersede ${oldId}=${target} matched ${matches.length} records `
          + `(local ids ${matches.join(', ')}); a supersession edge names exactly one successor, `
          + 'so split the source or supersede by resolved id');
      }
      return [oldId, matches[0]];
    });

    // --- 4-5. Plan ids, mask and supersessions (6.1 steps 4-5) --------------
    const plan = planLayer({
      parentRowBase: head.rowBase,
      parentRecords: head.records,
      parentBitset: head.bitset,
      parentDepth: head.depth,
      newRecordCount: chunks.length,
      remove: removeIds,
      supersede: resolvedSupersede,
      ...(flags['max-depth'] ? { maxDepth: Number(flags['max-depth']) } : {}),
    });

    // Drift: append MUST refuse past the effective limit unless --allow-drift.
    const drift = checkAppendDrift({
      baseRecords: head.baseRecords,
      appendedBefore: head.appendedBefore,
      newRecords: plan.records,
      headTombstones: plan.tombstones,
      effectiveLimit: info.driftLimit,
      allowDrift: flags['allow-drift'] === true,
    });

    log(`Planned layer: depth ${plan.depth}, rowBase ${plan.rowBase}, +${plan.records} record(s), `
      + `${plan.delta.length} new tombstone(s), ${plan.supersessions.length} supersession(s)`);
    log(`  calibrationDrift would be ${drift.drift.toFixed(4)} against limit ${info.driftLimit}`
      + `${drift.exceedsLimit ? ' — EXCEEDED, the chain will serve unscored' : ''}`);

    // --- 6. Embed with the BASE's encoder (6.1 step 3) ----------------------
    // Not this CLI's packaged weights: the base's own, resolved by the base's
    // open from its own query-interp segment. That is what makes "a pack is
    // sufficient to compile its own successors" true, and what makes
    // substituting a different encoder impossible.
    let vectors = [];
    if (chunks.length) {
      const enc = await head.passageEmbedder();
      if (!enc) {
        throw new CliError(`the base's query-interpretation kind ${head.qiKind} cannot embed passages; only a kind-3 base carries its teacher (6.1)`);
      }
      log(`Embedding ${chunks.length} chunk(s) with the base's own encoder`
        + `${enc.declaration.model ? ` (${enc.declaration.model})` : ''}`);
      for (const chunk of chunks) vectors.push(await enc.embed(chunk.text || ''));
      if (vectors[0].length !== head.dim) {
        throw new CliError(`the base's encoder produced ${vectors[0].length} dims but the chain declares ${head.dim}`);
      }
    }

    // --- 7. Build this layer's segments (6.1 step 4) ------------------------
    const { loadPikelet, loadArtifactContract } = await import('./common.mjs');
    const Pikelet = await loadPikelet();
    const artifactContract = await loadArtifactContract();
    const builder = await import('pikelet-wasm/complete/builder');

    let indexBytes = null;
    let corpusMeta = null;
    let corpusBytes = null;
    let lexicalBytes = null;
    if (chunks.length) {
      // The layer's own index over its own rows, with the base's geometry by
      // default (4.6 allows a layer its own; the default keeps a future
      // resident index over the union simple).
      const index = await Pikelet.create({
        dim: head.dim, metric: head.metric, quantized: true,
        maxElements: Math.max(chunks.length, Math.ceil(chunks.length * 1.25)),
      });
      try {
        // addBatch takes a plain array of vectors; ids are assigned by
        // position, which is what the layer's local numbering already is.
        index.addBatch(vectors);
        const snapshot = index.export();
        indexBytes = artifactContract.buildSketchArtifactBytes(snapshot, {
          // recommendedRerank applies to THIS layer's index only (4.1); a
          // small layer is cheapest to rerank exhaustively.
          recommendedRerank: chunks.length,
        }).bytes;
      } finally {
        index.dispose();
      }
      const { publicChunk } = await import('./ingest.mjs');
      const built = builder.buildCorpusSegment(
        chunks.map((c) => Buffer.from(JSON.stringify(publicChunk(c)), 'utf8')),
      );
      corpusBytes = built.bytes;
      corpusMeta = built.corpus;
      // The base's tokenizer, via the same builder the base used.
      lexicalBytes = builder.buildLexicalSegment(chunks.map((c) => c.text || '')).bytes;
    }

    // The layer's own vocabulary bloom, with the BASE's geometry (4.5, 6.1
    // step 4). Without it the chain degrades to unscored: 4.5 says a layer
    // shipping `{"kind":"none"}` while the base carries a fit takes the whole
    // chain to drift-exceeded, and 5.3 says the union is what keeps the
    // known-token signal from under-reading every query about a layer's
    // content.
    let layerBloom = null;
    if (chunks.length && head.baseBloomGeometry) {
      const { buildLayerVocabBloom, BLOOM_HASH_NAMES } = await import('./calibrate.mjs');
      const geom = head.baseBloomGeometry;
      const built = buildLayerVocabBloom(chunks, { bits: geom.bits, minCount: geom.minCount ?? 1 });
      layerBloom = {
        geometry: { bits: geom.bits, hashes: geom.hashes ?? BLOOM_HASH_NAMES },
        bytes: built.bloom,
      };
      log(`Built the layer's vocabulary bloom: ${built.keptWords} term(s) at the base's geometry `
        + `(${geom.bits} bits)`);
    } else if (chunks.length) {
      log('The base carries no vocabulary bloom, so this layer ships none (the chain stays unscored)');
    }

    const { buildLayerSegments } = await import('pikelet-wasm/complete/layer-append.mjs');
    const assembled = buildLayerSegments({
      layerBloom,
      plan,
      baseIdentity: head.baseIdentity,
      parentIdentity: head.identity,
      baseQueryInterpSha256: head.baseQueryInterpSha256,
      parentLocator: flags['parent-locator'] || null,
      ingest: plan0.ingest,
      ingestAsserted: plan0.ingestAsserted,
      dim: head.dim,
      metric: head.metric,
      encoder: head.encoder,
      corpus: corpusMeta || { records: 0 },
    });

    // --- 8. Write the file (6.1 step 6) ------------------------------------
    const segments = [];
    if (indexBytes) segments.push({ kind: 'index', bytes: indexBytes });
    if (corpusBytes) segments.push({ kind: 'corpus', bytes: corpusBytes });
    segments.push({ kind: 'query-interp', bytes: assembled.queryInterp });
    if (lexicalBytes) segments.push({ kind: 'lexical', bytes: lexicalBytes });
    segments.push({ kind: 'tombstones', bytes: assembled.tombstones });

    await fs.mkdir(path.dirname(outPath), { recursive: true });
    const written = builder.assemblePikeletFile(assembled.manifestFields, segments, outPath);
    log(`Wrote ${outPath}`);
    log(`  ${(written.fileBytes / 1024).toFixed(1)} KiB, depth ${plan.depth}, +${plan.records} record(s), `
      + `identity ${written.identity}`);
    log(`  calibrationDrift ${drift.drift.toFixed(4)}`
      + `${drift.exceedsLimit ? ' (drift-exceeded: the chain serves unscored)' : ''}`);
    log(`Mount the chain: openPikeletChain([base, ..., '${path.basename(outPath)}'])`);
  } finally {
    await chain.close();
  }
}

function toIdList(value, flagName) {
  if (value === undefined) return [];
  return [].concat(value).map((raw) => {
    const n = Number.parseInt(String(raw), 10);
    if (!Number.isSafeInteger(n) || n < 0) throw new CliError(`${flagName} takes non-negative record ids, got ${raw}`);
    return n;
  });
}

function toSupersedeList(value) {
  if (value === undefined) return [];
  return [].concat(value).map((raw) => {
    const at = String(raw).indexOf('=');
    if (at < 1) throw new CliError(`--supersede takes <oldId>=<sourcePath>, got ${raw}`);
    const oldId = Number.parseInt(String(raw).slice(0, at), 10);
    if (!Number.isSafeInteger(oldId) || oldId < 0) throw new CliError(`--supersede oldId must be a non-negative id, got ${raw}`);
    return [oldId, String(raw).slice(at + 1)];
  });
}
