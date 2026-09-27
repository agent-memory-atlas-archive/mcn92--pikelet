// The chain reader — LAYERED_PROFILE.md 5.1, 5.2, 5.3, 5.6.
//
// openPikeletChain() mounts a base plus zero or more layers and serves them as
// one corpus. It does NOT reimplement the container: every member is opened
// through openPikeletFile(), which is decision 1 — "a reader that implements the
// complete profile implements this one by adding the chain logic of section 5,
// not a second container parser."
//
// What this file adds on top of those opens:
//   - chain resolution and ancestry verification (5.1 steps 1-2, 3.2, 3.3)
//   - the interval table, per-tier masks and chain state (5.1 step 4)
//   - a query() that generates candidates per search tier with that tier's
//     mask supplied to the scan, merges by distance, and fuses once (5.3)
//   - record()/citation resolution over global ids (5.6)
//
// The caller supplies the head and, for now, the ancestors explicitly. Locator
// following through layer-locator.mjs is wired but a caller may also pass a
// lineage map (section 7), which is the cheaper path: every layer's identity
// and location is known up front and layers open in one parallel wave.

import { openPikeletFile, readChainMemberShell, base64Bytes } from './index.mjs';
import { fuseCandidates, FUSION_DEFAULTS } from './fusion.mjs';
import { parseTombstoneSegment, firstSupersetViolation } from './tombstones.mjs';
import { validateAgainstParent, chainIngestDeclaration, validateIngestAgainstChain, LAYER_PROFILE, MAX_DEPTH } from './layer-manifest.mjs';
import { parseInheritedQuerySegment, unionBloom } from './layer-encoder.mjs';
import {
    buildIntervalTable, ownerOf, projectMasks, liveCount, buildSupersessionMap,
    resolveSupersession, calibrationDrift, calibrationStatus, globalLexicalStats,
    buildAncestry, resolveCitation,
} from './layer-chain.mjs';

// 5.9's budgets, provisional: "chosen for round-trip count and plausibility,
// not measured" (13.5). Named here so a host can lower them and so a refusal
// can name the budget it hit.
export const CHAIN_DEFAULTS = Object.freeze({
    maxDepth: MAX_DEPTH,
    // 5.5: the reader's limit MUST be finite and MUST be declared. 13.1
    // measured the inherited fit out to 40% drift with no degradation, so 0.20
    // is conservative rather than a guess — but it stays a reader policy, not a
    // format constant, so it can be revisited without a format change.
    readerDriftLimit: 0.20,
});

/**
 * @param {Array<string|object>} members  head-last or base-first list of
 *   locations/sources. Base first is the natural order and what this accepts:
 *   [base, layer1, layer2, ...]. Every member is opened through
 *   openPikeletFile(), so a string is a path and an object is a range source.
 * @param {object} [options] forwarded to each member's open, plus:
 *   readerDriftLimit, maxDepth.
 */
export async function openPikeletChain(members, options = {}) {
    if (!Array.isArray(members) || members.length === 0) {
        throw new Error('openPikeletChain() needs at least a base');
    }
    const maxDepth = options.maxDepth ?? CHAIN_DEFAULTS.maxDepth;
    if (members.length - 1 > maxDepth) {
        throw new Error(`chain depth ${members.length - 1} exceeds the limit ${maxDepth} (2.7)`);
    }
    const readerDriftLimit = options.readerDriftLimit ?? CHAIN_DEFAULTS.readerDriftLimit;

    const opened = [];
    const closeAll = async () => {
        for (const r of opened) { try { await r.close(); } catch { /* best effort */ } }
    };

    try {
        // --- 5.1 steps 1-2: open every member, verify the chain links ---
        // asChainMember lets openPikeletFile accept a `pikelet-layer-v1`
        // manifest. Only this function passes it, and only because it goes on
        // to open every ancestor and verify the hash chain — a layer opened
        // alone still fails, which is what 4.1 requires.
        //
        // A tombstone-only layer has no index, corpus or lexical segment, so it
        // is read as a "shell": manifest plus its two segments, verified the
        // same way, with none of the corpus/sketch machinery a search tier
        // needs. It is a chain member but not a search tier (3.1).
        const tiersRaw = [];
        for (let i = 0; i < members.length; i++) {
            const shell = i > 0 ? await readChainMemberShell(members[i], options) : null;
            if (shell && shell.tombstoneOnly) {
                opened.push(shell);
                tiersRaw.push(shell.tier);
                continue;
            }
            if (shell) await shell.close();
            const reader = await openPikeletFile(members[i], { ...options, asChainMember: i > 0 });
            opened.push(reader);
            tiersRaw.push(reader.__chainTier);
        }
        const base = tiersRaw[0];
        if (base.manifest.profile === LAYER_PROFILE) {
            throw new Error('the first chain member is a layer, not a base: a layer opened alone would serve a fraction of a corpus (4.1)');
        }

        const chainMembers = [{
            depth: 0, identity: base.identity, rowBase: 0, records: base.recordCount,
        }];
        // Ingestion declaration: defined by the base, or asserted by depth 1.
        const depth1Manifest = tiersRaw.length > 1 ? tiersRaw[1].manifest : null;
        const ingestDecl = tiersRaw.length > 1 ? chainIngestDeclaration(base.manifest, depth1Manifest) : null;

        const perLayerTombstones = [];
        let headBitset = new Uint8Array(0);
        const layerBlooms = [];
        let everyLayerShipsBloom = true;

        for (let i = 1; i < tiersRaw.length; i++) {
            const tier = tiersRaw[i];
            const parent = tiersRaw[i - 1];
            // 3.2/3.3/4.1: verified against the parent ACTUALLY OPENED.
            const rel = validateAgainstParent(tier.manifest, {
                manifest: parent.manifest, identity: parent.identity, isBase: i === 1,
            });
            validateIngestAgainstChain(tier.manifest, ingestDecl);

            // 4.5: the encoder is resolved from the base, and the layer's
            // commitment must match the base's query-interp digest.
            const baseQiSha = base.manifest.segments.find((sg) => sg.kind === 'query-interp')?.sha256;
            if (!baseQiSha) throw new Error('the base carries no query-interp segment digest to inherit');
            const qiSeg = tier.segments.get('query-interp');
            if (!qiSeg) throw new Error(`layer at depth ${rel.depth} carries no query-interp segment`);
            const inherited = parseInheritedQuerySegment(
                await readSegment(opened[i], qiSeg),
                {
                    layerBaseIdentity: rel.identityFields.baseIdentity,
                    baseQueryInterpSha256: baseQiSha,
                    baseBloom: options.baseBloom || null,
                },
            );
            if (inherited.bloomBytes) layerBlooms.push(inherited.bloomBytes);
            else everyLayerShipsBloom = false;

            // 4.4: the tombstone segment is eager and digest-verified by the
            // member's own open; here it is parsed and structurally validated.
            const tombSeg = tier.segments.get('tombstones');
            if (!tombSeg) throw new Error(`layer at depth ${rel.depth} carries no tombstones segment (4.3)`);
            const tomb = parseTombstoneSegment(await readSegment(opened[i], tombSeg), {
                rowBase: rel.rowBase, tombstones: rel.tombstones,
                supersessions: rel.supersessions, records: rel.records,
            });
            // 3.4: the cumulative superset rule, against the parent's mask
            // zero-extended over the ids the parent did not yet address.
            const violation = firstSupersetViolation(tomb.bitset, headBitset, rel.rowBase);
            if (violation !== -1) {
                throw new Error(`layer at depth ${rel.depth} clears inherited tombstone for id ${violation}: a child's mask must be a superset of its parent's (3.4)`);
            }
            headBitset = tomb.bitset;
            perLayerTombstones.push({ depth: rel.depth, supersessions: tomb.supersessions });
            chainMembers.push({
                depth: rel.depth, identity: tier.identity,
                rowBase: rel.rowBase, records: rel.records,
                // Retained per member, not just the head's: `rebase` needs a
                // layer's OWN bitset and its parent's to compute the delta it
                // replays, and 6.2 refuses rather than guessing without them.
                bitset: tomb.bitset,
                supersessions: tomb.supersessions,
            });
        }

        // --- 5.1 step 4: chain state ---
        const table = buildIntervalTable(chainMembers);
        const byIdentity = new Map(tiersRaw.map((tr) => [tr.identity, tr]));
        const searchTiers = table.tiers.map((t) => ({ ...t, reader: byIdentity.get(t.identity) }));
        for (const t of searchTiers) {
            if (!t.reader || !t.reader.sketch) {
                throw new Error(`search tier at depth ${t.depth} has no index: a tier that owns ids must carry one (4.3)`);
            }
        }
        const masks = projectMasks(table, headBitset);
        // Per-tier popcount, so a query can tell an empty mask (skip it, keep
        // the WASM scan) from one that actually excludes something.
        const maskCounts = masks.map((m) => countBits(m));
        const supersessions = buildSupersessionMap(perLayerTombstones);
        const ancestry = buildAncestry(chainMembers);
        const headTombstoneCount = countBits(headBitset);
        const drift = chainMembers.length > 1
            ? calibrationDrift(chainMembers, headTombstoneCount) : 0;
        // 5.5: "the base carries no fit" is status `none`. The fit lives in the
        // base's query-interp calibration region (`asset`), never in the
        // manifest; an earlier draft of this file read manifest.calibration,
        // which does not exist.
        const baseCal = base.calibrationJson || null;
        const baseHasFit = !!(baseCal && baseCal.asset);
        const calibration = calibrationStatus({
            drift,
            // producerEnvelope is the base producer's declared validity
            // envelope. No claim is no constraint (+Infinity), which
            // calibrationStatus() handles; the reader's finite limit still
            // gates it.
            producerEnvelope: baseCal?.asset?.driftLimit,
            readerLimit: readerDriftLimit,
            baseHasFit,
            everyLayerShipsBloom: chainMembers.length === 1 ? true : everyLayerShipsBloom,
        });
        const lexStats = globalLexicalStats(searchTiers.map((t) => ({
            docCount: t.reader.lexicalIndex ? t.reader.lexicalIndex.docCount : 0,
            totalTokens: t.reader.lexicalIndex ? t.reader.lexicalIndex.totalTokens : 0,
        })));
        // The union bloom (4.5) starts from the base's own vocabulary bloom,
        // which ships in its calibration region beside the fit.
        const baseBloomBytes = baseCal?.vocabBloomBase64
            ? base64Bytes(baseCal.vocabBloomBase64)
            : null;
        const chainBloom = unionBloom([baseBloomBytes, ...layerBlooms]);

        let closed = false;
        const assertOpen = () => { if (closed) throw new Error('chain is closed'); };

        return {
            info() {
                assertOpen();
                return {
                    profile: LAYER_PROFILE,
                    identity: chainMembers[chainMembers.length - 1].identity,
                    baseIdentity: base.identity,
                    layers: chainMembers.length,
                    searchTiers: searchTiers.length,
                    records: table.rowTotal,
                    liveRecords: liveCount(table, headTombstoneCount),
                    tombstones: headTombstoneCount,
                    supersessions: supersessions.size,
                    dim: base.dim,
                    metric: base.manifest.metric,
                    calibrationDrift: calibration.drift,
                    calibrationStatus: calibration.status,
                    driftLimit: calibration.effectiveLimit,
                    producerEnvelope: calibration.producerEnvelope,
                    readerLimit: calibration.readerLimit,
                    lexical: { ...lexStats },
                    // 4.1: sampleQueries may appear on any member and applies
                    // to that member. A chain surfaces the base's, which are
                    // the ones fit against the corpus the encoder saw.
                    sampleQueries: Array.isArray(base.manifest.sampleQueries) ? base.manifest.sampleQueries : [],
                    members: chainMembers.map((m) => ({ depth: m.depth, identity: m.identity, records: m.records })),
                };
            },

            /**
             * 5.3. One encode, per-tier masked candidate generation, merge by
             * distance, fuse once. Equal distances break by ascending global id
             * — the reference reader's declared tie rule for chains.
             */
            async query(vector, k = 5, queryOptions = {}) {
                assertOpen();
                const retrieval = queryOptions.retrieval ?? 'hybrid';
                const perTier = await Promise.all(searchTiers.map(async (t, j) => {
                    // An all-zero mask is equivalent to no mask, and passing one
                    // is expensive: a mask disables the WASM scan kernel (it
                    // cannot honour an exclusion set), forcing a JS scan over
                    // every row of the tier. On a 456k-record base with three
                    // tombstones that was a 9x slowdown to skip three rows, so
                    // the mask is omitted entirely when the tier has no live
                    // deletion. maskCounts is computed once at mount.
                    const out = await t.reader.sketch.search(vector, Math.min(k * 4, t.records), {
                        rerank: queryOptions.rerank,
                        ...(maskCounts[j] > 0 ? { exclude: masks[j] } : {}),
                        // The tier's own staged scan kernel. Read per query, not
                        // captured at mount: staging finishes in the background
                        // after the member's open returns. With a mask present
                        // the sketch reader over-fetches and filters, so the
                        // kernel stays usable (5.3's no-post-truncation rule is
                        // kept by construction, not by skipping the kernel).
                        ...(t.reader.scanner ? { scanner: t.reader.scanner } : {}),
                    });
                    return out.results.map((r) => ({
                        id: t.rowBase + r.id, localId: r.id, tier: j,
                        depth: t.depth, distance: r.distance,
                    }));
                }));
                const searched = perTier.flat()
                    .sort((a, b) => (a.distance - b.distance) || (a.id - b.id));

                let lexicalIds = [];
                if (retrieval !== 'vector' && queryOptions.text) {
                    const merged = [];
                    for (let j = 0; j < searchTiers.length; j++) {
                        const lex = searchTiers[j].reader.lexicalIndex;
                        if (!lex) continue;
                        // openLexicalIndex's search is sync; the LAZY opener
                        // used for a large segment (the wiki pack's is 69 MB)
                        // returns a promise. Awaiting covers both.
                        const hits = await lex.search(queryOptions.text, 24);
                        for (const h of hits) {
                            // A tombstoned row must not occupy a lexical slot
                            // before the cap (5.3), so skip as postings score.
                            if (bitAt(masks[j], h.id)) continue;
                            merged.push({ id: searchTiers[j].rowBase + h.id, score: h.score });
                        }
                    }
                    merged.sort((a, b) => (b.score - a.score) || (a.id - b.id));
                    lexicalIds = merged.map((h) => h.id);
                }

                const fused = retrieval === 'vector' || !lexicalIds.length
                    ? searched
                    : fuseCandidates(searched, lexicalIds, queryOptions.fusion);
                const top = fused.slice(0, k);
                const results = await Promise.all(top.map(async (hit) => {
                    // ownerOf searches the interval table, whose entries carry
                    // no reader; searchTiers[] is the parallel array that does.
                    const owner = ownerOf(table, hit.id);
                    if (!owner) throw new Error(`fused result id ${hit.id} owns no search tier (5.2)`);
                    const tier = searchTiers[owner.tier.tier];
                    const record = await tier.reader.hydrate(owner.localId);
                    return { ...record, id: hit.id, distance: hit.distance, layer: tier.depth };
                }));
                // 5.5: under `inherited` the base's scorer runs unchanged over
                // the merged window. Under `drift-exceeded` matchQuality is
                // `unscored`, confidence is omitted, results still ship, and the
                // response says which input bound it — so a consumer can tell
                // "no fit" from "fit outgrown". A reader MUST NOT apply an
                // outgrown fit silently, and MUST NOT omit the verdict either:
                // omitting it makes "the fit said no" indistinguishable from
                // "nothing scored", which is exactly the confusion 5.5 exists
                // to prevent.
                let matchQuality = 'unscored';
                let confidence;
                if (calibration.status === 'inherited' && base.scoreQuality) {
                    const passages = await Promise.all(top.slice(0, 3).map(async (hit) => {
                        const owner = ownerOf(table, hit.id);
                        const rec = await searchTiers[owner.tier.tier].reader.hydrate(owner.localId);
                        return rec?.text ?? '';
                    }));
                    const scored = await base.scoreQuality(searched, { text: queryOptions.text ?? '', vector, passages }, fused);
                    if (scored && scored.match_quality) {
                        matchQuality = scored.match_quality;
                        if (Number.isFinite(scored.confidence)) confidence = scored.confidence;
                    }
                } else if (calibration.status === 'none') {
                    matchQuality = 'unscored';
                }
                // A 'none' verdict withholds results, exactly as the
                // single-file reader does: the calibrated abstention signal is
                // doing its job, and a chain must not be a way to get results
                // the base would have refused. showAbstained is the same
                // explicit opt-out, and never changes matchQuality.
                const withheld = matchQuality === 'none' && !queryOptions.showAbstained;
                return {
                    matchQuality,
                    ...(confidence === undefined ? {} : { confidence }),
                    results: withheld ? [] : results,
                    identity: chainMembers[chainMembers.length - 1].identity,
                    calibration: {
                        status: calibration.status,
                        drift: calibration.drift,
                        effectiveLimit: calibration.effectiveLimit,
                        producerEnvelope: calibration.producerEnvelope,
                        readerLimit: calibration.readerLimit,
                    },
                };
            },

            /** 5.6: a bare id is an id in the mounted head. */
            async record(id) {
                assertOpen();
                const owner = ownerOf(table, id);
                if (!owner) throw new Error(`id ${id} is outside every search tier's interval (5.2)`);
                const tier = searchTiers[owner.tier.tier];
                const record = await tier.reader.hydrate(owner.localId);
                const tombstoned = bitAt(headBitset, id);
                const sup = resolveSupersession(supersessions, id);
                return {
                    ...record, id, layer: tier.depth, tombstoned,
                    supersededBy: sup.immediate, successors: sup.all, currentSuccessor: sup.current,
                };
            },

            /** 5.6: a citation is (identity, id), resolved along the mounted ancestry. */
            citation(identity, id) {
                assertOpen();
                return resolveCitation(ancestry, identity, id);
            },

            chainBloom() { assertOpen(); return chainBloom; },

            /**
             * The chain's members with each layer's own bitset and edges, which
             * is what `rebase` replays (6.2) and what `compact` reads to
             * enumerate live ids and rebuild the lineage segment (6.3).
             */
            get __members() {
                assertOpen();
                return chainMembers.map((m) => ({ ...m }));
            },

            /**
             * Everything `append` needs about the head to plan a new layer
             * (6.1). Internal, like __chainTier: a producer is part of this
             * package, and the names carry no compatibility promise.
             */
            get __head() {
                assertOpen();
                const last = chainMembers[chainMembers.length - 1];
                const baseMember = chainMembers[0];
                return {
                    identity: last.identity,
                    depth: last.depth,
                    // The head's own rowBase and record count, which together
                    // give the next layer's rowBase (3.3).
                    rowBase: last.depth === 0 ? 0 : last.rowBase,
                    records: last.records,
                    bitset: headBitset,
                    baseIdentity: base.identity,
                    baseRecords: baseMember.records,
                    // Sum of every layer's records, for the drift numerator.
                    appendedBefore: chainMembers.slice(1).reduce((n, m) => n + m.records, 0),
                    // 6.1 needs the base's kind to decide appendability, its
                    // ingestion declaration to enforce granularity, and its
                    // chunking so new records are cut the same way. The kind
                    // comes from the segment header verbatim, not from
                    // encoderInfo: kind 2's encoderInfo is shaped by the host
                    // declaration and carries no fixed name to match on.
                    qiKind: base.qiKind ?? null,
                    // The base's own encoder, resolved by the base's open.
                    passageEmbedder: () => base.passageEmbedder(),
                    // The base's query-interp segment bytes. `compact` copies
                    // them verbatim: compaction MUST NOT change the encoder,
                    // dim, metric, ingestion or tokenization (6.3), and the
                    // surest way not to is to carry the same bytes.
                    // The base's bloom geometry, which every layer's bloom
                    // MUST match so the chain's union is an exact bitwise OR
                    // (4.5). Null when the base carries no fit.
                    baseBloomGeometry: baseCal?.asset?.vocabBloom
                        ? {
                            bits: baseCal.asset.vocabBloom.bits,
                            hashes: baseCal.asset.vocabBloom.hashes,
                            minCount: baseCal.asset.vocabBloom.minCount ?? null,
                        }
                        : null,
                    readBaseQueryInterp: () => {
                        const seg = base.segments.get('query-interp');
                        if (!seg) return null;
                        return base.readSegmentBytes(seg);
                    },
                    corpusIngest: base.manifest.corpus?.ingest ?? null,
                    chunking: base.manifest.corpus?.ingest?.chunking
                        ?? base.manifest.corpus?.ingest
                        ?? null,
                    baseQueryInterpSha256: base.manifest.segments.find((sg) => sg.kind === 'query-interp')?.sha256 ?? null,
                    dim: base.dim,
                    metric: base.manifest.metric,
                    encoder: base.manifest.encoder,
                };
            },

            async close() {
                if (closed) return;
                closed = true;
                await closeAll();
            },
        };
    } catch (err) {
        await closeAll();
        throw err;
    }
}

function countBits(bytes) {
    let n = 0;
    for (let i = 0; i < bytes.length; i++) {
        let x = bytes[i];
        while (x) { n += x & 1; x >>= 1; }
    }
    return n;
}

function bitAt(bitset, id) {
    const byte = id >> 3;
    if (!bitset || byte >= bitset.length) return false;
    return ((bitset[byte] >> (id & 7)) & 1) === 1;
}

/**
 * Read one segment's bytes through the member's own reader. The member's open
 * already verified the segment digest where the profile requires it eagerly;
 * this is the bytes for structural parsing.
 */
async function readSegment(member, seg) {
    const tier = member.__chainTier || member.tier;
    if (tier && typeof tier.readSegmentBytes === 'function') return tier.readSegmentBytes(seg);
    throw new Error('this reader build does not expose segment bytes for chain parsing');
}

export { FUSION_DEFAULTS };
