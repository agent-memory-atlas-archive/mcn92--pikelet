// Layered-profile conformance — LAYERED_PROFILE.md section 10.
//
// This suite grows with the implementation. Currently covered: the tombstone
// segment codec (4.4), the cumulative/superset rule and delta (3.4), and the
// supersession constraints (3.5) — including the hostile cases family (B) of
// section 10 that apply to this segment, since a publisher can sign anything
// and a digest proves only authorship.
//
// Run standalone: node test/layered_profile.mjs

import {
    buildTombstoneSegment, parseTombstoneSegment, popcountBytes, bitsetBytesFor,
    testBit, zeroExtend, firstSupersetViolation, tombstoneDelta,
    TOMBSTONE_HEADER_BYTES, MAX_RECORDS,
} from '../packages/pikelet-wasm/complete/tombstones.mjs';
import {
    validateLayerObject, validateAgainstParent, canonicalJson,
    chainIngestDeclaration, validateIngestAgainstChain, rowTotal,
    LAYER_PROFILE, MAX_DEPTH,
} from '../packages/pikelet-wasm/complete/layer-manifest.mjs';
import { KINDS, KIND_NAMES } from '../packages/pikelet-wasm/complete/format.mjs';
import {
    buildInheritedQuerySegment, parseInheritedQuerySegment, unionBloom,
    QI_HEADER_BYTES, QI_KIND_INHERITED, LAYER_VOCAB_KIND,
} from '../packages/pikelet-wasm/complete/layer-encoder.mjs';
import {
    validateLocatorShape, resolveLocatorUrl, assertConfined,
    resolveLocatorPath, resolveParentLocation,
} from '../packages/pikelet-wasm/complete/layer-locator.mjs';
import {
    buildIntervalTable, ownerOf, projectMasks, liveCount,
    buildSupersessionMap, resolveSupersession, calibrationDrift,
    calibrationStatus, globalLexicalStats, idfFor,
    buildAncestry, resolveCitation,
} from '../packages/pikelet-wasm/complete/layer-chain.mjs';

let passed = 0;
let failed = 0;
function check(label, cond, detail = '') {
    if (cond) { passed++; console.log(`  ok: ${label}`); }
    else { failed++; console.log(`  FAIL: ${label}${detail ? ` — ${detail}` : ''}`); }
}
function rejects(label, fn, pattern) {
    try {
        fn();
        check(label, false, 'returned instead of throwing');
    } catch (err) {
        const msg = String(err && err.message);
        const ok = pattern.test(msg);
        check(label, ok, ok ? '' : `threw: ${msg.slice(0, 140)}`);
    }
}

console.log('tombstones-v1: round trip');
{
    const seg = buildTombstoneSegment({ rowBase: 100, tombstonedIds: [0, 7, 8, 63, 99], records: 5 });
    const t = parseTombstoneSegment(seg, { rowBase: 100, tombstones: 5, supersessions: 0, records: 5 });
    check('rowBase survives', t.rowBase === 100);
    check('popcount matches the ids set', t.tombstoneCount === 5);
    check('every id reads back set', [0, 7, 8, 63, 99].every((id) => testBit(t.bitset, id)));
    check('an unset id reads clear', !testBit(t.bitset, 1) && !testBit(t.bitset, 98));
    check('bitset is ceil(rowBase/8) bytes', t.bitset.length === bitsetBytesFor(100) && t.bitset.length === 13);
    check('segment tiles exactly', seg.length === TOMBSTONE_HEADER_BYTES + 13);
    // LSB-first within each byte (3.4): id 0 is bit 0 of byte 0, id 7 is bit 7.
    check('bit order is LSB-first', t.bitset[0] === 0b10000001, `byte0=${t.bitset[0].toString(2)}`);
}

console.log('tombstones-v1: empty and boundary shapes');
{
    const empty = buildTombstoneSegment({ rowBase: 0, tombstonedIds: [] });
    const t0 = parseTombstoneSegment(empty, { rowBase: 0, tombstones: 0, supersessions: 0 });
    check('rowBase 0 yields a header-only segment', empty.length === TOMBSTONE_HEADER_BYTES && t0.tombstoneCount === 0);

    // rowBase 8 fills exactly one byte, so there are no pad bits at all.
    const exact = buildTombstoneSegment({ rowBase: 8, tombstonedIds: [7] });
    check('rowBase on a byte boundary has no pad bits', parseTombstoneSegment(exact).tombstoneCount === 1);

    const one = buildTombstoneSegment({ rowBase: 1, tombstonedIds: [0] });
    check('rowBase 1 uses one byte with 7 pad bits', one.length === TOMBSTONE_HEADER_BYTES + 1);

    const none = buildTombstoneSegment({ rowBase: 500, tombstonedIds: [] });
    check('a layer may tombstone nothing', parseTombstoneSegment(none).tombstoneCount === 0);
}

console.log('tombstones-v1: supersessions (3.5)');
{
    const seg = buildTombstoneSegment({
        rowBase: 100, tombstonedIds: [10, 20, 30], records: 5,
        // Deliberately unsorted on input: the builder sorts by oldId.
        supersessions: [[30, 102], [10, 100]],
    });
    const t = parseTombstoneSegment(seg, { rowBase: 100, tombstones: 3, supersessions: 2, records: 5 });
    check('supersessions are stored sorted by oldId', JSON.stringify(t.supersessions) === JSON.stringify([[10, 100], [30, 102]]));
    check('segment length accounts for 8 bytes per edge',
        seg.length === TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100) + 16);

    rejects('an oldId that is not tombstoned is refused at build',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10], records: 5, supersessions: [[11, 100]] }),
        /not tombstoned/);
    rejects('a newId outside this layer\'s range is refused at build',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10], records: 5, supersessions: [[10, 150]] }),
        /outside this layer/);
    rejects('a newId below rowBase is refused at build',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10], records: 5, supersessions: [[10, 99]] }),
        /outside this layer/);
    rejects('a duplicate oldId within one layer is refused at build',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10], records: 5, supersessions: [[10, 100], [10, 101]] }),
        /more than one supersession/);
}

console.log('tombstones-v1: a layer cannot retract its own records (3.4)');
{
    rejects('an id at rowBase is out of range',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [100] }), /outside \[0, 100\)/);
    rejects('an id above rowBase is out of range',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [500] }), /outside \[0, 100\)/);
    rejects('a negative id is refused',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [-1] }), /outside \[0, 100\)/);
}

console.log('tombstones-v1: hostile segments are refused (section 10 family B)');
{
    const good = buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10, 20], records: 4, supersessions: [[10, 100]] });
    const tamper = (fn) => { const c = good.slice(); fn(new DataView(c.buffer), c); return c; };

    rejects('a wrong version', () => parseTombstoneSegment(tamper((v) => v.setUint32(0, 2, true))), /version 2/);
    rejects('non-zero flags', () => parseTombstoneSegment(tamper((v) => v.setUint32(4, 1, true))), /flags must be 0/);
    rejects('non-zero reserved bytes', () => parseTombstoneSegment(tamper((v) => v.setUint32(28, 7, true))), /reserved bytes must be zero/);
    rejects('a tombstoneCount that disagrees with the popcount',
        () => parseTombstoneSegment(tamper((v) => v.setBigUint64(16, 99n, true))), /popcount is 2/);
    rejects('a tombstoneCount above rowBase',
        () => parseTombstoneSegment(tamper((v) => v.setBigUint64(16, 200n, true))), /exceeds rowBase/);
    rejects('a rowBase whose bitset no longer fits',
        () => parseTombstoneSegment(tamper((v) => v.setBigUint64(8, 100000n, true))), /needs \d+ for the bitset/);
    rejects('a supersessionCount that does not tile the segment',
        () => parseTombstoneSegment(tamper((v) => v.setUint32(24, 9, true))), /needs 72/);
    rejects('a truncated segment', () => parseTombstoneSegment(good.subarray(0, 20)), /shorter than the 32-byte header/);
    rejects('a u64 above MAX_SAFE_INTEGER',
        () => parseTombstoneSegment(tamper((v) => v.setBigUint64(8, 2n ** 60n, true))), /MAX_SAFE_INTEGER/);

    // Pad bits: rowBase 100 leaves 4 pad bits in byte 12. Setting one must fail,
    // otherwise two byte-different segments would carry the same meaning and the
    // popcount check could be satisfied with a lie.
    rejects('a non-zero pad bit above rowBase', () => parseTombstoneSegment(tamper((v, c) => {
        c[TOMBSTONE_HEADER_BYTES + 12] |= 0b10000000;
    })), /pad bit/);

    // Manifest disagreement: the segment is well-formed but contradicts the
    // manifest that commits to it.
    rejects('a segment whose rowBase disagrees with the manifest',
        () => parseTombstoneSegment(good, { rowBase: 200, tombstones: 2, supersessions: 1 }),
        /disagrees with manifest layer.rowBase/);
    rejects('a segment whose tombstone count disagrees with the manifest',
        () => parseTombstoneSegment(good, { rowBase: 100, tombstones: 5, supersessions: 1 }),
        /disagrees with manifest layer.tombstones/);
    rejects('a segment whose supersession count disagrees with the manifest',
        () => parseTombstoneSegment(good, { rowBase: 100, tombstones: 2, supersessions: 4 }),
        /needs 32|disagrees with manifest layer.supersessions/);

    // Unsorted / duplicate supersessions, written past the builder's checks.
    const twoEdges = buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10, 20], records: 4, supersessions: [[10, 100], [20, 101]] });
    rejects('supersessions out of order', () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
        const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
        v.setUint32(at, 20, true); v.setUint32(at + 8, 10, true);
    })), /sorted by oldId/);
    rejects('a duplicate oldId', () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
        const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
        v.setUint32(at + 8, 10, true);
    })), /sorted by oldId/);
    rejects('an oldId that is not tombstoned in the bitset', () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
        const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
        v.setUint32(at, 11, true);
    })), /not tombstoned/);
    rejects('an oldId at or above rowBase', () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
        const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
        v.setUint32(at, 100, true);
    })), /not below rowBase/);
    rejects('a newId below rowBase, with no manifest record count to bound it',
        () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
            const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
            v.setUint32(at + 4, 50, true);
        })), /below rowBase/);
    rejects('a newId outside the layer range stated by the manifest',
        () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
            const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
            v.setUint32(at + 4, 900, true);
        }), { rowBase: 100, tombstones: 2, supersessions: 2, records: 4 }), /outside this layer/);
}
function tamper2(seg, fn) { const c = seg.slice(); fn(new DataView(c.buffer), c); return c; }

console.log('zeroExtend and the cumulative rule (3.4)');
{
    const parent = buildTombstoneSegment({ rowBase: 64, tombstonedIds: [1, 5] });
    const parentMask = parseTombstoneSegment(parent).bitset;

    const ext = zeroExtend(parentMask, 100);
    check('zeroExtend widens to the child\'s byte length', ext.length === bitsetBytesFor(100));
    check('zeroExtend preserves the parent\'s bits', testBit(ext, 1) && testBit(ext, 5));
    check('zeroExtend leaves the newly addressable range clear',
        popcountBytes(ext) === 2 && !testBit(ext, 64) && !testBit(ext, 99));
    check('zeroExtend of an empty parent is all zeros', popcountBytes(zeroExtend(new Uint8Array(0), 100)) === 0);

    // A conforming child: keeps both inherited bits and adds two of its own.
    const childMask = parseTombstoneSegment(
        buildTombstoneSegment({ rowBase: 100, tombstonedIds: [1, 5, 70, 80] }),
    ).bitset;
    check('a superset child passes', firstSupersetViolation(childMask, parentMask, 100) === -1);
    check('the delta is exactly the child\'s own deletions',
        JSON.stringify(tombstoneDelta(childMask, parentMask, 100)) === JSON.stringify([70, 80]));

    // A child that clears an inherited bit: the rule 3.4 exists to catch, and
    // 6.2's rebase correctness depends on it.
    const cleared = parseTombstoneSegment(
        buildTombstoneSegment({ rowBase: 100, tombstonedIds: [1, 70] }),
    ).bitset;
    check('a child clearing an inherited bit is caught, naming the id',
        firstSupersetViolation(cleared, parentMask, 100) === 5);

    check('an identical child has an empty delta',
        tombstoneDelta(parentMask, parentMask, 64).length === 0);
}

console.log('scaling (3.4)');
{
    // The documented sizes: the cumulative bitset costs ceil(rowBase/8) bytes
    // in every layer.
    check('456,153 records cost 57,020 bytes', bitsetBytesFor(456153) === 57020);
    check('1,000,000 records cost 125,000 bytes', bitsetBytesFor(1000000) === 125000);
    check('100,000,000 records cost 12.5 MB', bitsetBytesFor(100000000) === 12500000);
    rejects('a rowBase above the id ceiling is refused',
        () => buildTombstoneSegment({ rowBase: MAX_RECORDS + 1, tombstonedIds: [] }), /rowBase must be an integer/);
    rejects('rowBase + records above the id ceiling is refused',
        () => buildTombstoneSegment({ rowBase: MAX_RECORDS - 1, tombstonedIds: [], records: 10 }), /must not exceed/);
}

console.log('segment kinds (4.2)');
{
    check('tombstones is kind 6', KINDS.tombstones === 6 && KIND_NAMES[6] === 'tombstones');
    check('lineage is kind 7', KINDS.lineage === 7 && KIND_NAMES[7] === 'lineage');
    check('the complete profile\'s kinds are unchanged',
        KINDS.index === 1 && KINDS.corpus === 2 && KINDS['query-interp'] === 3
        && KINDS.evaluation === 4 && KINDS.lexical === 5);
}

// ---------------------------------------------------------------------------
// Layer manifest (4.1) and row arithmetic (3.3)
// ---------------------------------------------------------------------------
const INGEST = { chunker: 'v1', targetTokens: 256, overlapPercent: 15 };
const ENCODER = { kind: 'inline-transformer-v1', model: 'all-MiniLM-L6-v2' };
const BASE_ID = 'a'.repeat(64);
const L1_ID = 'b'.repeat(64);

function baseManifest(over = {}) {
    return {
        profile: 'pikelet-complete-v2', dim: 384, metric: 'cosine', encoder: ENCODER,
        corpus: { records: 1000, layout: 'records-v2', ingest: INGEST },
        index: { headerSha256: 'c'.repeat(64) },
        segments: [], ...over,
    };
}
function layerManifest(over = {}) {
    const { layer: layerOver, corpus: corpusOver, ...rest } = over;
    return {
        profile: LAYER_PROFILE, dim: 384, metric: 'cosine', encoder: ENCODER,
        layer: {
            parent: { identity: BASE_ID }, baseIdentity: BASE_ID, depth: 1,
            rowBase: 1000, records: 10, tombstones: 0, supersessions: 0, ingest: INGEST,
            ...layerOver,
        },
        corpus: { records: 10, layout: 'records-v2', ...corpusOver },
        index: { headerSha256: 'd'.repeat(64) },
        segments: [], ...rest,
    };
}

console.log('layer manifest: a well-formed layer');
{
    const m = layerManifest();
    const v = validateLayerObject(m);
    check('depth, rowBase and records survive', v.depth === 1 && v.rowBase === 1000 && v.records === 10);
    check('a layer with records is not tombstone-only', v.tombstoneOnly === false);
    const rel = validateAgainstParent(m, { manifest: baseManifest(), identity: BASE_ID, isBase: true });
    check('it validates against its base', rel.rowBase === 1000);
    check('rowTotal is rowBase + records', rowTotal(m) === 1010 && rowTotal(baseManifest()) === 1000);
}

console.log('layer manifest: tombstone-only layers (4.1, 4.3)');
{
    const t = layerManifest({
        layer: { records: 0, tombstones: 3 },
        corpus: { records: 0, layout: undefined },
        index: undefined,
    });
    const v = validateLayerObject(t);
    check('a tombstone-only layer is accepted', v.tombstoneOnly === true && v.records === 0);

    rejects('a tombstone-only layer carrying corpus.layout',
        () => validateLayerObject(layerManifest({ layer: { records: 0 }, corpus: { records: 0, layout: 'records-v2' }, index: undefined })),
        /must omit corpus.layout/);
    rejects('a tombstone-only layer carrying an index object',
        () => validateLayerObject(layerManifest({ layer: { records: 0 }, corpus: { records: 0, layout: undefined } })),
        /must omit the index object/);
    rejects('a tombstone-only layer recording a supersession (3.5)',
        () => validateLayerObject(layerManifest({ layer: { records: 0, tombstones: 2, supersessions: 1 }, corpus: { records: 0, layout: undefined }, index: undefined })),
        /tombstone-only layer cannot record a supersession/);
    rejects('a layer with records but no corpus.layout',
        () => validateLayerObject(layerManifest({ corpus: { records: 10, layout: undefined } })),
        /must declare corpus.layout/);
    rejects('a layer with records but no index object',
        () => validateLayerObject(layerManifest({ index: undefined })),
        /must carry an index object/);
}

console.log('layer manifest: hostile shapes (section 10 family B)');
{
    rejects('a complete-profile string', () => validateLayerObject(layerManifest({ profile: 'pikelet-complete-v2' })), /profile is/);
    rejects('no layer object', () => validateLayerObject({ profile: LAYER_PROFILE, corpus: { records: 0 } }), /no layer object/);
    rejects('a missing parent', () => validateLayerObject(layerManifest({ layer: { parent: undefined } })), /layer.parent is missing/);
    rejects('a short parent identity', () => validateLayerObject(layerManifest({ layer: { parent: { identity: 'abc' } } })), /64 lowercase hex/);
    rejects('an uppercase parent identity', () => validateLayerObject(layerManifest({ layer: { parent: { identity: 'A'.repeat(64) } } })), /64 lowercase hex/);
    rejects('a non-string locator', () => validateLayerObject(layerManifest({ layer: { parent: { identity: BASE_ID, locator: 7 } } })), /locator, when present/);
    rejects('depth 0', () => validateLayerObject(layerManifest({ layer: { depth: 0 } })), /layer.depth must be an integer/);
    rejects(`depth ${MAX_DEPTH + 1}`, () => validateLayerObject(layerManifest({ layer: { depth: MAX_DEPTH + 1 } })), /layer.depth must be an integer/);
    rejects('a negative rowBase', () => validateLayerObject(layerManifest({ layer: { rowBase: -1 } })), /layer.rowBase must be an integer/);
    rejects('rowBase + records past the id ceiling',
        () => validateLayerObject(layerManifest({ layer: { rowBase: MAX_RECORDS - 1, records: 10 }, corpus: { records: 10, layout: 'records-v2' } })),
        /exceeds 2147483647/);
    rejects('more tombstones than rowBase (a layer cannot tombstone its own rows)',
        () => validateLayerObject(layerManifest({ layer: { rowBase: 10, tombstones: 11 } })), /layer.tombstones must be an integer/);
    rejects('more supersessions than tombstones (3.5)',
        () => validateLayerObject(layerManifest({ layer: { tombstones: 1, supersessions: 2 } })), /exceeds layer.tombstones/);
    rejects('a missing ingestion declaration',
        () => validateLayerObject(layerManifest({ layer: { ingest: undefined } })), /layer.ingest is missing/);
    rejects('ingestAsserted smuggled inside the declaration (4.1)',
        () => validateLayerObject(layerManifest({ layer: { ingest: { ...INGEST, ingestAsserted: true } } })),
        /must not contain ingestAsserted/);
    rejects('a non-boolean ingestAsserted',
        () => validateLayerObject(layerManifest({ layer: { ingestAsserted: 'yes' } })), /must be a boolean/);
    rejects('corpus.records disagreeing with layer.records',
        () => validateLayerObject(layerManifest({ corpus: { records: 9, layout: 'records-v2' } })), /must equal layer.records/);
}

console.log('layer manifest: relational checks against the opened parent (3.3)');
{
    const base = baseManifest();
    const good = { manifest: base, identity: BASE_ID, isBase: true };

    rejects('a layer committing to a different parent identity',
        () => validateAgainstParent(layerManifest(), { manifest: base, identity: 'e'.repeat(64), isBase: true }),
        /but the opened parent is/);
    rejects('a wrong rowBase for the parent actually opened',
        () => validateAgainstParent(layerManifest({ layer: { rowBase: 999 } }), good),
        /but the opened parent implies 1000/);
    rejects('a wrong depth for the parent actually opened',
        () => validateAgainstParent(layerManifest({ layer: { depth: 2 } }), good),
        /but the opened parent is at depth 0/);
    rejects('a baseIdentity that is not the base\'s identity',
        () => validateAgainstParent(layerManifest({ layer: { baseIdentity: 'f'.repeat(64) } }), good),
        /but the chain's base is/);
    rejects('a layer that changes the encoder (4.1)',
        () => validateAgainstParent(layerManifest({ encoder: { kind: 'inline-transformer-v1', model: 'other' } }), good),
        /may not change the encoder/);
    rejects('a layer that changes dim',
        () => validateAgainstParent(layerManifest({ dim: 768 }), good), /differs from its parent's 384/);
    rejects('a layer that changes the metric',
        () => validateAgainstParent(layerManifest({ metric: 'l2' }), good), /differs from its parent's/);

    // Depth 2 onto a depth-1 layer: rowBase accumulates through the parent's
    // own rowBase, and a tombstone-only parent leaves rowBase unchanged (3.3).
    const l1 = layerManifest();
    const l2 = validateAgainstParent(
        layerManifest({ layer: { parent: { identity: L1_ID }, depth: 2, rowBase: 1010, records: 5 }, corpus: { records: 5, layout: 'records-v2' } }),
        { manifest: l1, identity: L1_ID, isBase: false },
    );
    check('depth 2 accumulates rowBase through its parent', l2.rowBase === 1010 && l2.depth === 2);

    const tombOnly = layerManifest({ layer: { records: 0, tombstones: 4 }, corpus: { records: 0, layout: undefined }, index: undefined });
    const after = validateAgainstParent(
        layerManifest({ layer: { parent: { identity: L1_ID }, depth: 2, rowBase: 1000, records: 3 }, corpus: { records: 3, layout: 'records-v2' } }),
        { manifest: tombOnly, identity: L1_ID, isBase: false },
    );
    check('rowBase is not unique along a chain: a tombstone-only parent leaves it unchanged (3.3)', after.rowBase === 1000);
}

console.log('chain ingestion declaration (4.1, 6.1)');
{
    const withIngest = baseManifest();
    const decl = chainIngestDeclaration(withIngest, null);
    check('a base carrying corpus.ingest defines the declaration', decl.source === 'base' && decl.asserted === false);
    check('a layer matching it passes', validateIngestAgainstChain(layerManifest(), decl) === true);
    rejects('a layer whose declaration differs',
        () => validateIngestAgainstChain(layerManifest({ layer: { ingest: { chunker: 'v2' } } }), decl),
        /differs from the chain's/);
    rejects('a layer asserting on a chain whose base declares',
        () => validateIngestAgainstChain(layerManifest({ layer: { ingestAsserted: true } }), decl),
        /but the chain's base declares one/);

    // A base with no corpus.ingest: only the depth-1 layer can assert one.
    const noIngest = baseManifest({ corpus: { records: 1000, layout: 'records-v2' } });
    rejects('a base with no declaration and no depth-1 layer',
        () => chainIngestDeclaration(noIngest, null), /no ingestion declaration/);
    rejects('a depth-1 layer that supplies one without asserting it',
        () => chainIngestDeclaration(noIngest, layerManifest()), /must set layer.ingestAsserted true/);
    const asserted = chainIngestDeclaration(noIngest, layerManifest({ layer: { ingestAsserted: true } }));
    check('a depth-1 layer may assert the declaration', asserted.asserted === true && asserted.source === 'depth-1');
    rejects('a later layer dropping the assertion',
        () => validateIngestAgainstChain(layerManifest({ layer: { depth: 2 } }), asserted),
        /must carry ingestAsserted true/);
    check('a later layer keeping it passes',
        validateIngestAgainstChain(layerManifest({ layer: { depth: 2, ingestAsserted: true } }), asserted) === true);
}

console.log('canonical JSON (4.1)');
{
    check('key order does not matter', canonicalJson({ b: 1, a: 2 }) === canonicalJson({ a: 2, b: 1 }));
    check('nested key order does not matter',
        canonicalJson({ x: { q: 1, p: 2 } }) === canonicalJson({ x: { p: 2, q: 1 } }));
    check('array order does matter', canonicalJson([1, 2]) !== canonicalJson([2, 1]));
    check('undefined members are dropped', canonicalJson({ a: 1, b: undefined }) === canonicalJson({ a: 1 }));
    // The property 4.1 depends on: asserting a declaration must not change its
    // canonical form, which is why ingestAsserted is a sibling.
    check('the declaration\'s canonical form is independent of assertion',
        canonicalJson(INGEST) === canonicalJson({ ...INGEST }));
}

// ---------------------------------------------------------------------------
// Query-interpretation kind 4, inherited-v1 (4.5)
// ---------------------------------------------------------------------------
const QI_SHA = '1'.repeat(64);
const BLOOM_GEOM = { bits: 2048, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'] };
const bloomOf = (setBits) => { const b = new Uint8Array(BLOOM_GEOM.bits / 8); for (const i of setBits) b[i >> 3] |= 1 << (i & 7); return b; };

console.log('inherited-v1: round trip');
{
    const seg = buildInheritedQuerySegment({
        baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
        vocabBloom: BLOOM_GEOM, vocabBloomBytes: bloomOf([1, 100, 2047]),
    });
    const p = parseInheritedQuerySegment(seg, {
        layerBaseIdentity: BASE_ID, baseQueryInterpSha256: QI_SHA, baseBloom: BLOOM_GEOM,
    });
    check('kind is 4', p.kind === QI_KIND_INHERITED);
    check('inheritFrom round-trips', p.inheritFrom === BASE_ID);
    check('queryInterpSha256 round-trips', p.queryInterpSha256 === QI_SHA);
    check('the layer bloom decodes to the base geometry', p.bloomBytes.length * 8 === BLOOM_GEOM.bits);
    check('calibration kind is layer-vocab-v1', p.calibrationKind === LAYER_VOCAB_KIND);
    // The point of kind 4 is that the encoder region is a COMMITMENT, not
    // weights: a ~24 MiB MiniLM blob would be two orders of magnitude larger.
    // (The segment's bulk is the layer's own base64 bloom, not the encoder.)
    const encRegion = new DataView(seg.buffer).getUint32(8, true);
    check('the encoder region is a commitment, not encoder weights',
        encRegion < 256, `encoder region is ${encRegion} bytes`);

    const none = buildInheritedQuerySegment({ baseIdentity: BASE_ID, queryInterpSha256: QI_SHA });
    const pn = parseInheritedQuerySegment(none, { layerBaseIdentity: BASE_ID, baseQueryInterpSha256: QI_SHA });
    check('a layer may ship kind "none"', pn.calibrationKind === 'none' && pn.bloomBytes === null);
}

console.log('inherited-v1: the encoder invariant (decision 5)');
{
    const seg = buildInheritedQuerySegment({ baseIdentity: BASE_ID, queryInterpSha256: QI_SHA });
    rejects('inheritFrom that is not the manifest\'s baseIdentity',
        () => parseInheritedQuerySegment(seg, { layerBaseIdentity: 'a9'.repeat(32), baseQueryInterpSha256: QI_SHA }),
        /does not equal the manifest's layer.baseIdentity/);
    rejects('a queryInterpSha256 that is not the base\'s query-interp digest',
        () => parseInheritedQuerySegment(seg, { layerBaseIdentity: BASE_ID, baseQueryInterpSha256: '2'.repeat(64) }),
        /does not match the base's query-interp digest/);
    rejects('a non-hex inheritFrom is refused at build',
        () => buildInheritedQuerySegment({ baseIdentity: 'nope', queryInterpSha256: QI_SHA }), /baseIdentity must be 64/);
    rejects('a non-hex queryInterpSha256 is refused at build',
        () => buildInheritedQuerySegment({ baseIdentity: BASE_ID, queryInterpSha256: 'nope' }), /queryInterpSha256 must be 64/);
}

console.log('inherited-v1: hostile segments (section 10 family B)');
{
    const seg = buildInheritedQuerySegment({
        baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
        vocabBloom: BLOOM_GEOM, vocabBloomBytes: bloomOf([5]),
    });
    const exp = { layerBaseIdentity: BASE_ID, baseQueryInterpSha256: QI_SHA, baseBloom: BLOOM_GEOM };
    const tamp = (fn) => { const c = seg.slice(); fn(new DataView(c.buffer), c); return c; };

    rejects('a wrong version', () => parseInheritedQuerySegment(tamp((v) => v.setUint32(0, 2, true)), exp), /unsupported version 2/);
    rejects('a kind that is not 4', () => parseInheritedQuerySegment(tamp((v) => v.setUint32(4, 3, true)), exp), /segment kind is 3/);
    rejects('regions that do not tile the segment',
        () => parseInheritedQuerySegment(tamp((v) => v.setUint32(8, 4, true)), exp), /layout is inconsistent/);
    rejects('a truncated segment', () => parseInheritedQuerySegment(seg.subarray(0, 8), exp), /shorter than its 16-byte header/);
    rejects('no expectation supplied', () => parseInheritedQuerySegment(seg, null), /no expectation supplied/);

    // A layer bloom whose geometry differs from the base's: the union would no
    // longer be an exact bitwise OR (4.5).
    const wrongBits = buildInheritedQuerySegment({
        baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
        vocabBloom: { bits: 4096, hashes: BLOOM_GEOM.hashes }, vocabBloomBytes: new Uint8Array(512),
    });
    rejects('a layer bloom with the wrong bit count',
        () => parseInheritedQuerySegment(wrongBits, exp), /the union would not be exact/);
    const wrongHashes = buildInheritedQuerySegment({
        baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
        vocabBloom: { bits: BLOOM_GEOM.bits, hashes: ['fnv1a:0'] }, vocabBloomBytes: bloomOf([5]),
    });
    rejects('a layer bloom with different hashes',
        () => parseInheritedQuerySegment(wrongHashes, exp), /differ from the base's/);
    rejects('a bloom whose base64 length disagrees with its declared bits',
        () => buildInheritedQuerySegment({
            baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
            vocabBloom: BLOOM_GEOM, vocabBloomBytes: new Uint8Array(4),
        }), /but vocabBloom.bits says/);
    rejects('an unknown calibration kind', () => {
        // Hand-build a segment whose calibration names a kind this reader does
        // not implement; it must fail closed rather than serve unscored.
        const enc = new TextEncoder();
        const e = enc.encode(JSON.stringify({ inheritFrom: BASE_ID, queryInterpSha256: QI_SHA }));
        const c = enc.encode(JSON.stringify({ kind: 'layer-vocab-v2' }));
        const out = new Uint8Array(QI_HEADER_BYTES + e.length + c.length);
        const v = new DataView(out.buffer);
        v.setUint32(0, 1, true); v.setUint32(4, 4, true); v.setUint32(8, e.length, true); v.setUint32(12, c.length, true);
        out.set(e, QI_HEADER_BYTES); out.set(c, QI_HEADER_BYTES + e.length);
        return parseInheritedQuerySegment(out, exp);
    }, /calibration kind must be/);
}

console.log('the union bloom (4.5)');
{
    const base = bloomOf([1, 2, 3]);
    const l1 = bloomOf([3, 10]);
    const l2 = bloomOf([200]);
    const u = unionBloom([base, l1, l2]);
    const isSet = (b, i) => ((b[i >> 3] >> (i & 7)) & 1) === 1;
    check('the union carries every contributor\'s bits',
        [1, 2, 3, 10, 200].every((i) => isSet(u, i)));
    check('the union sets nothing else', popcountBytes(u) === 5);
    check('overlapping bits are not double-counted', isSet(u, 3) && popcountBytes(unionBloom([bloomOf([3]), bloomOf([3])])) === 1);
    check('the union is exact, not approximate: OR of identical geometry',
        popcountBytes(unionBloom([base])) === 3);
    check('a null contributor is ignored', popcountBytes(unionBloom([base, null])) === 3);
    check('an all-null list yields null', unionBloom([null, null]) === null);
    rejects('a geometry mismatch is refused rather than silently truncated',
        () => unionBloom([base, new Uint8Array(8)]), /geometry mismatch/);
    // The union is monotone: it can only gain bits. This is why pollution is
    // one-sided (4.5) and cannot be undone without compaction.
    check('the union is a superset of every contributor',
        [base, l1, l2].every((b) => b.every((byte, i) => (byte & ~u[i]) === 0)));
}

// ---------------------------------------------------------------------------
// Parent resolution (5.1.1) — the one place a hostile manifest steers fetches
// ---------------------------------------------------------------------------
const CHILD_URL = 'https://cdn.example.com/packs/docs/docs.0007.9f3c.pikelet';

async function rejectsAsync(label, fn, pattern) {
    try {
        await fn();
        check(label, false, 'resolved instead of throwing');
    } catch (err) {
        const msg = String(err && err.message);
        const ok = pattern.test(msg);
        check(label, ok, ok ? '' : `threw: ${msg.slice(0, 140)}`);
    }
}

console.log('locator shape: only a relative reference under the child (5.1.1)');
{
    check('a plain sibling filename is accepted', validateLocatorShape('docs.0006.8e2b.pikelet') === true);
    check('a subdirectory is accepted', validateLocatorShape('layers/docs.0006.pikelet') === true);

    rejects('a scheme', () => validateLocatorShape('https://evil.example/x.pikelet'), /no scheme/);
    rejects('an authority', () => validateLocatorShape('//evil.example/x.pikelet'), /no authority/);
    rejects('a root-absolute path', () => validateLocatorShape('/admin/secret'), /not root-absolute/);
    rejects('a parent dot segment', () => validateLocatorShape('../../etc/passwd'), /dot segment/);
    rejects('a current dot segment', () => validateLocatorShape('./x.pikelet'), /dot segment/);
    // Every form on which a validator and a server could disagree is refused
    // rather than normalized — the governing rule of 5.1.1.
    rejects('an encoded path separator', () => validateLocatorShape('a%2fb.pikelet'), /encoded path separator/);
    rejects('an uppercase encoded separator', () => validateLocatorShape('a%2Fb.pikelet'), /encoded path separator/);
    rejects('an encoded backslash', () => validateLocatorShape('a%5cb.pikelet'), /encoded backslash/);
    rejects('a literal backslash', () => validateLocatorShape('a\\b.pikelet'), /literal backslash/);
    rejects('an encoded dot segment', () => validateLocatorShape('%2e%2e/x.pikelet'), /encoded dot segment/);
    rejects('a doubled slash', () => validateLocatorShape('a//b.pikelet'), /doubled slash/);
    rejects('an encoded NUL', () => validateLocatorShape('a%00b.pikelet'), /encoded NUL/);
    rejects('a control character', () => validateLocatorShape('a\u0001b.pikelet'), /control character/);
    rejects('a query string', () => validateLocatorShape('x.pikelet?a=1'), /no query or fragment/);
    rejects('a fragment', () => validateLocatorShape('x.pikelet#abc'), /no query or fragment/);
    rejects('a directory', () => validateLocatorShape('layers/'), /name a file/);
    rejects('an empty locator', () => validateLocatorShape(''), /non-empty string/);
    rejects('a non-string locator', () => validateLocatorShape(7), /non-empty string/);
}

console.log('locator resolution against the child\'s own URL (5.1.1)');
{
    check('a sibling resolves within the child\'s directory',
        resolveLocatorUrl('docs.0006.8e2b.pikelet', CHILD_URL)
            === 'https://cdn.example.com/packs/docs/docs.0006.8e2b.pikelet');
    check('a subdirectory resolves',
        resolveLocatorUrl('old/docs.0005.pikelet', CHILD_URL)
            === 'https://cdn.example.com/packs/docs/old/docs.0005.pikelet');
    rejects('a non-http child location', () => resolveLocatorUrl('x.pikelet', 'ftp://h/a/b'), /http and https only/);
}

console.log('redirect confinement, re-checked at every hop (5.1.1)');
{
    const child = new URL(CHILD_URL);
    check('a same-directory target is confined',
        assertConfined('https://cdn.example.com/packs/docs/other.pikelet', child) === true);
    check('a deeper path is confined',
        assertConfined('https://cdn.example.com/packs/docs/old/x.pikelet', child) === true);

    rejects('a cross-origin redirect',
        () => assertConfined('https://evil.example/packs/docs/x.pikelet', child), /leaves the child's origin/);
    rejects('a port change',
        () => assertConfined('https://cdn.example.com:8443/packs/docs/x.pikelet', child), /leaves the child's origin/);
    rejects('a scheme downgrade',
        () => assertConfined('http://cdn.example.com/packs/docs/x.pikelet', child), /changes scheme/);
    // The case 5.1.1 calls out by name: same origin, outside the directory.
    rejects('a same-origin redirect outside the directory',
        () => assertConfined('https://cdn.example.com/admin/private-object', child), /not under the child's directory/);
    rejects('a same-origin redirect to a sibling directory',
        () => assertConfined('https://cdn.example.com/packs/other/x.pikelet', child), /not under the child's directory/);
    rejects('a redirect carrying credentials',
        () => assertConfined('https://u:p@cdn.example.com/packs/docs/x.pikelet', child), /carries credentials/);
    rejects('a redirect naming the directory itself',
        () => assertConfined('https://cdn.example.com/packs/docs/', child), /names the directory itself/);
    // A redirect's Location is a fresh string off the network, so the raw-form
    // refusals apply again rather than only to the validated locator.
    rejects('a redirect target with an encoded backslash',
        () => assertConfined('https://cdn.example.com/packs/docs/a%5cb', child), /encoded backslash/);
}

console.log('file-source resolution follows symlinks (5.1.1)');
{
    // A fake fs where `link.pikelet` in the child's directory really lives
    // outside it: the textual check passes, the realpath check must not.
    const real = {
        '/packs/docs': '/packs/docs',
        '/packs/docs/parent.pikelet': '/packs/docs/parent.pikelet',
        '/packs/docs/link.pikelet': '/etc/secret.pikelet',
    };
    const fsops = {
        realpath: async (p) => { if (!(p in real)) throw new Error(`ENOENT ${p}`); return real[p]; },
        dirname: (p) => p.slice(0, p.lastIndexOf('/')) || '/',
        join: (...parts) => parts.join('/').replace(/\/+/g, '/'),
        sep: '/',
    };
    const childPath = '/packs/docs/child.pikelet';

    await (async () => {
        const got = await resolveLocatorPath('parent.pikelet', childPath, fsops);
        check('a real sibling resolves', got === '/packs/docs/parent.pikelet');
    })();
    await rejectsAsync('a symlink escaping the directory is refused on its real path',
        () => resolveLocatorPath('link.pikelet', childPath, fsops), /outside the child's real directory/);
    await rejectsAsync('a locator naming nothing is an availability failure',
        () => resolveLocatorPath('missing.pikelet', childPath, fsops), /cannot be located/);
}

console.log('resolution order: lineage, then locator, then host (5.1.1)');
{
    const want = { identity: BASE_ID, locator: 'parent.pikelet' };
    const lineage = new Map([[BASE_ID, 'https://ops.example.com/mirror/base.pikelet']]);

    await (async () => {
        // A lineage listing is operator-supplied and MAY name any origin the
        // operator trusts — deliberately not confined like a locator.
        const r = await resolveParentLocation(want, { lineage, childLocation: CHILD_URL });
        check('a lineage entry wins and may be cross-origin', r.via === 'lineage' && r.location.includes('ops.example.com'));

        const r2 = await resolveParentLocation(want, { childLocation: CHILD_URL });
        check('the locator is used when no lineage entry exists',
            r2.via === 'locator' && r2.location === 'https://cdn.example.com/packs/docs/parent.pikelet');

        const r3 = await resolveParentLocation({ identity: BASE_ID, locator: null }, {
            childLocation: CHILD_URL, hostResolver: (id) => `https://host.example/${id}.pikelet`,
        });
        check('the host resolver is the last resort', r3.via === 'host');
    })();

    await rejectsAsync('resolveParents false refuses a locator',
        () => resolveParentLocation(want, { childLocation: CHILD_URL, resolveParents: false }),
        /resolution is disabled/);
    // 3.2: never serve the layers that were reached as a partial corpus.
    await rejectsAsync('an unlocatable ancestor fails the mount explicitly',
        () => resolveParentLocation({ identity: BASE_ID, locator: null }, { childLocation: CHILD_URL }),
        /cannot locate chain member/);
}

// ---------------------------------------------------------------------------
// Chain state: interval table, id ownership, masks (5.1 step 4, 5.2, 5.3)
// ---------------------------------------------------------------------------
const id64 = (n) => String(n).padStart(2, '0').repeat(32);

console.log('interval table and search tiers (3.1, 5.2)');
{
    // base 100 records, L1 +10, L2 tombstone-only, L3 +5.
    const members = [
        { depth: 0, identity: id64(1), rowBase: 0, records: 100 },
        { depth: 1, identity: id64(2), rowBase: 100, records: 10 },
        { depth: 2, identity: id64(3), rowBase: 110, records: 0 },
        { depth: 3, identity: id64(4), rowBase: 110, records: 5 },
    ];
    const table = buildIntervalTable(members);
    check('the base is search tier 0', table.tiers[0].depth === 0 && table.tiers[0].rowBase === 0);
    check('a tombstone-only layer is not a search tier', table.tiers.length === 3);
    check('rowTotal counts every member\'s records', table.rowTotal === 115);
    check('tier intervals are contiguous',
        table.tiers[1].rowBase === 100 && table.tiers[1].end === 110
        && table.tiers[2].rowBase === 110 && table.tiers[2].end === 115);

    check('owner of a base id', ownerOf(table, 0).tier.depth === 0 && ownerOf(table, 0).localId === 0);
    check('owner of the last base id', ownerOf(table, 99).tier.depth === 0 && ownerOf(table, 99).localId === 99);
    check('owner of the first L1 id', ownerOf(table, 100).tier.depth === 1 && ownerOf(table, 100).localId === 0);
    // The trap 5.2 names: a tombstone-only layer shares its successor's
    // rowBase, so "largest rowBase <= id" would land on a member owning no ids.
    check('owner of an id after a tombstone-only layer is the layer that owns it',
        ownerOf(table, 110).tier.depth === 3 && ownerOf(table, 110).localId === 0);
    check('owner of the last id', ownerOf(table, 114).tier.depth === 3 && ownerOf(table, 114).localId === 4);
    check('an id past the end is rejected', ownerOf(table, 115) === null);
    check('a negative id is rejected', ownerOf(table, -1) === null);
    check('a non-integer id is rejected', ownerOf(table, 1.5) === null);

    rejects('a chain whose first member is not the base',
        () => buildIntervalTable([{ depth: 1, identity: id64(2), rowBase: 0, records: 5 }]), /must be the base/);
    rejects('a chain with a rowBase gap',
        () => buildIntervalTable([
            { depth: 0, identity: id64(1), rowBase: 0, records: 100 },
            { depth: 1, identity: id64(2), rowBase: 105, records: 10 },
        ]), /expected 100/);
    rejects('a chain with no search tier',
        () => buildIntervalTable([{ depth: 0, identity: id64(1), rowBase: 0, records: 0 }]), /at least one search tier/);
}

console.log('mask projection onto each tier (5.3)');
{
    const members = [
        { depth: 0, identity: id64(1), rowBase: 0, records: 100 },
        { depth: 1, identity: id64(2), rowBase: 100, records: 10 },
    ];
    const table = buildIntervalTable(members);
    // Head tombstones global ids 5, 99 (base) and 100, 109 (L1).
    const head = parseTombstoneSegment(
        buildTombstoneSegment({ rowBase: 110, tombstonedIds: [5, 99, 100, 109] }),
    ).bitset;
    const masks = projectMasks(table, head);
    check('one mask per search tier', masks.length === 2);
    check('base mask is ceil(records/8) bytes', masks[0].length === 13);
    check('a base tombstone projects to its local row', testBit(masks[0], 5) && testBit(masks[0], 99));
    check('an untombstoned base row is clear', !testBit(masks[0], 0) && !testBit(masks[0], 50));
    // The projection is the reason a tier's scan needs no global arithmetic:
    // global 100 is L1's local 0.
    check('an L1 tombstone projects to local 0 and 9', testBit(masks[1], 0) && testBit(masks[1], 9));
    check('L1 rows in between are clear', !testBit(masks[1], 5));
    check('popcounts sum to the head total', popcountBytes(masks[0]) + popcountBytes(masks[1]) === 4);
    check('live count is rowTotal minus tombstones', liveCount(table, 4) === 106);

    const noneMasked = projectMasks(table, new Uint8Array(0));
    check('an empty head bitset masks nothing',
        popcountBytes(noneMasked[0]) === 0 && popcountBytes(noneMasked[1]) === 0);
}

console.log('supersession views (3.5)');
{
    // A -> B recorded at depth 1, B -> C at depth 3, and a second edge for A
    // at depth 2 (the keep-both rebase case).
    const map = buildSupersessionMap([
        { depth: 1, supersessions: [[10, 100]] },
        { depth: 2, supersessions: [[10, 150]] },
        { depth: 3, supersessions: [[100, 200]] },
    ]);
    const a = resolveSupersession(map, 10);
    check('immediate is the least-depth edge', a.immediate === 100);
    check('all edges are ascending by depth', JSON.stringify(a.all) === JSON.stringify([100, 150]));
    // current: start at the greatest-depth edge (150), which has no further
    // edge, so it is the answer. 100's chain is not followed, because 150 is
    // the latest thing recorded for id 10.
    check('current starts from the greatest-depth edge', a.current === 150);

    const twoHop = buildSupersessionMap([
        { depth: 1, supersessions: [[10, 100]] },
        { depth: 2, supersessions: [[100, 200]] },
    ]);
    const b = resolveSupersession(twoHop, 10);
    check('current follows a two-hop chain A -> B -> C', b.current === 200);
    check('an id with no edges resolves to null', resolveSupersession(twoHop, 42).current === null);

    rejects('two edges for one oldId at the same depth',
        () => buildSupersessionMap([{ depth: 1, supersessions: [[10, 100], [10, 101]] }]),
        /at most one edge/);
}

console.log('calibration drift and status (3.1, 5.5)');
{
    const members = [
        { depth: 0, identity: id64(1), rowBase: 0, records: 1000 },
        { depth: 1, identity: id64(2), rowBase: 1000, records: 50 },
        { depth: 2, identity: id64(3), rowBase: 1050, records: 30 },
    ];
    // Drift counts history: 80 appended + 20 tombstoned over a 1000-record base.
    check('drift is (appended + tombstoned) / base records',
        Math.abs(calibrationDrift(members, 20) - 0.1) < 1e-12);
    check('a record appended then tombstoned counts twice',
        Math.abs(calibrationDrift(members, 80) - 0.16) < 1e-12);

    const st = (drift, envelope, reader, opts = {}) => calibrationStatus({
        drift, producerEnvelope: envelope, readerLimit: reader,
        baseHasFit: true, everyLayerShipsBloom: true, ...opts,
    });
    check('within both limits is inherited', st(0.1, 0.5, 0.2).status === 'inherited');
    check('past the reader limit is drift-exceeded', st(0.3, 0.5, 0.2).status === 'drift-exceeded');
    check('past the producer envelope is drift-exceeded', st(0.15, 0.1, 0.2).status === 'drift-exceeded');
    // The precedence 5.5 fixes: a host can always be stricter; an artifact can
    // never make a host less strict.
    check('a looser manifest cannot raise the host\'s limit', st(0.5, 0.9, 0.2).effectiveLimit === 0.2);
    check('a stricter manifest lowers it', st(0.05, 0.1, 0.2).effectiveLimit === 0.1);
    check('an absent envelope is no constraint', st(0.15, undefined, 0.2).effectiveLimit === 0.2);
    check('no base fit means status none', st(0.01, 0.5, 0.2, { baseHasFit: false }).status === 'none');
    check('a layer shipping no bloom degrades the chain',
        st(0.01, 0.5, 0.2, { everyLayerShipsBloom: false }).status === 'drift-exceeded');
    rejects('an infinite reader limit is refused',
        () => st(0.1, 0.5, Infinity), /MUST be finite/);
}

console.log('global lexical statistics (5.4)');
{
    const stats = globalLexicalStats([
        { docCount: 1000, totalTokens: 150000 },
        { docCount: 50, totalTokens: 6000 },
    ]);
    check('N sums every search tier', stats.N === 1050);
    check('avgdl is total tokens over N', Math.abs(stats.avgdl - 156000 / 1050) < 1e-9);
    check('the reader must report that stats include tombstoned records',
        stats.statsIncludeTombstoned === true);
    // idf uses the chain-wide N and summed df.
    check('idf falls as df rises', idfFor(1050, 500) < idfFor(1050, 5));
    check('a single tier degenerates to the single-file case',
        globalLexicalStats([{ docCount: 10, totalTokens: 100 }]).avgdl === 10);
}

console.log('ancestry and citation scoping (5.6)');
{
    const members = [
        { depth: 0, identity: id64(1), rowBase: 0, records: 100 },
        { depth: 1, identity: id64(2), rowBase: 100, records: 10 },
        { depth: 2, identity: id64(3), rowBase: 110, records: 5 },
    ];
    const anc = buildAncestry(members);
    check('every head on the ancestry is listed', anc.size === 3);
    check('rowTotal is the ids that existed when that head was current',
        anc.get(id64(1)).rowTotal === 100 && anc.get(id64(2)).rowTotal === 110 && anc.get(id64(3)).rowTotal === 115);

    check('a citation against the base resolves', resolveCitation(anc, id64(1), 50).ok === true);
    check('a citation against a later head resolves', resolveCitation(anc, id64(3), 114).ok === true);
    // The case 5.6 calls out: (H3, id) with id >= rowTotal(H1) is invalid under
    // H1's identity, even though the id is valid in the current head.
    check('an id that postdates the cited head is invalid',
        resolveCitation(anc, id64(1), 105).reason === 'id-postdates-that-head');
    check('a fork not on this ancestry is answered as such',
        resolveCitation(anc, id64(9), 5).reason === 'not-on-this-history');
    check('a negative id is invalid', resolveCitation(anc, id64(1), -1).reason === 'invalid-id');
}

console.log(`\nLayered profile conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
