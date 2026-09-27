// Tombstone segment (kind 6), layout `tombstones-v1` — LAYERED_PROFILE.md 4.4.
//
// A layer hides records by setting bits in a cumulative bitset over
// [0, rowBase), one bit per global id, LSB-first within each byte. The set is
// cumulative, not a delta: every bit set in the parent's mask (zero-extended
// over the ids the parent did not yet address) is set in the child's. A layer
// may also record that a new record supersedes a tombstoned one, as
// (oldId, newId) pairs.
//
// The segment is read eagerly at mount and verified against its manifest
// digest, but a digest only proves the publisher wrote these bytes — the
// publisher is also the adversary for a distributed artifact, so every
// structural rule below is enforced on parse, not assumed. Section 9's stance:
// "the tombstone and lineage layouts carry normative structural validation
// precisely because a publisher can sign anything."
//
//   [0,4)    u32 version = 1
//   [4,8)    u32 flags (0; readers MUST reject non-zero)
//   [8,16)   u64 rowBase             MUST equal manifest layer.rowBase
//   [16,24)  u64 tombstoneCount      popcount; MUST equal manifest layer.tombstones
//   [24,28)  u32 supersessionCount   MUST equal manifest layer.supersessions
//   [28,32)  reserved, zero
//   [32, 32 + ceil(rowBase/8))       bitset, LSB-first; pad bits above rowBase MUST be zero
//   [...]    supersessionCount x (u32 oldId, u32 newId), sorted by oldId, oldId unique

export const TOMBSTONE_HEADER_BYTES = 32;
export const TOMBSTONES_VERSION = 1;
// The reader's id ceiling (3.3): a chain holds at most 2^31 - 1 records, so
// every rowBase and rowTotal is bounded by it and every id by 2^31 - 2.
export const MAX_RECORDS = 2 ** 31 - 1;

const POPCOUNT = new Uint8Array(256);
for (let i = 0; i < 256; i++) POPCOUNT[i] = (i & 1) + POPCOUNT[i >> 1];

export function popcountBytes(bytes) {
    let n = 0;
    for (let i = 0; i < bytes.length; i++) n += POPCOUNT[bytes[i]];
    return n;
}

export function bitsetBytesFor(rowBase) {
    return Math.ceil(rowBase / 8);
}

export function testBit(bitset, id) {
    const byte = id >> 3;
    if (byte >= bitset.length) return false;
    return ((bitset[byte] >> (id & 7)) & 1) === 1;
}

function setBit(bitset, id) {
    bitset[id >> 3] |= 1 << (id & 7);
}

/**
 * Build a tombstones-v1 segment.
 *
 * @param {{rowBase: number, tombstonedIds: Iterable<number>, supersessions?: Array<[number, number]>,
 *          records?: number}} spec
 *   `records` is the layer's own record count, used only to validate that each
 *   `newId` lies in this layer's own id range — a layer cannot supersede into
 *   another layer's ids (3.5).
 */
export function buildTombstoneSegment({ rowBase, tombstonedIds, supersessions = [], records = 0 }) {
    if (!Number.isInteger(rowBase) || rowBase < 0 || rowBase > MAX_RECORDS) {
        throw new Error(`tombstones: rowBase must be an integer in [0, ${MAX_RECORDS}], got ${rowBase}`);
    }
    if (!Number.isInteger(records) || records < 0 || rowBase + records > MAX_RECORDS) {
        throw new Error(`tombstones: rowBase + records must not exceed ${MAX_RECORDS}`);
    }
    const bitsetBytes = bitsetBytesFor(rowBase);
    const bitset = new Uint8Array(bitsetBytes);
    for (const id of tombstonedIds) {
        if (!Number.isInteger(id) || id < 0 || id >= rowBase) {
            // A layer cannot retract its own records (3.4): ids at or above
            // rowBase are out of range by construction.
            throw new Error(`tombstones: id ${id} is outside [0, ${rowBase})`);
        }
        setBit(bitset, id);
    }
    const tombstoneCount = popcountBytes(bitset);

    const edges = [...supersessions].map(([oldId, newId]) => {
        if (!Number.isInteger(oldId) || !Number.isInteger(newId)) {
            throw new Error('tombstones: supersession ids must be integers');
        }
        if (!testBit(bitset, oldId)) {
            throw new Error(`tombstones: supersession oldId ${oldId} is not tombstoned in this layer's bitset`);
        }
        if (newId < rowBase || newId >= rowBase + records) {
            throw new Error(`tombstones: supersession newId ${newId} is outside this layer's range [${rowBase}, ${rowBase + records})`);
        }
        return [oldId, newId];
    });
    edges.sort((a, b) => a[0] - b[0]);
    for (let i = 1; i < edges.length; i++) {
        if (edges[i][0] === edges[i - 1][0]) {
            throw new Error(`tombstones: oldId ${edges[i][0]} appears in more than one supersession in this layer`);
        }
    }

    const bytes = new Uint8Array(TOMBSTONE_HEADER_BYTES + bitsetBytes + edges.length * 8);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, TOMBSTONES_VERSION, true);
    view.setUint32(4, 0, true);
    view.setBigUint64(8, BigInt(rowBase), true);
    view.setBigUint64(16, BigInt(tombstoneCount), true);
    view.setUint32(24, edges.length, true);
    view.setUint32(28, 0, true);
    bytes.set(bitset, TOMBSTONE_HEADER_BYTES);
    let at = TOMBSTONE_HEADER_BYTES + bitsetBytes;
    for (const [oldId, newId] of edges) {
        view.setUint32(at, oldId, true);
        view.setUint32(at + 4, newId, true);
        at += 8;
    }
    return bytes;
}

/**
 * Parse and fully validate a tombstones-v1 segment.
 *
 * @param {Uint8Array} bytes
 * @param {{rowBase: number, tombstones: number, supersessions: number, records?: number}} [expect]
 *   The manifest's `layer` fields. When given, every one is checked against the
 *   segment: a segment that disagrees with the manifest committing to it is a
 *   defect, and the reader refuses rather than picking a winner.
 */
export function parseTombstoneSegment(bytes, expect = null) {
    if (!(bytes instanceof Uint8Array)) throw new Error('tombstones: segment must be a Uint8Array');
    if (bytes.length < TOMBSTONE_HEADER_BYTES) {
        throw new Error(`tombstones: segment is ${bytes.length} bytes, shorter than the ${TOMBSTONE_HEADER_BYTES}-byte header`);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(0, true);
    if (version !== TOMBSTONES_VERSION) {
        throw new Error(`tombstones: version ${version} is not ${TOMBSTONES_VERSION}`);
    }
    const flags = view.getUint32(4, true);
    if (flags !== 0) throw new Error(`tombstones: flags must be 0, got ${flags}`);

    // u64 fields are read as BigInt and refused above MAX_SAFE_INTEGER rather
    // than silently losing precision, matching the reader's u64() rule.
    const rowBaseBig = view.getBigUint64(8, true);
    const countBig = view.getBigUint64(16, true);
    if (rowBaseBig > BigInt(Number.MAX_SAFE_INTEGER) || countBig > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Error('tombstones: a u64 field exceeds MAX_SAFE_INTEGER');
    }
    const rowBase = Number(rowBaseBig);
    const tombstoneCount = Number(countBig);
    const supersessionCount = view.getUint32(24, true);
    const reserved = view.getUint32(28, true);
    if (reserved !== 0) throw new Error('tombstones: reserved bytes must be zero');
    if (rowBase > MAX_RECORDS) throw new Error(`tombstones: rowBase ${rowBase} exceeds ${MAX_RECORDS}`);
    if (tombstoneCount > rowBase) {
        throw new Error(`tombstones: tombstoneCount ${tombstoneCount} exceeds rowBase ${rowBase}`);
    }

    // The regions MUST tile the segment exactly (4.4). Computed with
    // subtraction so a hostile count cannot overflow the comparison.
    const bitsetBytes = bitsetBytesFor(rowBase);
    if (bytes.length - TOMBSTONE_HEADER_BYTES < bitsetBytes) {
        throw new Error(`tombstones: segment holds ${bytes.length - TOMBSTONE_HEADER_BYTES} bytes after the header, needs ${bitsetBytes} for the bitset`);
    }
    const afterBitset = bytes.length - TOMBSTONE_HEADER_BYTES - bitsetBytes;
    if (afterBitset !== supersessionCount * 8) {
        throw new Error(`tombstones: ${afterBitset} bytes after the bitset but supersessionCount ${supersessionCount} needs ${supersessionCount * 8}`);
    }

    const bitset = bytes.subarray(TOMBSTONE_HEADER_BYTES, TOMBSTONE_HEADER_BYTES + bitsetBytes);
    // Pad bits above rowBase MUST be zero: otherwise two segments with the
    // same meaning could differ in bytes, and popcount would disagree.
    const padBits = bitsetBytes * 8 - rowBase;
    if (padBits > 0 && bitsetBytes > 0) {
        const lastByte = bitset[bitsetBytes - 1];
        const validMask = (1 << (8 - padBits)) - 1;
        if ((lastByte & ~validMask & 0xff) !== 0) {
            throw new Error(`tombstones: ${padBits} pad bit(s) above rowBase ${rowBase} are non-zero`);
        }
    }
    const actual = popcountBytes(bitset);
    if (actual !== tombstoneCount) {
        throw new Error(`tombstones: header says ${tombstoneCount} tombstones, bitset popcount is ${actual}`);
    }

    const supersessions = [];
    let at = TOMBSTONE_HEADER_BYTES + bitsetBytes;
    let prevOld = -1;
    for (let i = 0; i < supersessionCount; i++) {
        const oldId = view.getUint32(at, true);
        const newId = view.getUint32(at + 4, true);
        at += 8;
        if (oldId >= rowBase) {
            throw new Error(`tombstones: supersession oldId ${oldId} is not below rowBase ${rowBase}`);
        }
        if (!testBit(bitset, oldId)) {
            throw new Error(`tombstones: supersession oldId ${oldId} is not tombstoned`);
        }
        if (oldId <= prevOld) {
            throw new Error(`tombstones: supersessions must be sorted by oldId ascending and unique (${oldId} after ${prevOld})`);
        }
        prevOld = oldId;
        if (expect && Number.isInteger(expect.records)) {
            const lo = rowBase;
            const hi = rowBase + expect.records;
            if (newId < lo || newId >= hi) {
                throw new Error(`tombstones: supersession newId ${newId} is outside this layer's range [${lo}, ${hi})`);
            }
        } else if (newId < rowBase) {
            // Without the manifest's record count only the lower bound is
            // checkable; 3.5 requires newId >= rowBase unconditionally.
            throw new Error(`tombstones: supersession newId ${newId} is below rowBase ${rowBase}`);
        }
        supersessions.push([oldId, newId]);
    }

    if (expect) {
        if (Number.isInteger(expect.rowBase) && expect.rowBase !== rowBase) {
            throw new Error(`tombstones: segment rowBase ${rowBase} disagrees with manifest layer.rowBase ${expect.rowBase}`);
        }
        if (Number.isInteger(expect.tombstones) && expect.tombstones !== tombstoneCount) {
            throw new Error(`tombstones: segment tombstoneCount ${tombstoneCount} disagrees with manifest layer.tombstones ${expect.tombstones}`);
        }
        if (Number.isInteger(expect.supersessions) && expect.supersessions !== supersessionCount) {
            throw new Error(`tombstones: segment supersessionCount ${supersessionCount} disagrees with manifest layer.supersessions ${expect.supersessions}`);
        }
    }

    return { version, rowBase, tombstoneCount, supersessions, bitset };
}

/**
 * `zeroExtend(T_parent, rowBase_child)` — 3.4. The parent's mask followed by
 * zero bits over [rowBase_parent, rowBase_child): the parent's own records
 * become addressable by the child, and start clear.
 */
export function zeroExtend(parentBitset, childRowBase) {
    const out = new Uint8Array(bitsetBytesFor(childRowBase));
    if (parentBitset && parentBitset.length) {
        out.set(parentBitset.subarray(0, Math.min(parentBitset.length, out.length)));
    }
    return out;
}

/**
 * The superset (cumulative) rule of 3.4: every bit set in the zero-extended
 * parent mask MUST be set in the child's. Returns the first offending id, or
 * -1 when the rule holds.
 */
export function firstSupersetViolation(childBitset, parentBitset, childRowBase) {
    const extended = zeroExtend(parentBitset, childRowBase);
    for (let byte = 0; byte < extended.length; byte++) {
        const missing = extended[byte] & ~(childBitset[byte] || 0) & 0xff;
        if (missing !== 0) {
            for (let bit = 0; bit < 8; bit++) {
                if ((missing >> bit) & 1) return byte * 8 + bit;
            }
        }
    }
    return -1;
}

/**
 * `ΔT_child` — the deletions a child introduced over its parent (3.4). This is
 * what a rebase replays; it is derived, never stored.
 */
export function tombstoneDelta(childBitset, parentBitset, childRowBase) {
    const extended = zeroExtend(parentBitset, childRowBase);
    const ids = [];
    for (let byte = 0; byte < childBitset.length; byte++) {
        const added = childBitset[byte] & ~(extended[byte] || 0) & 0xff;
        if (added === 0) continue;
        for (let bit = 0; bit < 8; bit++) {
            if ((added >> bit) & 1) {
                const id = byte * 8 + bit;
                if (id < childRowBase) ids.push(id);
            }
        }
    }
    return ids;
}
