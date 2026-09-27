// Query-interpretation kind 4, `inherited-v1` — LAYERED_PROFILE.md 4.5.
//
// A layer never carries encoder bytes. It carries a commitment to the base's
// encoder instead, and the reader resolves the real encoder from the base it
// actually opened. Decision 5's invariant:
//
//   inheritFrom        MUST equal manifest layer.baseIdentity
//   queryInterpSha256  MUST equal the base manifest's segments[kind=query-interp].sha256
//
// Those two together bind the base's encoder bytes AND its calibration under
// the base's identity, which the layer's baseIdentity in turn commits to. So a
// layer cannot substitute an encoder, and cannot point at a different base's
// encoder, without breaking a digest the head's identity covers.
//
// The segment reuses the complete profile's query-interp header (version,
// kind, encoderLen, calibrationLen, then two regions that tile exactly):
//
//   [0,4)   u32 version = 1
//   [4,8)   u32 kind = 4
//   [8,12)  u32 encoderLen        the inherited-v1 JSON below
//   [12,16) u32 calibrationLen    layer-vocab-v1 JSON, or {"kind":"none"}
//
// The calibration region is this layer's own vocabulary bloom, built with the
// base's geometry so the union of 4.5 is an exact bitwise OR. A layer whose
// region is `{"kind":"none"}` while the base carries a fit degrades the chain
// to `unscored` (5.5), so a producer SHOULD always ship the bloom.

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export const QI_HEADER_BYTES = 16;
export const QI_KIND_INHERITED = 4;
export const INHERITED_VERSION = 1;
export const LAYER_VOCAB_KIND = 'layer-vocab-v1';

const isHex64 = (v) => typeof v === 'string' && /^[0-9a-f]{64}$/.test(v);

/**
 * Build a kind-4 query-interp segment.
 *
 * @param {{baseIdentity: string, queryInterpSha256: string,
 *          vocabBloom?: {bits: number, hashes: string[]}|null,
 *          vocabBloomBytes?: Uint8Array|null}} spec
 *   Omitting the bloom writes `{"kind":"none"}`, which is legal but degrades a
 *   scored chain to unscored — the caller is expected to know that.
 */
export function buildInheritedQuerySegment({ baseIdentity, queryInterpSha256, vocabBloom = null, vocabBloomBytes = null }) {
    if (!isHex64(baseIdentity)) throw new Error('inherited-v1: baseIdentity must be 64 lowercase hex characters');
    if (!isHex64(queryInterpSha256)) throw new Error('inherited-v1: queryInterpSha256 must be 64 lowercase hex characters');
    const encoderJson = encoder.encode(JSON.stringify({ inheritFrom: baseIdentity, queryInterpSha256 }));

    let calibrationJson;
    if (vocabBloom && vocabBloomBytes) {
        if (!Number.isSafeInteger(vocabBloom.bits) || vocabBloom.bits <= 0 || vocabBloom.bits % 8 !== 0) {
            throw new Error('inherited-v1: vocabBloom.bits must be a positive multiple of 8');
        }
        if (vocabBloomBytes.length * 8 !== vocabBloom.bits) {
            throw new Error(`inherited-v1: bloom is ${vocabBloomBytes.length * 8} bits but vocabBloom.bits says ${vocabBloom.bits}`);
        }
        if (!Array.isArray(vocabBloom.hashes) || vocabBloom.hashes.length === 0) {
            throw new Error('inherited-v1: vocabBloom.hashes must be a non-empty array');
        }
        calibrationJson = encoder.encode(JSON.stringify({
            kind: LAYER_VOCAB_KIND,
            vocabBloom: { bits: vocabBloom.bits, hashes: vocabBloom.hashes },
            vocabBloomBase64: base64FromBytes(vocabBloomBytes),
        }));
    } else if (vocabBloom || vocabBloomBytes) {
        throw new Error('inherited-v1: vocabBloom and vocabBloomBytes must be supplied together');
    } else {
        calibrationJson = encoder.encode(JSON.stringify({ kind: 'none' }));
    }

    const out = new Uint8Array(QI_HEADER_BYTES + encoderJson.length + calibrationJson.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, INHERITED_VERSION, true);
    view.setUint32(4, QI_KIND_INHERITED, true);
    view.setUint32(8, encoderJson.length, true);
    view.setUint32(12, calibrationJson.length, true);
    out.set(encoderJson, QI_HEADER_BYTES);
    out.set(calibrationJson, QI_HEADER_BYTES + encoderJson.length);
    return out;
}

/**
 * Parse and validate a kind-4 segment against the layer's manifest and the
 * base a reader actually opened.
 *
 * @param {Uint8Array} bytes
 * @param {{layerBaseIdentity: string, baseQueryInterpSha256: string,
 *          baseBloom?: {bits: number, hashes: string[]}|null}} expect
 *   `layerBaseIdentity` comes from the layer's own manifest; the other two come
 *   from the BASE's manifest. Passing them separately is the point: the check
 *   is that the layer's claim matches the base actually resolved, so neither
 *   side can be supplied by the other.
 */
export function parseInheritedQuerySegment(bytes, expect) {
    if (!(bytes instanceof Uint8Array)) throw new Error('inherited-v1: segment must be a Uint8Array');
    if (bytes.length < QI_HEADER_BYTES) throw new Error('inherited-v1: segment is shorter than its 16-byte header');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(0, true);
    if (version !== INHERITED_VERSION) throw new Error(`inherited-v1: unsupported version ${version}`);
    const kind = view.getUint32(4, true);
    if (kind !== QI_KIND_INHERITED) throw new Error(`inherited-v1: segment kind is ${kind}, expected ${QI_KIND_INHERITED}`);
    const encoderLen = view.getUint32(8, true);
    const calibrationLen = view.getUint32(12, true);
    // Regions must tile the segment exactly, computed without overflow.
    if (bytes.length - QI_HEADER_BYTES !== encoderLen + calibrationLen
        || encoderLen > bytes.length || calibrationLen > bytes.length) {
        throw new Error('inherited-v1: segment layout is inconsistent');
    }

    let decl;
    try {
        decl = JSON.parse(decoder.decode(bytes.subarray(QI_HEADER_BYTES, QI_HEADER_BYTES + encoderLen)));
    } catch (err) {
        throw new Error('inherited-v1: encoder region is not valid JSON', { cause: err });
    }
    if (!decl || typeof decl !== 'object' || Array.isArray(decl)) {
        throw new Error('inherited-v1: encoder region must be a JSON object');
    }
    if (!isHex64(decl.inheritFrom)) throw new Error('inherited-v1: inheritFrom must be 64 lowercase hex characters');
    if (!isHex64(decl.queryInterpSha256)) throw new Error('inherited-v1: queryInterpSha256 must be 64 lowercase hex characters');

    if (!expect || typeof expect !== 'object') throw new Error('inherited-v1: no expectation supplied');
    // MUST equal manifest layer.baseIdentity (4.5).
    if (decl.inheritFrom !== expect.layerBaseIdentity) {
        throw new Error(`inherited-v1: inheritFrom ${decl.inheritFrom.slice(0, 12)}… does not equal the manifest's layer.baseIdentity ${String(expect.layerBaseIdentity).slice(0, 12)}…`);
    }
    // MUST equal the base manifest's query-interp segment digest (4.5). This is
    // the encoder invariant: it binds the base's encoder bytes and calibration
    // together, so a layer cannot load a different encoder.
    if (decl.queryInterpSha256 !== expect.baseQueryInterpSha256) {
        throw new Error(`inherited-v1: queryInterpSha256 ${decl.queryInterpSha256.slice(0, 12)}… does not match the base's query-interp digest ${String(expect.baseQueryInterpSha256).slice(0, 12)}…`);
    }

    let calibration;
    try {
        calibration = JSON.parse(decoder.decode(bytes.subarray(QI_HEADER_BYTES + encoderLen)));
    } catch (err) {
        throw new Error('inherited-v1: calibration region is not valid JSON', { cause: err });
    }
    if (!calibration || typeof calibration !== 'object' || Array.isArray(calibration)) {
        throw new Error('inherited-v1: calibration region must be a JSON object');
    }

    let bloomBytes = null;
    if (calibration.kind === LAYER_VOCAB_KIND) {
        const geom = calibration.vocabBloom;
        if (!geom || typeof geom !== 'object') throw new Error('inherited-v1: layer-vocab-v1 needs a vocabBloom object');
        if (typeof calibration.vocabBloomBase64 !== 'string') {
            throw new Error('inherited-v1: layer-vocab-v1 needs vocabBloomBase64');
        }
        bloomBytes = bytesFromBase64(calibration.vocabBloomBase64);
        if (bloomBytes.length * 8 !== geom.bits) {
            throw new Error(`inherited-v1: bloom decodes to ${bloomBytes.length * 8} bits but vocabBloom.bits says ${geom.bits}`);
        }
        // The geometry MUST match the base asset's, which is what makes the
        // union of 4.5 an exact bitwise OR rather than an approximation.
        if (expect.baseBloom) {
            if (geom.bits !== expect.baseBloom.bits) {
                throw new Error(`inherited-v1: layer bloom is ${geom.bits} bits but the base's is ${expect.baseBloom.bits}; the union would not be exact`);
            }
            const a = JSON.stringify(geom.hashes);
            const b = JSON.stringify(expect.baseBloom.hashes);
            if (a !== b) {
                throw new Error(`inherited-v1: layer bloom hashes ${a} differ from the base's ${b}`);
            }
        }
    } else if (calibration.kind !== 'none') {
        throw new Error(`inherited-v1: calibration kind must be ${LAYER_VOCAB_KIND} or "none", got ${JSON.stringify(calibration.kind)}`);
    }

    return {
        version, kind,
        inheritFrom: decl.inheritFrom,
        queryInterpSha256: decl.queryInterpSha256,
        calibrationKind: calibration.kind,
        bloomBytes,
        bloomGeometry: bloomBytes ? { bits: calibration.vocabBloom.bits, hashes: calibration.vocabBloom.hashes } : null,
    };
}

/**
 * The union bloom of 4.5: the bitwise OR of the base's and every layer's. The
 * identical geometry makes it exact. A reader MUST compute this before scoring
 * — without it the known-token signal under-reads every query about a layer's
 * content, and the scorer abstains on exactly the records the layer added.
 *
 * Note what the union means: it covers the vocabulary of every record EVER
 * appended, live or tombstoned. Nothing can be removed from a bloom, so this
 * is not, and cannot be made, a bloom over the live vocabulary. Section 13.2
 * measured the consequence.
 */
export function unionBloom(blooms) {
    const present = blooms.filter(Boolean);
    if (!present.length) return null;
    const bytes = present[0].length;
    for (const b of present) {
        if (b.length !== bytes) {
            throw new Error(`unionBloom: geometry mismatch — ${b.length} bytes against ${bytes}; every layer's bloom must use the base's bits`);
        }
    }
    const out = new Uint8Array(bytes);
    out.set(present[0]);
    for (let i = 1; i < present.length; i++) {
        const b = present[i];
        for (let j = 0; j < bytes; j++) out[j] |= b[j];
    }
    return out;
}

function base64FromBytes(bytes) {
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
    let s = '';
    for (const b of bytes) s += String.fromCharCode(b);
    return btoa(s);
}

function bytesFromBase64(text) {
    if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(text, 'base64'));
    return Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
}
