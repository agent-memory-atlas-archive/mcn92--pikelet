// Vector/lexical rank fusion for hybrid retrieval — the one definition the
// reader (index.mjs), the calibrator (pikelet/src/calibrate.mjs, which fits
// coverage on the passage fusion ranks first) and the BEIR ladder
// (benchmarks/beir/query-E.mjs) all call, so a fit-time or benchmark
// ranking is the ranking the reader serves.
//
// Reciprocal-rank fusion over the exact-reranked vector order and the BM25
// order: score = 1/(K + vRank) + lexicalWeight/(K + lexRank), lexical term
// present only for records BM25 returned. Every lexical candidate joins the
// vector pool (the reader reranks them exactly), so it always has a vRank.
//
// The vector-margin guard: when the vector top-1 leads the top-2 by at
// least guardMargin of its own distance, it keeps rank 1 regardless of
// fusion. Without it, with K=60 any record that appears in both lists
// outranks a vector-only rank 1 (1/(60+v) + w/(60+l) > 1/60 for any finite
// v when w=1), so a query whose terms match many chunks — a character name
// in a novel — fills the top ranks with term matches and buries the
// passage the encoder placed first. Measured on three sets with
// scripts/fusion-sim.mjs before any default changed.

export const FUSION_DEFAULTS = Object.freeze({
    rrfK: 60,
    lexicalWeight: 0.5,
    guardMargin: 0.05,
});

/**
 * @param {Array<{id: number, distance: number}>} vectorOrder hits sorted by
 *   exact distance ascending (the rerank output), including every lexical
 *   candidate.
 * @param {number[]} lexicalIds record ids in BM25 order (after the cutoff).
 * @param {{rrfK?: number, lexicalWeight?: number, guardMargin?: number}} [options]
 * @returns the same hit objects in fused order.
 */
export function fuseCandidates(vectorOrder, lexicalIds, options = {}) {
    const rrfK = options.rrfK ?? FUSION_DEFAULTS.rrfK;
    const lexicalWeight = options.lexicalWeight ?? FUSION_DEFAULTS.lexicalWeight;
    const guardMargin = options.guardMargin ?? FUSION_DEFAULTS.guardMargin;
    if (!(rrfK > 0) || !(lexicalWeight >= 0) || !(guardMargin >= 0)) {
        throw new Error('fuseCandidates: rrfK must be > 0, lexicalWeight and guardMargin >= 0');
    }
    if (!lexicalIds.length) return vectorOrder.slice();
    const lexRank = new Map();
    lexicalIds.forEach((id, i) => { if (!lexRank.has(id)) lexRank.set(id, i); });
    let fused = vectorOrder
        .map((hit, vRank) => ({
            hit,
            score: 1 / (rrfK + vRank) + (lexRank.has(hit.id) ? lexicalWeight / (rrfK + lexRank.get(hit.id)) : 0),
        }))
        .sort((a, b) => (b.score - a.score) || (a.hit.distance - b.hit.distance))
        .map((entry) => entry.hit);
    if (guardMargin > 0 && vectorOrder.length > 1) {
        const [first, second] = vectorOrder;
        // The margin is relative to the top-1's own distance, which needs a
        // positive denominator. first.distance === 0 is not a degenerate case
        // to skip but the STRONGEST possible top-1 -- cosine distance is
        // 1 - dot, so a query matching passage text verbatim scores exactly 0
        // -- and gating on `first.distance > 0` disabled the guard for exactly
        // the hit it exists to protect (a worse top-1 at 0.001 was guarded, a
        // perfect one at 0 was not). Any second distance strictly greater than
        // a zero first is an infinite relative lead, so the guard fires.
        // Distances are non-negative here (cosine/L2 as returned by the
        // rerank); a metric admitting negatives would need an absolute margin.
        const lead = first.distance > 0
            ? (second.distance - first.distance) / first.distance
            : (second.distance > 0 ? Infinity : 0);
        if (lead >= guardMargin && fused[0] !== first) {
            fused = [first, ...fused.filter((hit) => hit !== first)];
        }
    }
    return fused;
}
