// Client-side abstention scorer for the wiki pack. Mirrors the math in
// ../../calibrate_abstention.mjs exactly: retrieval signals (d0, margin,
// mean10) plus the corpus-vocabulary known-token fraction from the shipped
// bloom filter, standardized and passed through the fitted logistic model.
// Verdict semantics: 'answer' (strong match), 'weak' (closest match is
// distant — shown with a caveat), 'abstain' (nothing useful in the pack).
//
// Assets calibrated by pikelet's self-templates-v2+ may carry an additional
// grounding term (asset.coverage): the fraction of the query's content
// words that appear in the top retrieved passage's text (coverage1,
// exact/stemmed match). Every base feature measures topic similarity, so
// "the corpus discusses this area" and "this passage answers this
// question" are indistinguishable without a grounding term. The term is
// serialized outside features[]/weights[] deliberately: a reader that
// predates it scores the topic-only model against the same thresholds (a
// conservative degradation) instead of hitting an unknown feature name and
// computing NaN. Word rules (min length, stopwords) ship in the asset so
// builder and reader cannot drift.
//
// A semantic word-cosine grounding term (maxSim1, folded into grounding1
// via Math.max with coverage1, gated by asset.coverage.useMaxSim) was
// tried twice and reverted twice — see pikelet/src/calibrate.mjs's
// GROUNDING_FEAT design note for the two failure modes (an unrescaled
// version inflated the fitted threshold and caused widespread false
// abstention; a rescaled/damped version fixed that but still made
// real-query behavior worse than plain coverage1 alone). The mechanism
// itself was also measured to cost 200-400 individual per-word encoder
// calls per query once whatever built the calibration asset turned the
// flag on — a 30-80ms search became 1.5-4.6 seconds, with nothing in the
// response indicating why. Removed entirely (not just disabled) rather
// than left as a dormant code path: dead code an external asset can
// silently reactivate isn't actually dead, and both attempts already
// failed for a structural reason (see the design note), not a tuning
// one — there is no scenario where re-adding this via a config flag was
// going to end differently a third time.

export function createAbstentionScorer(asset, bloomBytes) {
    if (!asset || !Array.isArray(asset.weights) || !asset.thresholds) return null;
    const coverageCfg = asset.coverage && typeof asset.coverage.weight === 'number'
        && Number.isFinite(asset.coverage.mean) && Number.isFinite(asset.coverage.std)
        ? asset.coverage : null;
    const coverageStopwords = coverageCfg ? new Set(coverageCfg.stopwords || []) : null;
    const coverageMinLen = coverageCfg ? (coverageCfg.minWordLen || 3) : 3;
    // Words the corpus uses everywhere ground any query that mentions them,
    // so they are excluded from coverage; the builder ships them as a bloom.
    const commonBloom = coverageCfg?.commonBloom?.base64
        ? (typeof Buffer !== 'undefined'
            ? new Uint8Array(Buffer.from(coverageCfg.commonBloom.base64, 'base64'))
            : Uint8Array.from(atob(coverageCfg.commonBloom.base64), (c) => c.charCodeAt(0)))
        : null;
    const commonBits = coverageCfg?.commonBloom?.bits;
    const isCommon = commonBloom ? (w) => SEEDS.every((seed) => {
        let h = 0x811c9dc5 ^ seed;
        for (let i = 0; i < w.length; i++) { h ^= w.charCodeAt(i); h = Math.imul(h, 0x01000193); }
        const bit = (h >>> 0) % commonBits;
        return (commonBloom[bit >> 3] >> (bit & 7)) & 1;
    }) : null;

    // Max over the scored passages, with corpus-common words counted at
    // reduced weight — mirrors pikelet/src/calibrate.mjs
    // coverageFrac exactly (the weight ships in the asset).
    const commonWordWeight = coverageCfg?.commonWordWeight ?? 1 / 3;
    // Light suffix stripping so "quantize" grounds against "quantization"
    // and "configure" against "configuration" — kept byte-identical to
    // pikelet/src/calibrate.mjs's stem() (same suffix list and order):
    // present() otherwise only forgave plurals, so any word-form mismatch
    // at all scored zero coverage for a passage that plainly answers the
    // query. Deliberately conservative (STEM_MIN_LEN guards short words
    // like "king" from over-stripping); not a general stemmer.
    const STEM_MIN_LEN = 4;
    const STEM_SUFFIXES = ['ization', 'isation', 'ication', 'ation', 'ition', 'tion', 'ing', 'ed', 'ate', 'ize', 'ise'];
    function stem(w) {
        for (const suf of STEM_SUFFIXES) {
            if (w.length - suf.length >= STEM_MIN_LEN && w.endsWith(suf)) return w.slice(0, -suf.length);
        }
        if (w.length - 1 >= STEM_MIN_LEN && w.endsWith('e')) return w.slice(0, -1);
        return w;
    }
    // Kept byte-identical to pikelet/src/calibrate.mjs's stripPossessive:
    // the tokenizer regex keeps the apostrophe, so a possessive ("darcy's")
    // is one token that never matches a plain mention ("darcy") without
    // this, in either direction.
    function stripPossessive(w) {
        if (w.endsWith("'s")) return w.slice(0, -2);
        if (w.endsWith("s'")) return w.slice(0, -1);
        if (w.endsWith("'")) return w.slice(0, -1);
        return w;
    }
    // Kept byte-identical to pikelet/src/calibrate.mjs's isNumeral: a
    // numeral is exempted from the coverageMinLen floor because it is
    // often the only token distinguishing two otherwise-identical
    // passages ("Chamber 4" vs "Chamber 5"), unlike an ordinary short
    // word the floor exists to drop.
    const isNumeral = (w) => /^\d+$/.test(w);
    // A word present only in a passage's heading, not its body, still
    // grounds the query, at reduced credit — kept byte-identical to
    // pikelet/src/calibrate.mjs's HEADING_COVERAGE_WEIGHT. Full credit
    // there is the free-coverage bug body/heading splitting exists to fix
    // (a title-templated positive's own words ARE the heading); zero
    // credit is the opposite failure (a real rank-1 hit whose match is in
    // the heading scoring no coverage at all).
    const HEADING_COVERAGE_WEIGHT = 0.5;
    // Returns { value, grounding } rather than writing grounding detail to
    // a shared variable: score() is async, so a module- or closure-level
    // "last grounding" write would race between two concurrent queries
    // against the same open pack if anything ever awaited between this
    // call and reading it back — the second call's write could land, and
    // get read back, before the first call's own return. Threading the
    // value through the return keeps it local to this call regardless.
    function coverageFrac(text, passages) {
        const content = (String(text).toLowerCase().match(/[a-z0-9']+/g) || [])
            .map(stripPossessive)
            .filter((w) => (w.length >= coverageMinLen || isNumeral(w)) && !coverageStopwords.has(w));
        if (!content.length) return { value: 0, grounding: null };
        const weights = content.map((w) => (isCommon && isCommon(w) ? commonWordWeight : 1));
        const weightSum = weights.reduce((a, c) => a + c, 0);
        let best = 0; let bestGrounding = null; let passageIndex = 0;
        for (const { heading, body } of passages || []) {
            const bodyWords = (String(body || '').toLowerCase().match(/[a-z0-9']+/g) || []).map(stripPossessive);
            const bodySet = new Set(bodyWords);
            const bodyStems = new Set(bodyWords.map(stem));
            const headingSet = new Set((String(heading || '').toLowerCase().match(/[a-z0-9']+/g) || []).map(stripPossessive));
            const presentIn = (set, stems, w) => set.has(w) || set.has(`${w}s`) || set.has(`${w}es`)
                || (w.endsWith('s') && set.has(w.slice(0, -1)))
                || (stems && stems.has(stem(w)));
            const creditFor = (w) => {
                if (presentIn(bodySet, bodyStems, w)) return 1;
                if (presentIn(headingSet, null, w)) return HEADING_COVERAGE_WEIGHT;
                return 0;
            };
            const grounded = content.reduce((sum, w, i) => sum + creditFor(w) * weights[i], 0);
            const frac = grounded / weightSum;
            if (frac > best) {
                best = frac;
                bestGrounding = { passageIndex, coverage: +frac.toFixed(3), covered: content.filter((w) => creditFor(w) > 0), uncovered: content.filter((w) => creditFor(w) === 0) };
            }
            passageIndex++;
        }
        return { value: best, grounding: bestGrounding };
    }
    const bloom = new Uint8Array(bloomBytes);
    const bits = asset.vocabBloom.bits;

    function fnv1a(str, seed) {
        let h = 0x811c9dc5 ^ seed;
        for (let i = 0; i < str.length; i++) {
            h ^= str.charCodeAt(i);
            h = Math.imul(h, 0x01000193);
        }
        return (h >>> 0) % bits;
    }
    const SEEDS = [0, 0x9e3779b9];

    function knownFrac(text) {
        const words = text.toLowerCase().match(/[a-z0-9']+/g) || [];
        if (!words.length) return 0;
        let known = 0;
        for (const w of words) {
            const hit = SEEDS.every((seed) => {
                const bit = fnv1a(w, seed);
                return (bloom[bit >> 3] >> (bit & 7)) & 1;
            });
            if (hit) known++;
        }
        return known / words.length;
    }

    return {
        // The caller must hydrate the top passagesNeeded results' text
        // before scoring when usesPassage is true; the coverage/maxSim
        // terms read them (older assets scored one passage; topK now
        // defaults to it). maxSim never ships without coverage (calibrate.mjs
        // fits them together), so coverageCfg.topK already covers both.
        usesPassage: !!coverageCfg,
        passagesNeeded: coverageCfg ? (coverageCfg.topK || 1) : 0,
        async score(queryText, results, passageTexts) {
            // Fit at build time (pikelet/src/calibrate.mjs) always scores a
            // fixed top-10 window (K = min(10, candidates)), independent of
            // whatever k a caller later passes to query(). Slicing to the
            // caller's k here before windowing would shrink margin/mean10/
            // coverage's inputs at small k and change the verdict for an
            // identical retrieval — the window, not the caller's k, is what
            // must match the fit.
            const top = results.slice(0, Math.min(10, results.length));
            const d0 = top.length ? top[0].distance : 1;
            const margin = top.length > 1
                ? top[Math.min(4, top.length - 1)].distance - d0 : 0;
            const mean10 = top.length
                ? top.reduce((s, r) => s + r.distance, 0) / top.length : 1;
            const signals = { d0, margin, mean10, known_frac: knownFrac(queryText) };
            let z = asset.bias;
            asset.features.forEach((f, j) => {
                z += ((signals[f] - asset.standardize.mean[f]) / asset.standardize.std[f]) * asset.weights[j];
            });
            // Local to this call (not module/closure state — see
            // coverageFrac's comment) so concurrent queries against the
            // same open pack can never see each other's grounding detail.
            let grounding = null;
            if (coverageCfg) {
                // passageTexts is hydrated by the caller for the fixed
                // top-COVERAGE_TOP_PASSAGES window (see passagesNeeded
                // below), independent of the caller's k, matching
                // calibrate.mjs's top.slice(0, COVERAGE_TOP_PASSAGES).
                const passages = Array.isArray(passageTexts) ? passageTexts : [passageTexts];
                const coverage = coverageFrac(queryText, passages);
                signals.coverage1 = coverage.value;
                grounding = coverage.grounding;
                // grounding1 is coverage1 alone — see this file's header
                // comment and calibrate.mjs's GROUNDING_FEAT design note
                // for why a semantic (maxSim1) blend was tried twice and
                // removed both times.
                z += ((signals.coverage1 - coverageCfg.mean) / (coverageCfg.std || 1)) * coverageCfg.weight;
            }
            // A malformed asset (unknown feature name, non-numeric term) must
            // degrade to unscored, never to a NaN that compares false against
            // both thresholds and answers everything.
            if (!Number.isFinite(z)) return { p: null, verdict: 'unscored', signals };
            const p = 1 / (1 + Math.exp(-z));
            const verdict = p < asset.thresholds.hard ? 'abstain'
                : p < asset.thresholds.weak ? 'weak' : 'answer';
            return { p, verdict, signals, grounding };
        },
    };
}
