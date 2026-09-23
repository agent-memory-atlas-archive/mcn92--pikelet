// Shared loading and evidence matching for the labeled question sets under
// test/relevance/ (and the generated Veyra set). Used by
// scripts/calibration-harness.mjs and scripts/fusion-sim.mjs so both
// measure "the evidence is in this result" the same way.
//
// A set is either { questions: [...] } (Veyra style) or a flat array. A row
// is answerable unless its class is one of UNANSWERABLE_CLASSES, its type
// or expect is 'abstain', or its answer is null. Evidence comes in two
// shapes: { quote } objects (near-verbatim source text, checked against
// result text) or bare strings naming source records (fact-file ids such
// as "sup-velnor", matched against a record's sourcePath/url/anchor stem).

export const UNANSWERABLE_CLASSES = new Set(['unsupported', 'offdomain', 'unanswerable', 'nearmiss', 'near-miss']);

export function normalize(text) {
  return String(text || '').toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();
}

export function loadQuestions(raw) {
  const list = Array.isArray(raw) ? raw : raw.questions;
  return list.map((q) => ({
    id: q.id,
    cls: q.cls || q.category || 'direct',
    text: q.q || q.text || q.question,
    answer: q.answer ?? null,
    type: q.type || (q.answer == null ? 'abstain' : null),
    evidence: q.evidence || [],
    // { chapter, quote } evidence is near-verbatim source text, far more
    // reliable to check a passage against than a paraphrased `answer`
    // field the source may never restate word-for-word.
    evidenceQuotes: (q.evidence || [])
      .map((e) => (typeof e === 'object' && e?.quote) ? e.quote : null)
      .filter(Boolean),
    // String evidence names the source records; a multihop or counting
    // answer ("yes", "3") never appears verbatim in any single record.
    evidenceIds: (q.evidence || []).filter((e) => typeof e === 'string' && e),
  }));
}

export function isAnswerable(q) {
  if (UNANSWERABLE_CLASSES.has(q.cls)) return false;
  if (q.type === 'abstain') return false;
  if (q.answer == null) return false;
  return true;
}

// A short quote (<=6 words) is checked as a substring — long enough to be
// specific, short enough that near-verbatim source text reliably contains
// it whole. A longer quote is checked by significant-word overlap against
// a real threshold (60%+ of its content words present): a "near-verbatim"
// quote can carry minor transcription drift (punctuation, a dropped "the")
// that an exact substring check would wrongly fail on.
export function quotePresent(results, quote) {
  const needle = normalize(quote);
  if (!needle) return false;
  const words = needle.split(' ').filter((w) => w.length >= 3);
  return results.some((r) => {
    const hay = normalize(`${r.title || ''} ${r.text || ''}`);
    if (hay.includes(needle)) return true;
    if (words.length <= 6) return false; // short quotes: substring only, no partial credit
    const hits = words.filter((w) => hay.includes(w)).length;
    return hits / words.length >= 0.6;
  });
}

// Text evidence in any of the given results: evidence quotes when the set
// provides them (any one matching quote counts — a multi-hop question's
// evidence spans several passages), else the answer field.
export function answerPresent(results, q) {
  if (!results.length) return false;
  if (q.evidenceQuotes.length) {
    return q.evidenceQuotes.some((quote) => quotePresent(results, quote));
  }
  const needle = normalize(q.answer);
  if (!needle) return false;
  const words = needle.split(' ').filter((w) => w.length >= 3);
  return results.some((r) => {
    const hay = normalize(`${r.title || ''} ${r.text || ''}`);
    if (hay.includes(needle)) return true;
    if (!words.length) return false;
    return words.every((w) => hay.includes(w));
  });
}

export function recordMatchesId(r, id) {
  const stem = (v) => String(v || '').split('/').pop().replace(/\.[a-z0-9]+$/i, '').toLowerCase();
  const want = id.toLowerCase();
  return stem(r.sourcePath) === want || stem(r.url) === want || String(r.anchor || '').toLowerCase() === want;
}

// Evidence in one result: the named source record when the set gives ids,
// otherwise the quote/answer text check.
export function hasEvidence(r, q) {
  return q.evidenceIds.length ? q.evidenceIds.some((id) => recordMatchesId(r, id)) : answerPresent([r], q);
}

// 1-based rank of the first result carrying the evidence, or null.
export function rankIn(results, q) {
  const i = results.findIndex((r) => hasEvidence(r, q));
  return i === -1 ? null : i + 1;
}
