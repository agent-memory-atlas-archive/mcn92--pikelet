#!/usr/bin/env node
// Abstention-correctness regression harness: runs a labeled question set
// against a compiled .pikelet artifact and reports false-abstention and
// false-answer rates — the two failure modes calibration exists to avoid,
// and the numbers that must be measured against real questions, not just
// AUC on synthetic rows (see pikelet/src/calibrate.mjs's GROUNDING_FEAT
// design note for why that metric has misled this project more than once).
//
//   node scripts/calibration-harness.mjs <artifact.pikelet> <queries.json> [--baseline <baseline.json>]
//
// queries.json accepts two shapes:
//   1. { questions: [{ id, cls, q, answer, type, evidence }, ...] }  (Veyra style)
//   2. [{ id, category, q, answer, evidence }, ...]                  (flat array)
//
// A question is "answerable" unless its cls/category is one of
// UNANSWERABLE_CLASSES, or its type/expect is 'abstain', or its answer is
// null. For an answerable question, the harness checks BOTH that the
// verdict didn't abstain (matchQuality !== 'none') AND that the top
// result's text actually contains the expected answer — an answerable
// question with the right verdict but the wrong record is still a
// failure (a confident wrong answer), and is reported separately from a
// false abstention.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { openPikeletFile } = await import(path.join(ROOT, 'packages', 'pikelet-wasm', 'complete', 'index.mjs'));

function extractFlag(args, name) {
  const idx = args.indexOf(name);
  if (idx === -1) return [args, null];
  return [args.filter((_, i) => i !== idx && i !== idx + 1), args[idx + 1]];
}

let args = process.argv.slice(2);
let baselinePath, retrievalMode;
[args, baselinePath] = extractFlag(args, '--baseline');
[args, retrievalMode] = extractFlag(args, '--retrieval');
const [artifactPath, queriesPath] = args;
if (!artifactPath || !queriesPath) {
  console.error('usage: node scripts/calibration-harness.mjs <artifact.pikelet> <queries.json> [--baseline <baseline.json>] [--retrieval hybrid|vector|lexical|augmented]');
  process.exit(1);
}
if (retrievalMode && !['hybrid', 'vector', 'lexical', 'augmented'].includes(retrievalMode)) {
  console.error(`--retrieval must be hybrid, vector, lexical, or augmented, got ${retrievalMode}`);
  process.exit(1);
}

const UNANSWERABLE_CLASSES = new Set(['unsupported', 'offdomain', 'unanswerable', 'nearmiss', 'near-miss']);

function normalize(text) {
  return String(text || '').toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();
}

function loadQuestions(raw) {
  const list = Array.isArray(raw) ? raw : raw.questions;
  return list.map((q) => ({
    id: q.id,
    cls: q.cls || q.category || 'direct',
    text: q.q || q.text || q.question,
    answer: q.answer ?? null,
    type: q.type || (q.answer == null ? 'abstain' : null),
    evidence: q.evidence || [],
    // Veyra-style evidence is an array of fact-file ids (strings); the
    // Pride-and-Prejudice-style set instead carries { chapter, quote }
    // objects — quote is near-verbatim source text, and a much more
    // reliable thing to check a retrieved passage against than a
    // paraphrased/summarized `answer` field, which the source text may
    // never restate word-for-word at all.
    evidenceQuotes: (q.evidence || [])
      .map((e) => (typeof e === 'object' && e?.quote) ? e.quote : null)
      .filter(Boolean),
  }));
}

function isAnswerable(q) {
  if (UNANSWERABLE_CLASSES.has(q.cls)) return false;
  if (q.type === 'abstain') return false;
  if (q.answer == null) return false;
  return true;
}

// A short quote (<=6 words) is checked as a substring — long enough to be
// specific, short enough that near-verbatim source text reliably contains
// it whole. A longer quote is checked by significant-word overlap against
// a real threshold (60%+ of its content words present), not
// requiring every word — a "near-verbatim" quote can still have minor
// transcription drift (punctuation, a dropped "the") that an exact
// substring check would wrongly fail on.
function quotePresent(results, quote) {
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

function answerPresent(results, q) {
  if (!results.length) return false;
  // Prefer evidence quotes (near-verbatim source text) when the question
  // set provides them — far more reliable than matching against a
  // paraphrased/summarized answer field, which the source may never
  // restate word-for-word. Any one matching quote counts (a multi-hop
  // question's evidence spans several passages; the retrieved top-k only
  // needs to contain the specific fact being asked about).
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

const raw = JSON.parse(fs.readFileSync(queriesPath, 'utf8'));
const questions = loadQuestions(raw);

const search = await openPikeletFile(artifactPath);
const info = search.info();
console.log(`artifact: ${path.basename(artifactPath)} — ${info.records} records, identity ${info.identity?.slice(0, 16)}…`);
console.log(`questions: ${questions.length}  retrieval: ${retrievalMode || 'hybrid (default)'}\n`);

// This harness runs with showAbstained: true — the MCP search tool's own
// default (pikelet/src/mcp.mjs: "the calibrator can be wrong, especially
// on paraphrases... so results ship even under a 'none' verdict by
// default"). That means matchQuality === 'none' does NOT mean the caller
// got nothing: results still ship, just labeled low-confidence, and a
// reasoning caller (an LLM reading the passage, not a raw API consumer
// blindly trusting the label) can often still recover the right answer
// from them. So "genuinely denied any evidence" (results.length === 0,
// which only happens with showAbstained: false) is tracked separately
// from "labeled none but the evidence still shipped and was/wasn't
// actually usable" — conflating them (as an earlier version of this
// harness did) overstates the real-world failure rate for exactly the
// serving path this project actually ships by default.
const perClass = new Map();
const failures = { withheld: [], noneButRecovered: [], noneButWrong: [], falseAnswer: [], correct: 0 };

for (const q of questions) {
  const out = await search.query(q.text, { k: 3, showAbstained: true, ...(retrievalMode ? { retrieval: retrievalMode } : {}) });
  const answerable = isAnswerable(q);
  const withheld = out.results.length === 0;
  const stats = perClass.get(q.cls) || {
    n: 0, correct: 0, withheld: 0, noneButRecovered: 0, noneButWrong: 0, falseAnswer: 0,
  };
  stats.n++;

  if (answerable) {
    if (withheld) {
      stats.withheld++;
      failures.withheld.push({ id: q.id, cls: q.cls, q: q.text, verdict: out.matchQuality, confidence: out.confidence });
    } else if (!answerPresent(out.results, q)) {
      const bucket = out.matchQuality === 'none' ? 'noneButWrong' : 'falseAnswer';
      stats[bucket]++;
      failures[bucket].push({ id: q.id, cls: q.cls, q: q.text, expected: q.answer, verdict: out.matchQuality, confidence: out.confidence, top: out.results[0]?.title });
    } else if (out.matchQuality === 'none') {
      // The label said "no confident answer," but the answer was in the
      // shipped results anyway — this is the case a reasoning caller
      // recovers and a raw-results consumer (or the old stricter-default
      // harness) would have missed or miscounted as a hard failure.
      stats.noneButRecovered++;
      failures.noneButRecovered.push({ id: q.id, cls: q.cls, q: q.text, verdict: out.matchQuality, confidence: out.confidence });
      stats.correct++;
      failures.correct++;
    } else {
      stats.correct++;
      failures.correct++;
    }
  } else {
    // Unanswerable: correct behavior is to withhold, abstain (none), or
    // answer with a verdict weak enough (weak) that a caller wouldn't
    // trust it outright — strong verdict on an unanswerable question is
    // a false answer regardless of whether results were technically
    // withheld.
    if (withheld || out.matchQuality === 'none' || out.matchQuality === 'weak') {
      stats.correct++;
      failures.correct++;
    } else {
      stats.falseAnswer++;
      failures.falseAnswer.push({ id: q.id, cls: q.cls, q: q.text, expected: null, verdict: out.matchQuality, confidence: out.confidence, top: out.results[0]?.title });
    }
  }
  perClass.set(q.cls, stats);
}

const table = [...perClass.entries()].map(([cls, s]) => ({
  cls,
  n: s.n,
  correct: s.correct,
  withheld: s.withheld,
  noneButWrong: s.noneButWrong,
  falseAnswer: s.falseAnswer,
  'correct%': ((s.correct / s.n) * 100).toFixed(1),
}));
console.table(table);

const totalAnswerable = questions.filter(isAnswerable).length;
const totalUnanswerable = questions.length - totalAnswerable;
const result = {
  artifact: path.basename(artifactPath),
  identity: info.identity,
  totalQuestions: questions.length,
  totalAnswerable,
  totalUnanswerable,
  // "Withheld" is the only true false-abstention: results.length === 0,
  // meaning a caller (LLM or otherwise) received literally nothing to
  // reason over. This only happens with showAbstained: false; this
  // harness runs with showAbstained: true (the MCP tool's own default),
  // so a genuinely low corpus/question mismatch is the only way to reach
  // it — see mcp.mjs's search tool description for why that default
  // exists: the calibrator can misjudge paraphrases, so withholding by
  // default was measured to hide real answers too often.
  withheldCount: failures.withheld.length,
  withheldRate: totalAnswerable ? +(failures.withheld.length / totalAnswerable).toFixed(4) : null,
  // "None but recovered": labeled low-confidence, but the correct answer
  // was in the shipped results anyway — the case a reasoning caller
  // (reading the passage, not just trusting the label) gets right despite
  // the calibrator's own uncertainty. Counted as correct above, reported
  // here separately so this number is visible rather than hidden inside
  // "correct".
  noneButRecoveredCount: failures.noneButRecovered.length,
  // "None but wrong": labeled low-confidence AND the shipped results
  // don't contain the answer either — a genuinely hard case, but not a
  // false abstention, since nothing was withheld; a reasoning caller who
  // can't extract the answer from what shipped is in the same position a
  // human researcher would be in with a weak search result, not one who
  // was denied evidence.
  noneButWrongCount: failures.noneButWrong.length,
  falseAnswerCount: failures.falseAnswer.length,
  falseAnswerRate: +(failures.falseAnswer.length / questions.length).toFixed(4),
  correctCount: failures.correct,
};
console.log('\nSummary:', JSON.stringify(result, null, 2));

if (baselinePath && fs.existsSync(baselinePath)) {
  const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
  console.log('\nvs baseline:');
  console.log(`  withheld rate:     ${baseline.withheldRate} -> ${result.withheldRate}`);
  console.log(`  false-answer rate: ${baseline.falseAnswerRate} -> ${result.falseAnswerRate}`);
}

if (process.env.HARNESS_VERBOSE) {
  console.log('\nwithheld (true false-abstention):', JSON.stringify(failures.withheld, null, 2));
  console.log('\nnone but recovered:', JSON.stringify(failures.noneButRecovered, null, 2));
  console.log('\nnone but wrong:', JSON.stringify(failures.noneButWrong, null, 2));
  console.log('\nfalse answers:', JSON.stringify(failures.falseAnswer, null, 2));
}

await search.close();
