#!/usr/bin/env node
// Abstention-correctness regression harness: runs a labeled question set
// against a compiled .pikelet artifact and reports false-abstention and
// false-answer rates — the two failure modes calibration exists to avoid,
// and the numbers that must be measured against real questions, not just
// AUC on synthetic rows (see pikelet/src/calibrate.mjs's GROUNDING_FEAT
// design note for why that metric has misled this project more than once).
//
//   node scripts/calibration-harness.mjs <artifact.pikelet> <queries.json> [--baseline <baseline.json>]
//       [--retrieval hybrid|vector|lexical|augmented]
//       [--rank-depth N] [--rank-modes hybrid,vector,lexical] [--rank-report <out.json>]
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
const { loadQuestions, isAnswerable, answerPresent, recordMatchesId, rankIn } = await import(path.join(ROOT, 'scripts', 'lib', 'relevance-sets.mjs'));

function extractFlag(args, name) {
  const idx = args.indexOf(name);
  if (idx === -1) return [args, null];
  return [args.filter((_, i) => i !== idx && i !== idx + 1), args[idx + 1]];
}

let args = process.argv.slice(2);
let baselinePath, retrievalMode, rankDepthArg, rankModesArg, rankReportPath;
[args, baselinePath] = extractFlag(args, '--baseline');
[args, retrievalMode] = extractFlag(args, '--retrieval');
[args, rankDepthArg] = extractFlag(args, '--rank-depth');
[args, rankModesArg] = extractFlag(args, '--rank-modes');
[args, rankReportPath] = extractFlag(args, '--rank-report');
const [artifactPath, queriesPath] = args;
if (!artifactPath || !queriesPath) {
  console.error('usage: node scripts/calibration-harness.mjs <artifact.pikelet> <queries.json> [--baseline <baseline.json>] [--retrieval hybrid|vector|lexical|augmented]\n'
    + '         [--rank-depth N] [--rank-modes hybrid,vector,lexical] [--rank-report <out.json>]');
  process.exit(1);
}
// --rank-depth N adds a retrieval-side report: for every answerable
// question and each mode in --rank-modes, the rank (1..N) of the first
// result carrying the evidence, or null when it is not in the top N. That
// sorts the misses by what could fix them: 'gap' (no mode reaches the
// evidence within N — only the index contents or the encoder can move
// it), 'range' (reached, but deeper than 3 — a reranker's territory),
// 'demotion' (vector or lexical has it in the top 3 and hybrid fusion
// pushed it out — a fusion-logic fix), 'top3' (fine).
const rankDepth = rankDepthArg ? Number(rankDepthArg) : 0;
if (rankDepthArg && (!Number.isInteger(rankDepth) || rankDepth < 3)) {
  console.error(`--rank-depth must be an integer >= 3, got ${rankDepthArg}`);
  process.exit(1);
}
const rankModes = (rankModesArg || 'hybrid,vector,lexical').split(',').map((m) => m.trim()).filter(Boolean);
for (const m of rankModes) {
  if (!['hybrid', 'vector', 'lexical', 'augmented'].includes(m)) {
    console.error(`--rank-modes entries must be hybrid, vector, lexical, or augmented, got ${m}`);
    process.exit(1);
  }
}
if (retrievalMode && !['hybrid', 'vector', 'lexical', 'augmented'].includes(retrievalMode)) {
  console.error(`--retrieval must be hybrid, vector, lexical, or augmented, got ${retrievalMode}`);
  process.exit(1);
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

if (rankDepth) {
  // Evidence rank per mode, searched to rankDepth. The sketch sizes its
  // candidate pool as max(k, the artifact's recommendedRerank), so k alone
  // widens the pool to the depth without shrinking a larger default.
  // showAbstained so the verdict never hides the list.
  const rerank = `max(${rankDepth}, artifact default)`;
  const answerable = questions.filter(isAnswerable);
  // Evidence per result: the named source record when the set gives ids,
  // otherwise the quote/answer text check the verdict pass uses.
  const rows = [];
  for (const q of answerable) {
    const ranks = {};
    let allWithin = null;
    for (const mode of rankModes) {
      // The serving path first (k=3, the verdict pass's own call): the
      // sketch's candidate pool grows with k, so a deep search can shift
      // the top few by one and turn a served rank 3 into a reported 4.
      const served = await search.query(q.text, { k: 3, retrieval: mode, showAbstained: true });
      let rank = rankIn(served.results, q);
      if (rank === null) {
        const deep = await search.query(q.text, { k: rankDepth, retrieval: mode, showAbstained: true });
        rank = rankIn(deep.results, q);
        if (mode === 'hybrid' && q.evidenceIds.length > 1) {
          allWithin = q.evidenceIds.every((id) => deep.results.some((r) => recordMatchesId(r, id)));
        }
      } else if (mode === 'hybrid' && q.evidenceIds.length > 1) {
        const deep = await search.query(q.text, { k: rankDepth, retrieval: mode, showAbstained: true });
        allWithin = q.evidenceIds.every((id) => deep.results.some((r) => recordMatchesId(r, id)));
      }
      ranks[mode] = rank;
    }
    const found = Object.values(ranks).filter((r) => r !== null);
    const best = found.length ? Math.min(...found) : null;
    const hybrid = ranks.hybrid ?? best;
    const others = Object.entries(ranks).filter(([m]) => m !== 'hybrid').map(([, r]) => r).filter((r) => r !== null);
    let bucket;
    if (best === null) bucket = 'gap';
    else if (hybrid !== null && hybrid <= 3) bucket = 'top3';
    else if (others.length && Math.min(...others) <= 3) bucket = 'demotion';
    else bucket = 'range';
    rows.push({ id: q.id, cls: q.cls, q: q.text, ranks, best, bucket, ...(allWithin === null ? {} : { allEvidenceWithinDepth: allWithin }) });
  }

  const at = (rs, mode, n) => rs.filter((r) => r.ranks[mode] !== null && r.ranks[mode] <= n).length;
  const classes = [...new Set(rows.map((r) => r.cls))];
  console.log(`\nEvidence rank to depth ${rankDepth} (rerank ${rerank}), answerable questions only:`);
  if (rankDepth >= info.records) {
    console.log(`  note: depth ${rankDepth} covers the whole corpus (${info.records} records); 'gap' cannot occur — use a smaller depth for a reachability measurement.`);
  }
  console.table(classes.flatMap((cls) => {
    const rs = rows.filter((r) => r.cls === cls);
    return rankModes.map((mode) => ({
      cls, mode, n: rs.length, '@1': at(rs, mode, 1), '@3': at(rs, mode, 3), [`@${rankDepth}`]: at(rs, mode, rankDepth),
    }));
  }));
  console.log('Miss buckets (by hybrid rank; gap = no mode within depth, demotion = another mode has it in the top 3):');
  console.table(classes.map((cls) => {
    const rs = rows.filter((r) => r.cls === cls);
    const count = (b) => rs.filter((r) => r.bucket === b).length;
    const multi = rs.filter((r) => r.allEvidenceWithinDepth !== undefined);
    return {
      cls, n: rs.length, top3: count('top3'), demotion: count('demotion'), range: count('range'), gap: count('gap'),
      ...(multi.length ? { [`allEvidence@${rankDepth}`]: `${multi.filter((r) => r.allEvidenceWithinDepth).length}/${multi.length}` } : {}),
    };
  }));
  const misses = rows.filter((r) => r.bucket !== 'top3');
  if (misses.length) {
    console.log('Misses:');
    for (const r of misses) {
      const rk = rankModes.map((m) => `${m[0]}=${r.ranks[m] ?? `>${rankDepth}`}`).join(' ');
      console.log(`  ${r.bucket.padEnd(8)} ${String(r.id).padEnd(5)} ${rk.padEnd(24)} ${r.q.slice(0, 80)}`);
    }
  }
  if (rankReportPath) {
    fs.writeFileSync(rankReportPath, JSON.stringify({
      artifact: path.basename(artifactPath), identity: info.identity, depth: rankDepth, rerank, modes: rankModes, rows,
    }, null, 2));
    console.log(`rank report written to ${rankReportPath}`);
  }
}

if (process.env.HARNESS_VERBOSE) {
  console.log('\nwithheld (true false-abstention):', JSON.stringify(failures.withheld, null, 2));
  console.log('\nnone but recovered:', JSON.stringify(failures.noneButRecovered, null, 2));
  console.log('\nnone but wrong:', JSON.stringify(failures.noneButWrong, null, 2));
  console.log('\nfalse answers:', JSON.stringify(failures.falseAnswer, null, 2));
}

await search.close();
