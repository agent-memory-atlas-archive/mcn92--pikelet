#!/usr/bin/env node
// Static-hosting proof: query a 648 MB .pikelet mounted directly from a
// GitHub release asset (plain static HTTP, no application server) and show
// that traffic is nothing but byte-range GETs against that one file.
//
//   node examples/one-file-search/remote-demo.mjs "question one" "question two"
//
// With no arguments, asks two default questions. Runs two queries
// deliberately: the first pays the real cost of fetching the lexical
// segment's posting lists for whatever common words it contains; every
// later query that reuses those cached postings is far cheaper. Showing
// only a single cherry-picked cheap query would be misleading — the honest
// and, once you see the numbers, more interesting claim is "first query
// costs real bytes, every query after is nearly free."
//
// Screen-recording notes: the live counter line updates in place (\r), so
// a terminal recorder shows it ticking up during each query, not scrolling
// text.

import { openPikeletFile } from '../../complete/index.mjs';
import { httpRangeSource } from '../../complete/sources.mjs';

const URL_ = process.env.PIKELET_WIKI_URL
    || 'https://github.com/mcn92/pikelet/releases/download/wikipedia-pack-v1/wikipedia.pikelet';
const argQuestions = process.argv.slice(2);
const questions = argQuestions.length ? argQuestions
    : ['who was the first person on the moon', 'what causes lightning'];

let requests = 0;
let bytes = 0;
const origFetch = globalThis.fetch;
globalThis.fetch = async (url, opts) => {
    const range = opts?.headers?.Range || opts?.headers?.range;
    const response = await origFetch(url, opts);
    // Only count real range reads against the pack itself — not the
    // redirect hop GitHub inserts to resolve the signed blob URL.
    if (range) {
        requests++;
        const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range) || [];
        if (a !== undefined) bytes += Number(b) - Number(a) + 1;
        process.stdout.write(
            `\r  requests: ${String(requests).padStart(4)}   bytes read: ${formatBytes(bytes).padStart(10)}` + ' '.repeat(10),
        );
    }
    return response;
};

function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

console.log(`Pack:  ${URL_}`);
console.log('There is no search server at this URL — it is a static file.');
console.log('');

const source = httpRangeSource(URL_);
const search = await openPikeletFile(source);
const fileBytes = (await search.info()).fileBytes;
console.log(`Mounted. ${(fileBytes / 1024 ** 2).toFixed(1)} MB, no application server involved.`);

const totals = [];
for (const [i, question] of questions.entries()) {
    console.log('');
    console.log(`> ${question}`);
    console.log('');
    const before = { requests, bytes };
    const out = await search.query(question, { k: 3 });
    process.stdout.write('\n\n');
    const used = { requests: requests - before.requests, bytes: bytes - before.bytes };
    totals.push(used);

    console.log(`matchQuality: ${out.matchQuality}   confidence: ${out.confidence?.toFixed(3)}`);
    for (const r of out.results) {
        console.log(`  - ${r.title}  (${r.sourcePath || r.url || ''})`);
    }
    console.log('');
    const pct = ((used.bytes / fileBytes) * 100).toFixed(3);
    console.log(`this query: ${used.requests} range reads, ${formatBytes(used.bytes)} — ${pct}% of the file`);
    if (i === 0 && questions.length > 1) {
        console.log('(first query — pays to fetch the lexical index'
            + "'s posting lists for whatever common words it used; later queries reuse them)");
    }
}

console.log('');
console.log('---');
if (totals.length > 1) {
    const first = totals[0];
    const rest = totals.slice(1);
    const restBytes = rest.reduce((s, t) => s + t.bytes, 0) / rest.length;
    console.log(`first query: ${formatBytes(first.bytes)}.   `
        + `later queries averaged ${formatBytes(restBytes)} — `
        + `${(first.bytes / Math.max(restBytes, 1)).toFixed(1)}x cheaper once warm.`);
}
console.log(`mount + all queries: ${requests} requests, ${formatBytes(bytes)} of a ${(fileBytes / 1024 ** 2).toFixed(0)} MB file.`);
console.log('No /search. No /embed. No /query. No /api. Only bytes=X-Y against one file.');

await search.close();
