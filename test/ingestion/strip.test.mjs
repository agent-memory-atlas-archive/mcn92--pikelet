// stripMarkdown conformance: markup is removed by position, content is
// never damaged. Regression for record text that shipped with hyphens,
// minus signs, underscores, and operators deleted (uint8_float_hnsw.hpp
// became "uint8 float hnsw.hpp"; "range = vmax - vmin" became "range =
// vmax vmin"). The stored record is evidence; it must read as the source.
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const { stripMarkdown } = await import(pathToFileURL(path.resolve(here, '..', '..', 'pikelet', 'src', 'ingest.mjs')).href);

section('stripMarkdown: content survives');
const keep = [
  ['snake_case identifier', 'call uint8_float_hnsw.hpp:213 and my_var', 'call uint8_float_hnsw.hpp:213 and my_var'],
  ['hyphenated words', 'Row-wise affine, degenerate-vector guard', 'Row-wise affine, degenerate-vector guard'],
  ['minus sign and exponent', 'range = vmax - vmin; if (range < 1e-30)', 'range = vmax - vmin; if (range < 1e-30)'],
  ['operators', 'x = a * b - c; if (a > b && q[0..D-1])', 'x = a * b - c; if (a > b && q[0..D-1])'],
  ['intra-word underscore is not emphasis', 'not_this_one stays', 'not_this_one stays'],
  ['fenced code verbatim, indentation kept', '```python\ndef f():\n    if x:\n        return a_b - 1  # not a heading\n```', 'def f():\n    if x:\n        return a_b - 1  # not a heading'],
  ['inline code contents kept', 'use `offset + scale * q[d]` here', 'use offset + scale * q[d] here'],
];
for (const [label, input, want] of keep) {
  const got = stripMarkdown(input);
  check(label, got === want, `got ${JSON.stringify(got)}`);
}

section('stripMarkdown: markup is removed');
const strip = [
  ['ATX heading hashes', '## Title ##\nbody', 'Title\nbody'],
  ['blockquote prefixes, nested', '> a\n>> b', 'a\nb'],
  ['list bullets', '- one\n* two\n+ three', 'one\ntwo\nthree'],
  ['horizontal rule', 'a\n\n---\n\nb', 'a\n\nb'],
  ['boundary emphasis', '**bold** and _em_ and ~~gone~~', 'bold and em and gone'],
  ['links keep text, drop url', 'see [the docs](http://x) and ![alt](i.png)', 'see the docs and alt'],
  ['table scaffolding', '| a | b |\n|---|---|\n| 1 | 2 |', 'a b\n1 2'],
  ['backslash escapes', 'a \\* b \\_ c', 'a * b _ c'],
];
for (const [label, input, want] of strip) {
  const got = stripMarkdown(input);
  check(label, got === want, `got ${JSON.stringify(got)}`);
}
