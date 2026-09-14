// Derives corpus-chamber43/ from corpus/ (run gen.mjs first): the full
// Veyra corpus with exactly one fact edited so it disagrees with itself —
// loc-tovash.md now claims Chamber 43 instead of Chamber 17, the answer
// every other record and cross-reference still implies. Used by the
// ablation demo to show a single contradicted record instead of a missing
// one (compare against veyra-ablated.pikelet, which removes facts rather
// than corrupting one).
import fs from 'node:fs';
import path from 'node:path';

const OUT = path.dirname(new URL(import.meta.url).pathname);
const SRC = path.join(OUT, 'corpus');
const DST = path.join(OUT, 'corpus-chamber43');

if (!fs.existsSync(SRC)) {
  console.error('corpus/ not found — run gen.mjs first');
  process.exit(1);
}

fs.rmSync(DST, { recursive: true, force: true });
fs.mkdirSync(DST, { recursive: true });
for (const name of fs.readdirSync(SRC)) {
  const text = fs.readFileSync(path.join(SRC, name), 'utf8');
  const patched = name === 'loc-tovash.md'
    ? text.replace(/Chamber 17/g, 'Chamber 43')
    : text;
  fs.writeFileSync(path.join(DST, name), patched);
}
console.log(`corpus-chamber43/ written (${fs.readdirSync(DST).length} facts, loc-tovash.md edited)`);
