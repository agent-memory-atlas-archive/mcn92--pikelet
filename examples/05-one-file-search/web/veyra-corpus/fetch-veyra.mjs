#!/usr/bin/env node
// Fetches the compiled Veyra demo packs into ../public/ from the
// veyra-packs-v1 GitHub release, verifying each download against its
// pinned SHA-256 before writing it. Source for these packs is corpus/ and
// corpus-ablated/ (gen.mjs) and corpus-chamber43/ (make-chamber43.mjs) in
// this directory; the compiled .pikelet files themselves are not
// committed — each is a ~25MB self-calibrated inline-encoder artifact, and
// three of them is most of what used to make this repo's checkout heavy.
// Re-run is a no-op once the files are present; --force re-downloads.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(HERE, '..', 'public');
const RELEASE_BASE = 'https://github.com/mcn92/pikelet/releases/download/veyra-packs-v1';

const PACKS = [
  { name: 'veyra.pikelet', sha256: '4790eeca0b1c0b5010bc8e6471461d85b17c71a0f5ea89132d2224ac06fcbf10' },
  { name: 'veyra-ablated.pikelet', sha256: '57a33fb9b17fe7d1dc02946128192368a1c934ea403e8acebf79e275468298aa' },
  { name: 'veyra-chamber43.pikelet', sha256: '100f393d9de385f60f28aab91c41702ebdb7999b58f528b7d8f449dfb7f082b5' },
];

const force = process.argv.includes('--force');

function sha256OfFile(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

async function fetchOne({ name, sha256 }) {
  const dest = path.join(PUBLIC_DIR, name);
  if (!force && fs.existsSync(dest) && sha256OfFile(dest) === sha256) {
    console.log(`  ok (cached): ${name}`);
    return;
  }
  const url = `${RELEASE_BASE}/${name}`;
  console.log(`  fetching ${url}`);
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(`${name}: HTTP ${response.status} fetching ${url}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  const actual = crypto.createHash('sha256').update(bytes).digest('hex');
  if (actual !== sha256) {
    throw new Error(`${name}: SHA-256 mismatch — expected ${sha256}, got ${actual}`);
  }
  fs.mkdirSync(PUBLIC_DIR, { recursive: true });
  fs.writeFileSync(dest, bytes);
  console.log(`  ok: ${name} (${(bytes.length / 1e6).toFixed(1)} MB, verified)`);
}

async function main() {
  console.log('Fetching Veyra demo packs...');
  for (const pack of PACKS) await fetchOne(pack);
  console.log('Done.');
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
