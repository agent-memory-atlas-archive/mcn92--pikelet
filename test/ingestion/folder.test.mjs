// ingestFolder conformance: what the folder walk skips, it reports. A
// symlink (not followed) and an oversized file are both logged as warnings
// naming the path; nothing in the source tree disappears silently.
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const { ingestFolder } = await import(pathToFileURL(path.resolve(here, '..', '..', 'packages', 'pikelet', 'src', 'ingest.mjs')).href);

section('ingestFolder: skipped entries are reported');
{
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-ingest-folder-'));
  try {
    await fs.mkdir(path.join(root, 'docs'));
    await fs.writeFile(path.join(root, 'docs', 'real.md'), '# Real\n\nA paragraph of real content that is long enough to keep.\n');
    await fs.symlink(path.join(root, 'docs', 'real.md'), path.join(root, 'docs', 'linked.md'));
    await fs.symlink(path.join(root, 'docs'), path.join(root, 'docs-link'));
    // A sparse file: the size is what the cap reads; no bytes are written.
    const big = await fs.open(path.join(root, 'docs', 'huge.md'), 'w');
    await big.truncate(16 * 1024 * 1024 + 1);
    await big.close();

    const logs = [];
    const docs = await ingestFolder(root, {}, (line) => logs.push(line));
    check('the real file is ingested', docs.length === 1 && docs[0].sourcePath === 'docs/real.md', JSON.stringify(docs.map((d) => d.sourcePath)));
    check('a symlinked file is skipped with a warning naming it', logs.some((l) => l.includes('skipped symlink docs/linked.md')), JSON.stringify(logs));
    check('a symlinked directory is skipped with a warning naming it', logs.some((l) => l.includes('skipped symlink docs-link')), JSON.stringify(logs));
    check('a file over the per-file limit is skipped with a warning naming it and the limit',
      logs.some((l) => l.includes('skipped docs/huge.md') && l.includes('per-file limit')), JSON.stringify(logs));
    check('no other warnings are emitted', logs.length === 3, JSON.stringify(logs));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
}
