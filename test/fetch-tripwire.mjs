// Preloaded with --import under the CLI-facing suites in `npm test`, and
// passed on through NODE_OPTIONS to every process they spawn: any call to
// the runtime's own fetch is a pack or crawl read that bypassed nodeFetch,
// and fails the run even where the caller catches the error and falls back.

import { fileURLToPath } from 'node:url';

const self = `--import="${fileURLToPath(import.meta.url)}"`;
if (!(process.env.NODE_OPTIONS || '').includes(self)) {
    process.env.NODE_OPTIONS = [process.env.NODE_OPTIONS, self].filter(Boolean).join(' ');
}
const leaks = [];
globalThis.fetch = (input) => {
    const target = String(input?.url ?? input);
    leaks.push(target);
    process.stderr.write(`fetch tripwire: built-in fetch called for ${target}\n${new Error().stack.split('\n').slice(2, 6).join('\n')}\n`);
    throw new Error(`leaked to built-in fetch: ${target}`);
};
process.on('exit', (code) => {
    if (leaks.length && code === 0) {
        process.stderr.write(`fetch tripwire: ${leaks.length} call(s) leaked to built-in fetch\n`);
        process.exitCode = 1;
    }
});
