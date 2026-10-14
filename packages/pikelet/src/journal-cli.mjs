// `pikelet journal <init|list|compact>`: the user's side of the journal
// (docs/agent-journal.md). Agents write through MCP; people create, read and
// fold journals here.

import { CliError } from './common.mjs';
import {
  NOTE_KINDS, initJournal, compactJournal, openJournalChain, latestNotes,
} from './journal.mjs';

export async function runJournalCommand(positionals, flags) {
  const [sub, target] = positionals;
  if (!sub || !['init', 'list', 'compact'].includes(sub)) {
    throw new CliError('usage: pikelet journal <init|list|compact> <name|dir>');
  }
  if (!target) throw new CliError(`pikelet journal ${sub} needs a journal name or directory`);
  if (sub === 'init') {
    await initJournal({ nameOrPath: target, force: flags.force === true });
    return;
  }
  if (sub === 'compact') {
    await compactJournal(target);
    return;
  }
  // list
  if (flags.kind !== undefined && !NOTE_KINDS.includes(flags.kind)) {
    throw new CliError(`--kind must be one of ${NOTE_KINDS.join(', ')}`);
  }
  const limit = flags.limit === undefined ? 20 : Number(flags.limit);
  if (!Number.isSafeInteger(limit) || limit < 1) throw new CliError('--limit must be a positive integer');
  const { dir, chain } = await openJournalChain(target);
  try {
    const notes = await latestNotes(chain, { kind: flags.kind, tag: flags.tag, limit });
    const info = chain.info();
    console.log(`${dir}: ${notes.length} note(s) shown, ${info.liveRecords - 1} in the journal, identity ${info.identity.slice(0, 12)}…`);
    for (const r of notes) {
      const n = r.note;
      console.log(`\n#${r.id}  ${n.created}  ${n.kind}  ${r.title}`);
      console.log(`    by ${n.authoredBy?.client ?? 'unknown'}${n.tags?.length ? `  tags: ${n.tags.join(', ')}` : ''}`);
      const body = String(r.text || '').split('\n').slice(1).join(' ');
      console.log(`    ${body.slice(0, 240)}${body.length > 240 ? '…' : ''}`);
    }
  } finally {
    await chain.close();
  }
}
