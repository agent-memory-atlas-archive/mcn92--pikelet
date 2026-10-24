#!/usr/bin/env node
import { main } from '../src/cli.mjs';
import { nodeSupportWarning } from '../src/node-support.mjs';

// stderr, so it never mixes into `pikelet mcp`'s protocol stream on stdout.
const nodeWarning = process.env.PIKELET_NO_NODE_WARNING ? null : nodeSupportWarning();
if (nodeWarning) console.error(nodeWarning);

main(process.argv).catch((error) => {
  const code = Number.isInteger(error?.exitCode) ? error.exitCode : 2;
  console.error(error?.message || String(error));
  process.exit(code);
});
