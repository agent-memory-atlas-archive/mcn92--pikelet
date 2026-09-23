// Shared by the pikelet modules: package paths and version, config
// defaults, the model table, CliError, and the loaders that resolve pikelet-wasm
// (engine, artifact layer, complete profile). All three resolve the bare
// package: the workspace link inside this repository, the installed
// dependency for npm consumers.

import crypto from 'node:crypto';
import fssync from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest();
let completeModulesPromise = null;
function loadCompleteModules() {
  completeModulesPromise ??= (async () => {
    try {
      return {
        builder: await import('pikelet-wasm/complete/builder'),
        reader: await import('pikelet-wasm/complete'),
      };
    } catch (error) {
      throw new CliError(`could not resolve pikelet-wasm/complete; install pikelet-wasm >= 0.3 alongside pikelet. ${error.message.split('\n')[0]}`, 2);
    }
  })();
  return completeModulesPromise;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PACKAGE_ROOT = path.resolve(__dirname, '..');
const STUDENT_TRAINER = path.join(PACKAGE_ROOT, 'tools', 'train_student.py');
const OWN_PACKAGE = JSON.parse(fssync.readFileSync(path.join(PACKAGE_ROOT, 'package.json'), 'utf8'));
const CLI_VERSION = OWN_PACKAGE.version;
// The generated project must run its artifacts on a pikelet-wasm at least as
// new as the one that built them (range artifact format revisions are not
// readable by older readers), so it inherits this package's own range rather
// than carrying a second hardcoded one that drifts.
const PIKELET_WASM_RANGE = OWN_PACKAGE.dependencies['pikelet-wasm'];
const DEFAULT_PREFIX = 'Represent this sentence for searching relevant passages: ';
const CONFIG_SCHEMA_URL = 'https://raw.githubusercontent.com/mcn92/pikelet/main/pikelet/schemas/v1/pikelet.config.schema.json';
const DEFAULT_CONFIG = Object.freeze({
  chunking: { targetTokens: 256, overlapPercent: 15 },
  embedding: {
    mode: 'workers-ai',
    buildModel: 'bge-small-en-v1.5',
    dims: 384,
    prefixPolicy: { passage: '', query: DEFAULT_PREFIX },
    pooling: 'mean',
    normalize: true,
  },
  index: { metric: 'cosine', quantized: true, M: 16, efConstruction: 200, efSearch: 120 },
  runtime: { mode: 'snapshot', storage: 'bundled' },
  validation: { minRecallAt10: 0.98 },
});
const RANGE_ARTIFACT_MAGIC = 0x31415250;
const MODEL_MAP = Object.freeze({
  'bge-small-en-v1.5': {
    dims: 384,
    workersAiModel: '@cf/baai/bge-small-en-v1.5',
    hfModel: 'Xenova/bge-small-en-v1.5',
    maxInputTokens: 512,
  },
});
class CliError extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.exitCode = exitCode;
  }
}
async function loadArtifactContract() {
  let mod;
  try {
    mod = await import('pikelet-wasm/artifact');
  } catch (error) {
    throw new CliError(`could not resolve pikelet-wasm/artifact; install pikelet-wasm alongside pikelet. ${error.message.split('\n')[0]}`, 2);
  }
  const contract = mod.default || mod;
  if (typeof contract.buildSketchArtifactBytes === 'function') return contract;
  throw new CliError('pikelet-wasm/artifact does not expose buildSketchArtifactBytes; install pikelet-wasm >= 0.3 alongside pikelet.', 2);
}
async function loadPikelet() {
  return (await import('pikelet-wasm')).default;
}

export {
  sha256,
  completeModulesPromise,
  loadCompleteModules,
  __filename,
  __dirname,
  PACKAGE_ROOT,
  STUDENT_TRAINER,
  OWN_PACKAGE,
  CLI_VERSION,
  PIKELET_WASM_RANGE,
  DEFAULT_PREFIX,
  CONFIG_SCHEMA_URL,
  DEFAULT_CONFIG,
  RANGE_ARTIFACT_MAGIC,
  MODEL_MAP,
  CliError,
  loadArtifactContract,
  loadPikelet,
};
