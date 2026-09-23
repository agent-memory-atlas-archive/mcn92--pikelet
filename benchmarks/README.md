# Benchmarks

Everything that measures Pikelet lives here. None of it runs under `npm test`;
each tree documents its own inputs and how to reproduce the numbers quoted in
the main README.

| Directory | What it measures | Reproduce |
| --- | --- | --- |
| `beir/` | Retrieval quality on BEIR datasets, as a ladder of configurations (dense at several quantization/rerank points, hybrid RRF, lexical-only, Arctic-XS encoder) so a change can be attributed to one component. Frozen config in `config.json`. | `beir/README.md` |
| `range-proof/` | The static-hosting claim: a `.pikelet` served by a server that only does `fs.createReadStream` + `Range`, mounted by the real reader and the real `pikelet mcp` server, with per-query bytes and request counts from the server's own log; failure modes (Range ignored, file replaced, truncated response); a headless-LLM loop. | `node benchmarks/range-proof/proof.mjs [pack]` and the other scripts in that directory (see `range-proof/README.md`) |
| `engine/` | The HNSW engine itself: recall/QPS Pareto frontiers against faiss, hnswlib and usearch, parameter sweeps, deletion/compaction recall, restore timing. Has its own `package.json` for the native baselines (`cd benchmarks/engine && npm install`); never a dependency of `pikelet-wasm`. | `npm run bench` (runs every `engine/*.js` via `engine/run.js`) or a single script, e.g. `node benchmarks/engine/pareto_frontier.js --dataset nytimes` |
| `results/` | Committed release runs of the engine suite (`pareto_*` CSV/JSON/PNG/log per dataset), with the run environment in `results/README.md`. Raw runs land in `results/raw/`, which is ignored. | — |

Datasets are downloaded or generated on demand and are ignored: `beir/cache/`
and `beir/work/`, `results/raw/`, and the dataset directories the engine
scripts fetch (`dbpedia/`, `sift/`, `glove/`, `nytimes/` at the repo root).
