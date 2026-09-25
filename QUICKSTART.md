# Pikelet Quick Start

Compile a corpus into one `.pikelet` file, query it from any JavaScript
runtime, and serve it to an LLM over MCP. The last section covers the engine
layer underneath, for when you already have vectors and want the index alone.

## Install

```bash
npm install -g pikelet          # the CLI: compile, mcp, doctor, create
npm install pikelet-wasm        # the library: readers and the engine
```

`compile` and `mcp` do not need the CLI's optional `@xenova/transformers`
dependency; skip its install with `npm install -g pikelet --omit=optional`
unless you also want the `create` scaffold path.

The CLI needs Node 20+. `pikelet-wasm` runs on Node 18+ (CI tests 18, 20 and
22), browsers, and Cloudflare Workers; this guide assumes `pikelet-wasm@0.8.0`
or later.

## Compile a pack

Point `compile` at a folder or a live site:

```bash
npx pikelet compile --source ./docs --out docs.pikelet
# or crawl a site:
npx pikelet compile --source https://docs.example.com --out docs.pikelet
```

```text
Ingested 3 docs -> 3 chunks
Embedded 3/3 chunks with inline transformer
Validated self-recall@1 on 3 sampled chunks
Built lexical index: 90 terms over 3 records (2 KiB)
Calibrated abstention: 42 answerable / 40 ablation + 0 entity-swap + 12 held-out-doc
  hard negatives / 81 off-domain / 24 gibberish, 5-fold CV AUC 1 vs hard negatives
Measured rerank operating point: C=3 (recall@3 1 over 3 corpus-embeddings queries)
Golden replay against the assembled artifact: 24/24 reproduce
Compiled docs.pikelet
  24.54 MB, 3 records, identity 223d1ec6...
```

The file carries the corpus records, the semantic index, a BM25 lexical
index, the inline MiniLM query encoder, integrity commitments, the
calibrated abstention threshold, and the evaluation fixtures. There is no
service, no model host and no vector database behind it.

Every pack is at least ~24 MB regardless of corpus size, because the query
encoder ships inside it. The first `compile` fetches that encoder (~25 MiB)
from a GitHub release and caches it; later runs are offline.

Useful flags: `--force` overwrites the output, `--max-pages` caps a crawl,
`--include`/`--exclude` filter a folder, and `--calibration-queries` supplies
real corpus-author questions used to validate the abstention fit and as the
pack's embedded golden queries. `pikelet compile --help` lists the rest,
including the `--encoder-*` flags for swapping the packaged MiniLM.

## Query it from code

```js
import { openPikeletFile } from 'pikelet-wasm/complete';

const pack = await openPikeletFile('docs.pikelet');

const out = await pack.query('how do workers restore snapshots', { k: 3 });
console.log(out.matchQuality, out.confidence, out.results[0].title);
// 'strong' 0.9999898531465357 'Snapshot restore'

await pack.close();
```

`query()` returns `matchQuality` — `'strong' | 'weak' | 'none' |
'unscored'` — alongside the results. An off-domain query abstains instead of
answering:

```js
const miss = await pack.query('what is the best pizza in Chicago', { k: 3 });
console.log(miss.matchQuality, miss.results.length);
// 'none' 0
```

That threshold was fit from the corpus at build time and stored in the file,
so the refusal behavior travels with the pack rather than living in your
application code.

### Mount a pack over HTTP

`openPikeletFile` also takes a URL. The reader fetches byte ranges instead of
downloading the artifact, so a static host that answers 206 is a complete
backend:

```js
const pack = await openPikeletFile('https://example.com/docs.pikelet');
```

Pin the publisher's identity when you mount someone else's pack; a mismatch
refuses to open:

```js
const pack = await openPikeletFile('https://example.com/docs.pikelet', {
  expectedIdentity: '223d1ec67145ec94e1237d284cdfdb4cecd6de91c5c0454f54d59674d6910a13',
});
```

Use `npx pikelet doctor <url>` to check whether a host serves ranges well
before you rely on it — it probes 206 support, cache-key ranges, HTTP/2,
ETags and RTT.

## Serve a pack to an LLM

`mcp` exposes packs over the Model Context Protocol on stdio, so an MCP
client can attach them as a retrieval tool:

```bash
npx pikelet mcp install --client claude-code --pack ./docs.pikelet
```

```text
Wrote MCP server "knowledge-packs" to ./.mcp.json
Claude Code picks it up on the next session in this project.
```

Any other MCP client can run `npx pikelet mcp --pack ./docs.pikelet`
directly, with no install step. The server offers four tools: `search`
(per-pack results with provenance and calibrated abstention), `list_packs`
(names and immutable identities, for citation pinning), `get_record`, and
`verify_pack` (replays the golden queries and abstention probes stored inside
the pack).

`--pack` takes a local file or an HTTP(S) URL, repeats for several packs, and
accepts `#<sha256>` to pin the manifest identity. `--shelf` mounts every pack
on a static `packs.json` listing.

## The engine layer

Everything above is the artifact layer. Underneath it is the HNSW engine,
which you can use directly when you already have vectors and want an index
rather than a queryable pack. At this layer you supply the embedding model,
the query vectors, and the metadata hydration yourself; `matchQuality`,
abstention, the lexical index and the corpus records are all artifact-layer
features that do not exist here.

Reach for it when you are embedding with your own model and only need
approximate nearest neighbors. If you have documents and want search, use
`compile` above instead.

`loadJsonFile()` and `loadSnapshotFile()` are Node-only; the rest of the
engine API also runs in browsers and Workers.

### From a repository checkout

```bash
git clone https://github.com/mcn92/pikelet.git
cd pikelet
npm install          # dev deps for the demos, benchmarks, and tests
```

No build step is needed: the checkout ships the prebuilt WASM engine in
`packages/pikelet-wasm/dist/` (`engine.{js,wasm}` plus the scalar fallback `engine.scalar.{js,wasm}`),
and the root entry points use it directly.

Rebuild the engine only if you are changing the C++ under `packages/pikelet-wasm/engine/`. That
requires an Emscripten toolchain (see the README's "Building from source"):

```bash
npm run build:all    # rebuilds packages/pikelet-wasm/dist/engine.* — engine development only
```

(`./build.sh` builds only the SIMD pair, `packages/pikelet-wasm/dist/engine.{js,wasm}`; use `build:all` when you also need the scalar fallback.)

### Pick an ingest path

Use the path that matches what you already have:

- Vectors already in memory: `Pikelet.fromVectors(...)`
- Vectors saved as JSON / JSONL: `Pikelet.loadJsonFile(...)` on the Node entrypoints
- Existing Pikelet snapshot on disk: `Pikelet.loadSnapshotFile(...)` on the Node entrypoints

If you are working from a repo checkout, `import Pikelet from 'pikelet-wasm'` already resolves to the in-tree package: the repository is an npm workspace, and `npm install` at the root links `node_modules/pikelet-wasm` to `packages/pikelet-wasm`. The entrypoints themselves live at `packages/pikelet-wasm/src/`.

### Local Node workflow

#### 1. Build an index from in-memory vectors

```js
import Pikelet from 'pikelet-wasm';

const rows = [
  { id: 'doc-1', vector: [1, 0, 0, 0] },
  { id: 'doc-2', vector: [0, 1, 0, 0] },
  { id: 'doc-3', vector: [0, 0, 1, 0] },
];

const { index, ids, idMap } = await Pikelet.fromVectors(rows, {
  metric: 'cosine',
  quantized: true,
});

const results = index.search(new Float32Array([1, 0, 0, 0]), 2);
console.log(results);
console.log(idMap.get(results[0].id)); // -> 'doc-1'
```

Use this path when your embedder already returns arrays or `Float32Array`s in the current process.

#### 2. Build an index from JSON or JSONL

On the Node.js entrypoints, Pikelet can load vectors directly from disk:

```js
import Pikelet from 'pikelet-wasm';

const { index, ids, idMap } = await Pikelet.loadJsonFile('vectors.jsonl', {
  metric: 'cosine',
  quantized: true,
  vectorKey: 'embedding', // default: 'vector'
  idKey: 'docId',         // default: 'id'
  maxFileBytes: 64 * 1024 * 1024,
});

const results = index.search(new Float32Array([1, 0, 0, 0]), 5);
console.log(results);
console.log(idMap.get(results[0].id));
```

Supported file types:

- `.json`
- `.jsonl`
- `.ndjson`

Accepted row shapes:

- `[1, 2, 3, ...]`
- `{ "id": "doc-1", "vector": [1, 2, 3, ...] }`
- `{ "docId": "doc-1", "embedding": [1, 2, 3, ...] }` with `idKey` / `vectorKey`

If your embedding pipeline already writes JSONL, this is the simplest file-based path.

#### 3. Export and restore a snapshot

If you want to reuse a built index later, export a snapshot:

```js
import fs from 'node:fs';
import Pikelet from 'pikelet-wasm';

const { index } = await Pikelet.fromVectors([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
], {
  metric: 'cosine',
  quantized: true,
});

// If you have deleted anything, compact() before export().
const snapshot = index.export();
fs.writeFileSync('index.pnck', snapshot);

const restored = await Pikelet.loadSnapshotFile('index.pnck', {
  dim: 4,
  maxElements: 2,
  metric: 'cosine',
  quantized: true,
});

console.log(restored.search(new Float32Array([1, 0, 0, 0]), 1));
```

Snapshot notes:

- `export()` throws if `ghostCount > 0`; call `compact()` first after deletions
- `loadSnapshotFile()` is Node-only
- `loadSnapshotFile()` restores Pikelet snapshots from disk, not arbitrary ANN binary formats

### Embedding your own documents

Pikelet does not care which embedder you use, as long as you end up with vectors.

Typical workflow:

1. Read your source documents.
2. Generate embeddings with your model or API of choice.
3. Store them either:
   - directly in memory and call `fromVectors()`, or
   - in JSON / JSONL and call `loadJsonFile()`
4. Query the built index with new query embeddings produced by the same model family.

If you already have parquet, numpy, or another upstream format, convert it into JSONL or feed the vectors into `fromVectors()` directly from your application code.

### Reference Worker example

This example predates the `.pikelet` artifact and lives under
`examples/legacy/`; a pack mounted over HTTP Range needs no Worker at all. It
is still the reference for exposing the raw engine over HTTP, and the
`/export`/`/import` contract below is still current, so it is documented here
rather than removed.

It is repo-based, loads the checked-in WASM engine artifacts, and exposes
Pikelet over HTTP. Treat it as a deployment pattern you run in your own
Cloudflare account, not as a centrally hosted Pikelet service.

When you run or deploy this example, it runs in your own Cloudflare environment:

- `npx wrangler dev` starts a local dev instance on your machine
- `wrangler deploy` publishes the Worker into the Cloudflare account authenticated in your local Wrangler setup
- any R2 bucket, auth settings, and rate limits belong to your own deployment, not to this repository

#### Run the Worker locally

```bash
cd examples/legacy/reference-worker
npx wrangler dev --port 8787 --var ALLOW_INSECURE_ADMIN:1
```

`ALLOW_INSECURE_ADMIN=1` is a local-only opt-in: without it (or an `API_KEY`), admin routes such as `/init`, `/add`, and `/import` return 403, so the curl examples below would be rejected.

#### Deploy the Worker to your own Cloudflare account

```bash
cd examples/legacy/reference-worker
wrangler r2 bucket create pikelet-indexes
wrangler deploy
```

#### Initialize a small index over HTTP

In another terminal:

```bash
curl -X POST http://localhost:8787/init \
  -H 'Content-Type: application/json' \
  -d '{
    "dims": 4,
    "maxElements": 16,
    "vectors": [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0]
    ]
  }'
```

Then query it:

```bash
curl -X POST http://localhost:8787/search \
  -H 'Content-Type: application/json' \
  -d '{"query": [1, 0, 0, 0], "k": 2}'
```

Useful local checks:

```bash
curl http://localhost:8787/health
curl http://localhost:8787/stats
```

#### Worker import / export contract

The Worker has its own binary envelope for `/export` and `/import`.

- `GET /export` returns a Worker snapshot blob
- `POST /import` expects a previous blob from that Worker `/export` path

Do not assume that a Node.js `index.export()` snapshot is interchangeable with the Worker `/import` format. The Worker wraps the engine snapshot with additional metadata for its own restore path.

#### Worker integration test

```bash
node test/test_worker_features.js
```

Starts a local `wrangler dev` instance against `examples/legacy/reference-worker` and runs a synthetic 1536D Worker/API integration test against it.

### Sizing and tuning

| Parameter | Typical range | Effect |
|-----------|---------------|--------|
| `M` | 8–32 | Graph connectivity. Higher improves recall, increases memory and build time. |
| `efConstruction` | 50–400 | Build beam width. Higher improves graph quality, slows build. |
| `efSearch` / `ef` | 50–500 | Query beam width. Higher improves recall, slows search. |
| `quantized` | `true` / `false` | uint8 quantization cuts memory significantly with some recall tradeoff. |
| `metric` | `'cosine'` / `'l2'` | Use cosine for normalized embeddings and L2 for unnormalized vectors. |

For memory-constrained Worker deployments, the quantized path is usually the right default.

## Troubleshooting

### `loadJsonFile()` rejects my file

Check:

- the extension is `.json`, `.jsonl`, or `.ndjson`
- the rows contain vectors under the expected field name
- `vectorKey` / `idKey` match your actual JSON schema

### I already have vectors in another format

Pikelet does not currently load formats like `.fvecs`, `.npy`, or `.parquet` directly.

Use one of these instead:

- convert the vectors to JSONL and call `loadJsonFile()`
- load them in your own code and pass them to `fromVectors()`

### Worker `/import` does not accept my local snapshot

The Worker `/import` route expects the Worker export format, not a raw local package snapshot. Use the Worker's own `/export` output when testing `/import`.

## Next steps

- [README.md](README.md) for what a `.pikelet` is and the engine API surface
- [packages/pikelet/README.md](packages/pikelet/README.md) for the full CLI reference: `compile`, `mcp`, `create`, `doctor`
- [spec/COMPLETE_PROFILE.md](spec/COMPLETE_PROFILE.md) for the artifact format
- [examples/legacy/reference-worker/README.md](examples/legacy/reference-worker/README.md) for the reference Worker deployment model
- [docs/architecture.md](docs/architecture.md) for the system design document
