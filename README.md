# Pikelet

**Knowledge that ships as a file.**

Pikelet compiles a corpus into one self-contained, queryable artifact. A model can interrogate a 456,153-record knowledge base whose backend is a static file.

A `.pikelet` carries the source text, semantic and keyword indexes, the query encoder, retrieval calibration, evaluation fixtures, and integrity commitments over all of them. So it is more than stored search data: the file fixes how its corpus is queried — the model that turns a question into a vector, how candidates are ranked, and when the honest answer is "this corpus doesn't say" — and its identity is a hash that commits to all of it. **Pikelet makes retrieval semantics part of the versioned knowledge artifact.**

Put the file on disk, S3, R2, a CDN, or any static HTTP host. A reader can mount it locally or over HTTP Range and search it without a vector database, embedding API, or retrieval server. Updates ship as small immutable layers, and an answer can name the exact version of the knowledge it came from.

The `pikelet` CLI requires Node 20+; the `pikelet-wasm` library runs on Node 18+ (CI tests 18, 20, 22), browsers, and Cloudflare Workers.

```bash
npx pikelet compile --source ./docs --out docs.pikelet
```

```text
Ingested 3 docs -> 3 chunks
Embedded 3/3 chunks with inline transformer
Built complete .pikelet artifact with 24.5 MB
Compiled docs.pikelet
  24.5 MB, 3 records, identity 8d731a...
```

First run fetches the ~25 MiB query encoder from a GitHub release and caches it; every `.pikelet` file is at least that size regardless of corpus, because the encoder ships inside it — a 5-file folder and a 500-file folder both start around 25 MiB.

Then query it from an LLM — Claude Code, Claude Desktop, or any MCP client:

```bash
npx pikelet mcp install --client claude-code --pack ./docs.pikelet
# or --client claude-desktop; any other MCP client can run
# `npx pikelet mcp --pack ./docs.pikelet` directly, no install step
```

```text
Wrote MCP server "knowledge-packs" to ./.mcp.json
Claude Code picks it up on the next session in this project.
```

Claude now has a `search` tool over your docs. Or query it directly from code:

```js
import { openPikeletFile } from 'pikelet-wasm/complete';
const pack = await openPikeletFile('docs.pikelet');
const out = await pack.query('how do workers restore snapshots', { k: 5 });
console.log(out.matchQuality, out.results[0]?.title);
// 'strong' 'Snapshot restore'
```

That's the whole loop. `compile` also takes a live URL (`--source https://docs.example.com`) instead of a directory. If you want a deployed search app — a Worker + UI, not a file — use `npx pikelet create` instead; see [`packages/pikelet/README.md`](packages/pikelet/README.md) for the full CLI reference and the tradeoffs between the two. To install rather than use `npx`, run `npm install -g pikelet --omit=optional`: the optional dependency is only for `create` ([why it matters](packages/pikelet/README.md#installing)).

Under the hood, the file is one container for everything a reader needs:

```text
documents
    │
    ▼
┌──────────────────┐
│   docs.pikelet   │
│                  │
│ corpus           │
│ semantic index   │
│ lexical index    │
│ query encoder    │
│ integrity        │
│ calibration      │
│ evaluation       │
└──────────────────┘
    │
    ├── local file
    └── static HTTP / object storage
              │
              ▼
        browser / Node / edge / agent
```

Search is the interface. **The file is the knowledge deployment unit.**

---

## The shortest demonstration

A real pack holds **456,153 Simple English Wikipedia records** in a **648.5 MiB** file. The [live browser demo](https://pikelet-wiki-playground.pages.dev/) is static HTML on Cloudflare Pages that mounts it straight from an R2 bucket — there is no `/search`, `/embed`, `/query` or `/api` behind it. The browser range-reads the file, verifies it, loads the query encoder from inside it, and shows every byte range it fetches.

For the agent test, the same file was served by a server whose only relevant capability was `fs.createReadStream` + HTTP Range and mounted by URL into a headless Claude restricted to the pack's three MCP tools (no web, no filesystem). The model answered questions such as "What causes earthquakes?" by searching, reading several records, and citing them.

| In one persistent session                | Transferred                 |
| ---------------------------------------- | --------------------------: |
| Mount (once)                             | 51.7 MiB in 11 requests     |
| Query encoder (once, with the first query) | ~25 MiB                   |
| Each fresh query after that              | ~0.5–1.1 MiB                |
| Whole session: 5 queries and a repeat    | 80.8 MiB, 12.5% of the file |

The complete artifact is never downloaded. The full transcript, the per-query table, and what a cold process per question costs instead are in [the Wikipedia demonstration](docs/wikipedia-demonstration.md); [`benchmarks/range-proof/`](benchmarks/range-proof/) reproduces it.

---

## Attach a pack to an LLM

`pikelet mcp` exposes one or more packs through the Model Context Protocol:

```bash
npx pikelet mcp install \
  --client claude-code \
  --pack https://example.com/docs.pikelet#<sha256>
```

The agent gets `search`, `get_record`, `list_packs`, `verify_pack`. A search result carries the identity of the pack and the location of the source record, so an agent can work against `product-docs.pikelet`, `rust-reference.pikelet`, `policy-2026-09.pikelet`, `customer-manual-v4.pikelet` without each publisher operating a retrieval API. The artifact can be local, private, public, behind authenticated object storage, or distributed like any other static asset.

A pack mounted with a content hash has a stable identity — `https://example.com/docs.pikelet#8d731...` — so "what body of knowledge did this agent query?" has a reproducible answer. A citation is `(pack identity, record id)`, and it stays meaningful as the pack changes: a chain of layers resolves citations made against any of its earlier versions, and a pack compacted from a chain translates them forward, reporting what an old record became or that it was deleted (the old record's bytes stay in the old files). Citation translation is a library call today, `citation(identity, id)`; the MCP tools do not expose it yet.

**A mounted pack's content reaches the model as tool output.** `verify_pack` proves the bytes are intact and match their pinned identity; it does not prove the corpus itself is trustworthy. Mounting a pack from a source you don't control is the same trust decision as giving an agent any other untrusted-content tool — treat pack text the way you'd treat search results or fetched web pages, not as instructions.

**Try it with your own questions.** The three Veyra packs from the [ablation](docs/veyra-ablation.md) are published as pinned GitHub release assets. Fetch and verify them first:

```bash
npm run demo:veyra
```

Then mount all three through Pikelet's MCP server:

```bash
claude -p "your question here" \
  --mcp-config examples/one-file-search/web/public/veyra.mcp.json \
  --strict-mcp-config \
  --allowedTools "mcp__veyra-demo__search,mcp__veyra-demo__list_packs,mcp__veyra-demo__get_record"
```

For an interactive session, use the same config without `-p`:

```bash
claude \
  --mcp-config examples/one-file-search/web/public/veyra.mcp.json
```

All three packs derive from the same small synthetic Station Veyra corpus, with controlled differences in the Tovash evidence. Ask where the Tovash project is housed: the full pack supports Chamber 17, the modified pack supports Chamber 43, and the ablated pack should not support any chamber.

---

## Update a pack without rebuilding it

A pack doesn't have to be recompiled when a few documents change. `append` never modifies its parent: it writes a new, small `.pikelet` — a **layer** — holding new records, deletions, or replacements, and the base plus its layers (a chain) is searched as one pack. Every published file stays byte-for-byte immutable; the newest layer defines the current state, and its identity pins the whole history.

```bash
npx pikelet append --parent docs.pikelet --source ./changed-docs --out docs.0001.pikelet
npx pikelet append --parent docs.0001.pikelet --remove 412 --out docs.0002.pikelet
npx pikelet mcp --pack docs.0002.pikelet
```

Layers are small. On a 20-record test pack with a 24.6 MiB base, deleting a record took a 1.2 KiB layer and replacing one took 6.9 KiB; a layer that adds records also carries a vocabulary filter sized to its base's (about 43 KiB on a 3,255-record docs pack).

A layer records its parent's identity, and its location when the parent sits in the same directory, so mounting the newest layer finds the rest of the chain — locally or over HTTP — and verifies every link. `#<sha256>` on the newest layer pins the whole chain. Layers always inherit the base's encoder. They inherit its calibration only while the chain stays within the drift limit (by default, records added plus deleted up to 20% of the base's); past it, the chain serves `unscored` until `compact` folds it into one fresh base. `rebase` moves a layer onto a different parent. The format is specified in [`LAYERED_PROFILE.md`](LAYERED_PROFILE.md); `npx pikelet --help` lists every flag.

## Match quality and abstention

A nearest neighbour is not automatically evidence that a corpus covers a question. At build time Pikelet fits a classifier over retrieval signals — distances, and how much of the question the top passages actually contain — against queries the corpus does not cover ([details](docs/how-a-query-runs.md#how-abstention-is-decided)), and results carry `matchQuality: strong | weak | none`. It measures retrieval relevance: `none` means the pack does not cover the question's topic; `strong` means it does, not that a returned passage states the specific fact asked for. For that, each result lists the question's words no top passage contains (`grounding.uncovered`), and an asked-for value in that list is unsupported whatever `matchQuality` says. When calibration can't separate on-topic from off-topic queries reliably, Pikelet reports `matchQuality: unscored` and records why calibration was skipped, rather than manufacturing confidence. In the library, a `none` verdict withholds `results` by default; pass `query(text, { showAbstained: true })` to see the raw retrieval anyway — `matchQuality` and `confidence` are unaffected either way. MCP `search` does the opposite: it returns results under `none` with a note telling the model the support is indirect, because the calibrator can misjudge a paraphrase; `showAbstained: false` withholds them.

`matchQuality` is evidence about retrieval support. It is **not** a guarantee that an LLM will never hallucinate.

---

## Deeper dives

- [The Wikipedia demonstration](docs/wikipedia-demonstration.md) — the full agent transcript, the per-query network table, and what cold processes cost.
- [How a `.pikelet` query runs](docs/how-a-query-runs.md) — what is inside the file, what a query costs on the wire, remote execution, the integrity checks on the read path, and how abstention is decided.
- [The Veyra ablation](docs/veyra-ablation.md) — what happens to answers when the supporting record is edited or removed from the pack, with the reproduction commands.
- [Why make knowledge a file](docs/why-a-file.md) — the reasoning behind the single-artifact design, and what it commits to.
- [Architecture](docs/architecture.md) — the engine, the C ABI and the JavaScript wrapper; the artifact formats are specified in [`spec/`](spec/).

## What this is not

**Not a claim that vector databases are obsolete.** If your corpus changes continuously, needs transactional updates, serves many tenants, or already lives comfortably in a database, use a database. Pikelet targets `build → publish → query many times → replace on release`, with small layered updates in between.

**Not state-of-the-art embedding research.** The bundled model is MiniLM-L6, chosen for being small enough to live inside the artifact. ArguAna demonstrates a real quality cost from the compact encoder.

**Not "LLMs can no longer hallucinate."** The artifact exposes retrieved evidence, provenance, integrity, and an explicit support signal. Whether an agent obeys those signals is an agent behavior question.

**Not a new HNSW algorithm.** HNSW is HNSW. HTTP Range is HTTP Range. BM25 is BM25. MiniLM is MiniLM. Affine quantization is not new either. The project is about what becomes possible when those pieces are arranged around one constraint: **the knowledge base itself must be distributable as a file and remain useful without dedicated retrieval infrastructure.**

---

## When Pikelet is a good fit — and when it isn't

**Use it when:**

- your corpus is static or changes on a release cycle
- you want semantic search without operating search infrastructure
- you want search inside a browser, Node process, edge worker, Electron app, or offline tool
- you want to distribute a searchable corpus to someone else
- you want an LLM agent to query a frozen, versioned body of knowledge
- static/object storage is easier to deploy than a database
- provenance and reproducibility matter
- a content-addressed knowledge snapshot is useful

Examples: documentation, SDK/API references, technical manuals, research corpora, standards, legal texts, books, product knowledge, code/documentation snapshots, evaluation corpora, agent reference packs.

**Don't use it if:**

- your corpus has heavy online writes
- you need transactional mutation
- you need a multi-tenant authoritative search service
- you need exact nearest-neighbour search at very large scale
- your corpus is so large that a linear resident sketch scan is inappropriate
- you already operate a vector database happily and portability buys you nothing

The resident remote scan is linear in row count. The current architecture targets corpora through the low millions of records, not arbitrary web scale.

---

## Repository map

```text
packages/pikelet-wasm/            The published pikelet-wasm package:
  engine/                          C++ vector engine (HNSW, float32 and
                                   affine-u8 backends, mutation, compaction,
                                   snapshot import/export), compiled to WASM
                                   in dist/. Used at compile time to build and
                                   quantize the index; query time does not load
                                   it (see complete/index.mjs's header comment)
                                   — .pikelet reads run a pure-JS sketch scan,
                                   optionally SIMD-accelerated.
  src/                             Engine wrapper and artifact readers/builders
                                   (the entrypoints, core/, artifact/, errors/)
  complete/                        Reader and builder for the complete
                                   range-readable .pikelet artifact, and the
                                   layered-chain reader and producers
  native/                          N-API build of the same engine (benchmarks)
packages/pikelet/                 The published pikelet CLI: compiler, MCP
                                   server, encoder integration, Docusaurus plugin
docs/                             Deeper dives (how a query runs, the Veyra
                                   ablation, why a file), architecture notes,
                                   rename history, measurement reports
spec/                             Byte-level artifact contracts
LAYERED_PROFILE.md                The layered profile: append, compact,
                                   rebase, and how chains are read
benchmarks/beir/                       Frozen BEIR ablation harness (encoder,
                                   quantization, HNSW quality)
benchmarks/range-proof/                The deliberately boring static-HTTP proof:
                                   dumb-server.mjs, proof.mjs, mcp-proof.mjs,
                                   llm-proof.mjs, failure-modes.mjs
examples/one-file-search/      Large single-artifact search and embedded
                                   encoder work
examples/mcp-knowledge-pack/   Compile, mount, search, and hydrate
                                   records through MCP
local-packs/                      Prebuilt example .pikelet artifacts used
                                   by the demo commands above
packs/                            Pack hosting and distribution examples
```

---

## Reproduce the proofs

**Range proof** (`benchmarks/range-proof/`) is built around a server that does not understand Pikelet. It checks: static HTTP + Range + Pikelet client = remote semantic retrieval, without a full download. The main script reports artifact size, records, mount bytes/requests, per-query bytes/requests, cache behavior. `failure-modes.mjs` independently exercises Range supported / ignored / tampered bytes / truncated response. `mcp-proof.mjs` and `llm-proof.mjs` use a real MCP client and a real headless Claude process rather than a mocked model response.

**BEIR evaluation** (`benchmarks/beir/`) keeps stages separate so a quality change can be attributed to the component that caused it, plus a standalone conformance fixture checking the JS benchmark path against the actual C++ affine-u8 representation. Don't read the latency columns as a universal native-performance comparison — they measure the benchmark paths under the stated harness. The quality deltas are the important part.

---

## Status

Pikelet is early. The implementation is real; the format is not frozen. One primary author. Draft 2 artifact format.

- prebuilt WASM included
- local and HTTP readers
- affine-u8 and float HNSW backends for compile-time indexing/quantization
- embedded query encoder
- BM25 hybrid retrieval
- content identity and lazy-read integrity verification
- calibration/abstention support
- MCP mounting
- layered updates: append, compact, rebase
- BEIR retrieval evaluation
- range-read and failure-mode tests
- a large 456k-record example artifact

`npm test` exercises the engine, artifact profiles, MCP, ingestion, format hardening, and related conformance suites.

Previously known as **Pancake** (renamed September 2026); artifacts, profile strings and files from before the rename remain readable — see [`docs/history.md`](docs/history.md) for what kept the old name and why. Pikelet is unrelated to the pre-existing Pikelet programming language.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
