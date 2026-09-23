# Pikelet

**Knowledge that ships as a file.**

Pikelet compiles a corpus into one self-contained, queryable artifact.

A model can interrogate a 456,153-record knowledge base whose backend is a static file.

A `.pikelet` can carry the source text, semantic index, keyword index, query encoder, integrity commitments, retrieval calibration, and evaluation fixtures needed to interrogate that corpus. Put the file on disk, S3, R2, a CDN, or any static HTTP host. A reader can mount it locally or over HTTP Range and search it without a vector database, embedding API, or retrieval server.

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

That's the whole loop. `compile` also takes a live URL (`--source https://docs.example.com`) instead of a directory. If you want a deployed search app — a Worker + UI, not a file — use `npx pikelet create` instead; see [`pikelet/README.md`](pikelet/README.md) for the full CLI reference and the tradeoffs between the two. `compile` never needs the scaffold path's `@xenova/transformers` dependency; skip its ~140 MB install with `npm install -g pikelet --omit=optional` if you only need `compile`/`mcp`.

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

A real Pikelet pack contains **456,153 Simple English Wikipedia records** in a **648.5 MiB** artifact.

Try the live browser demo:

```text
https://pikelet-wiki-playground.pages.dev/
```

The page is static HTML/JS on Cloudflare Pages. It mounts the pack from R2:

```text
https://pub-6da2384a3bca4a44b2b2fa29a94cc811.r2.dev/wikipedia.pikelet
```

There is no `/search`, `/embed`, `/query`, or `/api` backend behind the demo.
The browser opens the `.pikelet` file over HTTP Range, verifies the artifact,
loads the embedded query encoder, and shows the byte ranges it reads.

That file was served by a deliberately dumb HTTP server whose only relevant capability was:

```text
fs.createReadStream + HTTP Range
```

There was no vector database, no search service, no embedding service, no application server that understood the pack, no index process on the server.

The pack was mounted by URL and exposed to a real headless Claude process through MCP. Each invocation was restricted with `--strict-mcp-config` to only:

```text
mcp__wiki-pack__search
mcp__wiki-pack__list_packs
mcp__wiki-pack__get_record
```

No web tool. No filesystem tool. No alternate retrieval tool.

One of the actual responses:

> **Q: What causes earthquakes?**
>
> Earthquakes are caused by the movement of tectonic plates in the Earth's crust. Stress builds up along fault boundaries where plates grind against or ride over one another; when that stress exceeds the friction holding the locked fault in place, it suddenly releases as shock waves (seismic waves) that shake the ground [record 429637].
>
> There are three main fault types that can produce quakes — normal, reverse (thrust), and strike-slip — and some quakes have less clear-cut causes, like deep shifts in ancient rock or mine-collapse "tremors" [record 429638].
>
> Two related mechanisms compound this:
>
> * Subduction, where one plate is forced under another into the mantle, produces the most powerful earthquakes and can trigger tsunamis [record 132833].
> * More broadly, the slow convection-driven movement of crustal plates over the hot mantle is the underlying driver of quakes, volcanoes, and mountain formation [record 132267].
>
> Volcanic eruptions can also cause earthquakes, though these are localized to areas near active volcanoes [record 429637].

The model did not just repeat the first search hit. It retrieved multiple records and synthesized them into a sourced answer.

Other runs produced:

```text
Who was the first person on the moon?          → Neil Armstrong          (records 408686, 408684)
How do volcanoes form?                          → plates + subduction + hotspots
                                                                          (records 132770, 132765, 132852, 132853)
How does photosynthesis work?                   → two-phase reaction, cited (5 records: 114219, 114220, 114222, 114223, 269279)
What is the capital of France?                  → Paris                  (records 454641, 302684)
```

That is the product in one demonstration:

> **A model is interrogating a 456,153-record knowledge base whose backend is a static file.**

The model may already contain some of these facts in its pretrained parameters. This test demonstrates the retrieval, synthesis, citation, and deployment path; the [Veyra ablation](docs/veyra-ablation.md) tests whether support changes when evidence is removed from the pack.

**Network cost.** In a single persistent session (one mount, five queries, one repeat):

| Operation                       | Bytes transferred | Requests |
| ------------------------------- | ----------------: | -------: |
| Initial mount                   |          51.7 MiB |       11 |
| Query 1 + one-time encoder load |          25.9 MiB |       77 |
| Query 2                         |         504.4 KiB |       37 |
| Query 3                         |         499.4 KiB |       46 |
| Query 4                         |         968.9 KiB |       75 |
| Query 5                         |           1.1 MiB |      116 |
| Repeated query 1                |         238.4 KiB |        4 |
| **Total**                       |      **80.8 MiB** |  **366** |

Roughly **12.5% of the 648.5 MiB artifact** crossed the wire across that whole session. Excluding the one-time ~25 MiB encoder load, fresh-query traffic ran **~0.5–1.1 MiB per query**. The complete artifact was never downloaded.

The headless-Claude test above used a fresh process per question, so each invocation repaid the ~52 MiB mount and ~25 MiB encoder cost — about 78 MiB per cold query. That's a real operational distinction: **persistent sessions amortize mount and encoder cost; independent cold processes do not.**

*(The large benchmark fixture still carries its historical `.pancake` filename from before the Pikelet rename; current artifacts use `.pikelet`.)*

---

## Attach a pack to an LLM

`pikelet mcp` exposes one or more packs through the Model Context Protocol:

```bash
npx pikelet mcp install \
  --client claude-code \
  --pack https://example.com/docs.pikelet#<sha256>
```

The agent gets `search`, `get_record`, `list_packs`, `verify_pack`. A search result carries the identity of the pack and the location of the source record, so an agent can work against `product-docs.pikelet`, `rust-reference.pikelet`, `policy-2026-09.pikelet`, `customer-manual-v4.pikelet` without each publisher operating a retrieval API. The artifact can be local, private, public, behind authenticated object storage, or distributed like any other static asset.

A pack mounted with a content hash has a stable identity — `https://example.com/docs.pikelet#8d731...` — so "what body of knowledge did this agent query?" has a reproducible answer.

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

All three packs derive from the same small synthetic Station Veyra corpus, with controlled differences in the Tovash evidence. Ask where the Tovash project is housed: the full pack supports Chamber 17, the modified pack supports Chamber 43, and the ablated pack should abstain.

---

## Match quality and abstention

A nearest neighbour is not automatically evidence that a corpus answers a question. Pikelet can calibrate retrieval signals at build time (best semantic distance, distance margin, lexical coverage, retrieval agreement). When the corpus supports a reliable classifier, results carry `matchQuality: strong | weak | none`. When calibration can't separate supported from unsupported reliably — a single novel may be semantically homogeneous enough that the fit isn't trustworthy — Pikelet reports `matchQuality: unscored` and records why calibration was skipped, rather than manufacturing confidence. A `none` verdict withholds `results` by default; pass `query(text, { showAbstained: true })` to see the raw retrieval anyway — `matchQuality` and `confidence` are unaffected either way.

`matchQuality` is evidence about retrieval support. It is **not** a guarantee that an LLM will never hallucinate.

---

## Deeper dives

- [How a `.pikelet` query runs](docs/how-a-query-runs.md) — what is inside the file, what a query costs on the wire, remote execution, and the integrity checks on the read path.
- [The Veyra ablation](docs/veyra-ablation.md) — what happens to answers when the supporting record is edited or removed from the pack, with the reproduction commands.
- [Why make knowledge a file](docs/why-a-file.md) — the reasoning behind the single-artifact design, and what it commits to.
- [Architecture](docs/architecture.md) — the engine, the C ABI and the JavaScript wrapper; the artifact formats are specified in [`spec/`](spec/).

## What this is not

**Not a claim that vector databases are obsolete.** If your corpus changes continuously, needs transactional updates, serves many tenants, or already lives comfortably in a database, use a database. Pikelet targets `build → publish → query many times → replace on release`.

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
src/                              C++/WASM vector engine: HNSW, float32 and
                                   affine-u8 backends, mutation, compaction,
                                   snapshot import/export. Used at compile
                                   time to build and quantize the index;
                                   query time does not load this engine (see
                                   complete/index.mjs's header comment) —
                                   .pikelet reads run a pure-JS sketch scan
                                   instead, optionally SIMD-accelerated.
complete/, pikelet-artifact.js    Readers and builders for the complete
                                   range-readable artifact
pikelet/                          CLI, compiler, MCP server, encoder
                                   integration, higher-level tooling
docs/                             Deeper dives (how a query runs, the Veyra
                                   ablation, why a file), architecture notes,
                                   rename history, measurement reports
spec/                             Byte-level artifact contracts
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
- BEIR retrieval evaluation
- range-read and failure-mode tests
- a large 456k-record example artifact

`npm test` exercises the engine, artifact profiles, MCP, ingestion, format hardening, and related conformance suites.

Previously known as **Pancake** (renamed September 2026); artifacts, profile strings and files from before the rename remain readable — see [`docs/history.md`](docs/history.md) for what kept the old name and why. Pikelet is unrelated to the pre-existing Pikelet programming language.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
