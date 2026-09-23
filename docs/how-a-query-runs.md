# How a `.pikelet` query runs

What is inside the file, what a query costs on the wire, how a remote query executes, and how integrity is enforced on the read path. Moved verbatim from the README on 2026-09-23.

## What's inside a `.pikelet`

```text
manifest        format versions, segment offsets, artifact identity
semantic index  resident sketch rows, full affine-u8 rows, row commitments
corpus          source records, offsets, per-record commitments
query encoder   WordPiece vocabulary, quantized MiniLM weights, encoder declaration
lexical index   BM25 postings
calibration     supported / unsupported retrieval model
evaluation      golden queries / expected behavior
```

The artifact is immutable. Publish a new corpus by publishing a new artifact; old artifacts retain their identity.

**Encoder profiles.** Self-containment has a cost — the default profile carries ~25 MiB of MiniLM data regardless of corpus size. Three arrangements: **inline** (tokenizer + weights + runtime in the artifact — largest file, no external dependency, used by the Wikipedia demo above), **distilled/compact** (smaller corpus-specific representation, lower footprint, potentially lower quality), **host-supplied** (the artifact declares expected encoder behavior and verifies the host implementation against embedded test vectors — smallest artifact, no longer fully self-contained). The format treats the encoder as a capability, not Pikelet's identity — MiniLM is the current choice, not a permanent requirement.

---

## What does the compression cost?

The artifact architecture is only useful if the compact representation does not quietly destroy retrieval. The retrieval path was decomposed on BEIR rather than evaluated only end-to-end, across four configurations that isolate each transformation:

```text
A  upstream all-MiniLM-L6-v2, float32 corpus, exhaustive search
B  Pikelet embedded encoder,  float32 corpus, exhaustive search
C  Pikelet embedded encoder,  affine-u8 corpus, exhaustive search
D  Pikelet embedded encoder,  affine-u8 corpus, Pikelet HNSW
```

Corpus mapping and evaluation config were frozen before the runs. Official BEIR qrels are used; no LLM judges relevance.

| Dataset  | Config                                     |    nDCG@10 |  Recall@10 | Recall@100 | median ms/q | p95 ms/q |
| -------- | ------------------------------------------ | ---------: | ---------: | ---------: | ----------: | -------: |
| SciFact  | A — upstream float exhaustive              |     0.6451 |     0.7833 |     0.9250 |         3.6 |      4.0 |
| SciFact  | B — Pikelet encoder / float exhaustive     |     0.6512 |     0.7942 |     0.9417 |         3.8 |      4.4 |
| SciFact  | C — Pikelet encoder / affine-u8 exhaustive |     0.6500 |     0.7942 |     0.9417 |         6.9 |      8.7 |
| SciFact  | D — Pikelet encoder / affine-u8 HNSW       | **0.6500** | **0.7942** | **0.9417** |    **0.15** | **0.24** |
| NFCorpus | A — upstream float exhaustive              |     0.3159 |     0.1550 |     0.3115 |         2.6 |      3.1 |
| NFCorpus | B — Pikelet encoder / float exhaustive     |     0.3154 |     0.1511 |     0.3044 |         2.8 |      3.3 |
| NFCorpus | C — Pikelet encoder / affine-u8 exhaustive |     0.3154 |     0.1511 |     0.3036 |         2.9 |      3.6 |
| NFCorpus | D — Pikelet encoder / affine-u8 HNSW       | **0.3135** | **0.1482** | **0.3061** |    **0.16** | **0.25** |
| ArguAna  | A — upstream float exhaustive              |     0.3698 |     0.7653 |     0.9772 |         7.8 |     10.8 |
| ArguAna  | B — Pikelet encoder / float exhaustive     |     0.3506 |     0.7397 |     0.9801 |         6.1 |      7.3 |
| ArguAna  | C — Pikelet encoder / affine-u8 exhaustive |     0.3496 |     0.7368 |     0.9801 |         6.4 |      7.5 |
| ArguAna  | D — Pikelet encoder / affine-u8 HNSW       | **0.3496** | **0.7368** | **0.9808** |    **0.12** | **0.20** |

These results are more useful because they are not uniformly flattering.

**Affine-u8 storage is not the main quality cost** (B→C nDCG@10: SciFact 0.6512→0.6500, NFCorpus 0.3154→0.3154, ArguAna 0.3506→0.3496). **HNSW is similarly close to exhaustive search** at these settings (C→D: SciFact and ArguAna unchanged at reported precision; NFCorpus exposes a small approximation loss, 0.3154→0.3135).

**The largest observed loss is the embedded query encoder on ArguAna**: A→B moves nDCG@10 from 0.3698 to 0.3506, about a 5% relative reduction. Recall@100 actually improves slightly (0.9772→0.9801) — the relevant document is generally still in the candidate set; the degradation is in fine ordering near the top. That limitation isn't hidden: MiniLM is small *because* the goal is to fit the query encoder inside the artifact. It is not state of the art, and the compact implementation is not behaviorally identical to an upstream sentence-transformers runtime on every task.

The benchmark harness, quantization-conformance test, frozen configuration, and raw runs live under `benchmarks/beir/`.

---

## How a remote query runs

Over a network, what makes search slow is not bytes; it's sequential round trips. Graph traversal is a chain of dependent reads — fetch node, inspect neighbors, fetch the next node, repeat — and across object storage those dependent round trips dominate. Measured against SIFT1M over real network storage, a resident scan plus one batched candidate-fetch phase beat graph traversal by about 5× at equal recall. So the remote artifact path carries no graph. The HNSW graph still exists — it's the in-memory engine, used when the whole index is local and round trips are free; that's the path BEIR Config D measures.

For the remote path:

1. **Resident tier** — a pooled sketch of every semantic row loads once when the artifact opens and stays in memory. For the 456,153-record Wikipedia pack: 648.5 MiB artifact, 51.7 MiB mount fetch. A SIMD kernel scans all of it in milliseconds.
2. **Candidate selection** — the scan picks the top `C` candidates against the query vector; the lexical BM25 index can contribute additional candidates into the same set. `C` is the only real knob: the compiler measures recall against brute force on held-out queries at build time and writes the operating point into the file, so readers don't have to guess it.
3. **Batched byte-range fetch** — full affine-u8 rows for the candidate set are fetched in one parallel round, nearby ranges coalesced, rather than a graph-dependent network walk.
4. **Full rerank** — fetched rows are scored against the float query and verified against their digest.
5. **Corpus hydration** — only the records needed for returned results are fetched, each carrying record ID, title, section, source, and artifact identity. The reader returns evidence, not anonymous vector IDs.

That architecture is why a 648.5 MiB pack can answer a fresh query while fetching about 0.5–1.1 MiB after warmup.

---

## Integrity is part of the read path

A range-readable artifact cannot hash the entire file on every open without defeating the point of range reads, so integrity is layered: the resident structural portion is verified during open; lazily fetched index rows and corpus records carry independent commitments and are checked when read. Bytes a query never touches don't have to cross the network merely to prove the bytes it *did* use were correct.

One exception: a lexical (BM25) segment above 8 MiB opens lazily and is covered only by the manifest's whole-segment digest, not per-read like index rows and corpus records — the same transitional stance format-1 sketch rows carry. A pack large enough for this to apply can have its lexical candidates altered between open and a full verification pass without failing a query.

The failure-mode suite (`benchmarks/range-proof/failure-modes.mjs`) exercises the important cases:

```text
correct Range server         → mounts, queries successfully
server ignores Range         → small files fall back to a bounded download;
                                large files refuse ("host ignores Range and the
                                file is 680029254 bytes; refusing full download" —
                                the cap is currently 64 MiB)
bytes change under a pinned identity → refused; tampered data is not silently served
truncated Range response     → fetch fails; no partial result is silently
                                interpreted as valid data
```

The point is not that static HTTP is magically reliable. The point is that a static artifact can fail in explicit, testable ways.

---
