# Project history: the Pancake → Pikelet rename

Pikelet was developed and first published as **Pancake** (`pancake-wasm`,
`create-pancake-search`). It was renamed in September 2026. This page records
what changed, what deliberately kept the old name, and why, so that a
`pancake` mention found in the tree can be classified without re-deriving
the reasoning.

## Timeline

| Date | Change |
| --- | --- |
| 2026-09-02 | Packages renamed: `pancake-wasm` → `pikelet-wasm`, `create-pancake-search` → `pikelet` (one CLI: `create`, `compile`, `doctor`, `mcp`). New compiles default to the `.pikelet` extension. The wire format kept its pancake-era names on purpose. Released as 0.7.0. |
| 2026-09-03 – 09-05 | Repo-wide rename of directories, entrypoints, error strings, and prose. The GitHub repository moved to `mcn92/pikelet` (the old URL redirects). |
| 2026-09-12 | The published surface renamed: `Pancake*` API names → `Pikelet*` with the compatibility aliases dropped, the C ABI `pancake_*` → `pikelet_*`, the N-API addon source to `packages/pikelet-wasm/native/pikelet_napi.cpp`, and the manifest profile strings `pancake-complete-v1/v2` → `pikelet-complete-v1/v2`. Readers accept both profile strings for the same format version (`LEGACY_PROFILES` in `packages/pikelet-wasm/complete/index.mjs`). Released as 0.8.0 — see the CHANGELOG for the identity consequence. |
| 0.8.x | The `PANCAKE_*` environment-variable fallbacks were removed; only `PIKELET_*` names are read. |

## What still says "pancake", and why it stays

These are not leftovers. Each is either a compatibility guarantee, an
external identifier, or data.

- **Profile strings on read.** `packages/pikelet-wasm/complete/index.mjs` accepts
  `pancake-complete-v1/v2` alongside the `pikelet-` strings. Every pack
  published before 2026-09-12 — including the live wiki, astro-docs and
  rust-book release assets — carries the old string, and its identity is
  the hash of a manifest that contains it. Builders only ever write the new
  string.
- **`.pancake` files.** Readers dispatch on magic bytes, never on extension.
  `.pancake`, `.pancake-range` and `.pancake-sketch` files open exactly as
  their `.pikelet*` equivalents. The magics themselves (`PNCK`, `PSF1`,
  `PSA1`, `FLH1`, `I8H1`) never encoded the name.
- **Release-asset filenames.** `pancake-wiki-inline.pancake` under the
  `artifact-wiki-inline-v4` GitHub release is referenced by URL from
  `packs/packs.json`, the CLI README and the range-proof scripts; renaming
  the asset would break every pinned URL and change nothing about its
  content or identity.
- **Deprecated npm packages.** `pancake-wasm` and `create-pancake-search`
  remain on the registry as deprecated pointers to `pikelet-wasm` and
  `pikelet`.
- **Deployed infrastructure names** in `examples/static-wiki-pack`
  (`pancake-wiki-pack` R2 bucket, `pancake-wiki-pack-demo` Pages project).
  Those are the names the deployment actually has; the docs that reference
  them are deploy instructions, not branding.
- **Committed measurement data.** Benchmark result files under
  `benchmarks/results/`, the ANN-Benchmarks `local_results`, and the
  encoder-conformance fixtures under `test/fixtures/` contain
  `pancake` as config labels or corpus text. They are records or
  hash-verified inputs; editing them would falsify a measurement or break
  a fixture.
- **Corpus vocabulary in the student trainer.** `packages/pikelet/tools/train_student.py`
  lists `pancake` among the documentation-corpus tokens it was distilled
  against. Changing the list changes the training procedure.
- **The test fixture `pancake-wiki.pancake`** named in
  `examples/one-file-search` scripts is a locally compiled file whose
  name matches the release asset it mirrors.
- **CHANGELOG entries** describe the releases as they shipped and are not
  rewritten.

## What was renamed after the fact

The 2026-09-12 rename left references that only broke later or were only
noticed later; they were fixed on the `repo-cleanup` branch (2026-09-23):
the engine benchmark scripts' `native.pancake_*` calls (the N-API exports
had become `pikelet_*`), the ANN-Benchmarks adapter's `pancake_py` import
(the pybind module had become `pikelet_py`), the WASM demo pages' `_pancake_*`
calls and the iOS smoke page's inlined pre-rename engine, the Docusaurus
plugin's webpack rule matching `pancake-(artifact|errors).js`, and API names
in several READMEs (`openPancakeFile`, `PancakeRangeArtifact`,
`PancakeSketchArtifact`, `PancakeIndex`).

## Unrelated: the Pikelet programming language

Pikelet the search project is unrelated to the pre-existing Pikelet
dependently typed programming language (`pikelet-lang/pikelet`).
