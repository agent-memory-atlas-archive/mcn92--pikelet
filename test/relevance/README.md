# Relevance regression suite

Labeled query sets with expected sections, run against compiled artifacts
with `scripts/bakeoff-retrieval.mjs`. This is the permanent record of what
"retrieval works" means: when ranking, chunking, calibration, or the
encoder changes, rerun the suite and compare against the baselines
committed beside each query set.

## Query sets

- `nodeapi-queries.json` — nodejs.org/api, crawled with
  `compile --source https://nodejs.org/api/ --max-pages 30`. A real,
  substantial docsite: ~2,000 sections, dense API identifiers, authored
  anchor ids. Mixes natural-language questions, API identifiers
  (`fs.readFileSync`), error codes, CLI flags, and HTTP-ish exact
  lookups.
- `nodeapi-baseline.json` — the metrics recorded when the set was
  authored, with the artifact identity they were measured against.

## Abstention sets

Labeled question sets for `scripts/calibration-harness.mjs`, which reports
false-abstention and false-answer rates per question class. Each row is
answerable unless its class is unsupported/offdomain/unanswerable/nearmiss
or its answer is null; answerable rows carry near-verbatim evidence quotes
the harness checks the returned text against.

- `pride-prejudice-queries.json` — 174 questions over the 61-chapter novel
  (corpus: `local-packs/pride-prejudice-src/`, not tracked); 157 direct,
  12 unanswerable, 5 near-miss. `pride-prejudice-questions.txt` is the bare
  source list the labeled set was built from.
- `internal-docs-queries.json` — 48 questions over a 4-file, 45-record docs
  corpus (`local-packs/internal-docs/corpus/`, not tracked): 24 paraphrase
  (reworded to avoid the section's own terms), 6 direct, 12 unsupported,
  6 near-miss.
- `veyra-baseline.json` — harness result for the Station Veyra synthetic
  registry. Its 90-question set is generated, not tracked: `node
  examples/one-file-search/web/veyra-corpus/gen.mjs` writes
  `questions.json` beside the corpus it generates (deterministic; the
  generator has no random source).

```bash
node scripts/calibration-harness.mjs <pack.pikelet> test/relevance/internal-docs-queries.json
node scripts/calibration-harness.mjs <pack.pikelet> \
  examples/one-file-search/web/veyra-corpus/questions.json \
  --baseline test/relevance/veyra-baseline.json
```

Not part of `npm test` (needs the local corpora above). Rerun on Veyra and
one real corpus before and after any change to `pikelet/src/calibrate.mjs`
or `complete/retrieval-abstention.mjs`.

## Running

```bash
# recompile the corpus (network; page content drifts over time)
npx pikelet compile --source https://nodejs.org/api/ \
  --max-pages 30 --out nodeapi.pikelet

# measure all three retrieval modes
node scripts/bakeoff-retrieval.mjs nodeapi.pikelet \
  test/relevance/nodeapi-queries.json
```

Not part of `npm test`: it needs the network and ~20 minutes of local
embedding. Treat a drop against the committed baseline as a regression to
explain, not noise to re-record — the corpus can drift when the site
publishes new docs, so re-author expectations only when a target section
verifiably moved or was renamed upstream.
