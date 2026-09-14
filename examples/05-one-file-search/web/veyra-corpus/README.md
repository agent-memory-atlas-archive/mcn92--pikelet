# Veyra demo corpus

Source for the abstention/containment demo (`../ablation.mjs`,
`../public/reproduce-ablation.mjs`): a fictional research-station corpus
generated deterministically so every fact, evidence chain, and expected
abstention label is known exactly.

```bash
node gen.mjs              # writes corpus/ and corpus-ablated/
node make-chamber43.mjs   # writes corpus-chamber43/ from corpus/ (run gen.mjs first)
```

- `corpus/` — full corpus, 94 facts (`veyra.pikelet`)
- `corpus-ablated/` — 6 facts removed; the removed facts' questions must abstain (`veyra-ablated.pikelet`)
- `corpus-chamber43/` — `corpus/` with exactly one fact edited to contradict itself: `loc-tovash.md` says "Chamber 43" where every other record and cross-reference implies Chamber 17 (`veyra-chamber43.pikelet`)

`gen.mjs` also writes `questions.json` (90 questions across direct/multihop/
counting/unsupported/off-domain/near-miss classes, with the ablation plan
and expected post-ablation labels).

## Compiling and publishing a pack

```bash
node gen.mjs && node make-chamber43.mjs
npx pikelet compile --source corpus            --out veyra.pikelet           --name "Station Veyra"
npx pikelet compile --source corpus-ablated     --out veyra-ablated.pikelet    --name "Station Veyra (ablated)"
npx pikelet compile --source corpus-chamber43   --out veyra-chamber43.pikelet  --name "Station Veyra (chamber43)"
```

Compilation involves a self-calibrated inline encoder and isn't
byte-deterministic (embedding and calibration both touch floats), so a
recompile won't reproduce the exact bytes of the published packs — it will
reproduce their behavior. The three packs actually served by the demo are
published as GitHub release assets (`veyra-packs-v1`), not committed here:
each is a ~25MB self-contained artifact, and three of them was most of what
used to make this repo's checkout heavy. `fetch-veyra.mjs` downloads and
SHA-256-verifies them into `../public/`:

```bash
node fetch-veyra.mjs
```

If you recompile and want to republish, update the pinned SHA-256 values in
`fetch-veyra.mjs` after uploading the new assets to the release.
