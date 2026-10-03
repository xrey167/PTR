# Dataset Card: operator_routing_v2

- Version: v2 (`generator_version = 2`, benchmark seed `20261001`, codebook version 1).
- Source: fully synthetic, deterministic, standard-library-only generation by `benchmarks/operator-routing-v2/generator.py`.
- Purpose: confirmatory evaluation of compositional generalization to role x regime cells that are absent from both train and validation data.
- Confirmatory folds: `evidence-temporal`, `evidence-tabular`, and `claim-interventional`.
- Diagnostic-only fold: `evidence-interventional`. It reproduces the v1 target cell for continuity and must not be used as primary confirmatory evidence.
- Development-only fold: `claim-temporal-development`. Together with the diagnostic fold it is restricted to the two-seed pilot and cannot enter confirmatory evidence.
- Split contract: every fold has `train` (2,048), `val` (256), `test_iid` (512), and `test_ood` (512). The target role x regime combination is absent from train, validation, and IID test. Every OOD example uses the target regime and contains at least one target-role fact that is both `live` and nonzero (`confidence_bucket > 0`).
- Labels: the fixed v1 routing utility rule and codebook are retained, while sampling and split identity are v2-specific. `score.py` independently re-derives labels from TSV fields.
- Schema: `datasets/schemas/operator_route_v2.schema.json`.
- Outputs: generated JSONL/TSV files live under ignored `datasets/generated/operator_routing_v2/<fold>/`. The committed `benchmarks/operator-routing-v2/splits.lock.json` pins full and prefix digests; `samples.jsonl` contains one small example per fold and split.
- Contamination controls: tests enumerate all generated train/val examples for target-cell leakage and all OOD examples for a live/nonzero target fact.
- Sensitive data and rights: no personal or third-party data; integer entity ids and rendered synthetic text only.
- Known limitations: synthetic rule and vocabulary, one benchmark seed, index-tagged attributes, and no claim about natural-language routing performance.

Regenerate or verify:

```sh
python benchmarks/operator-routing-v2/generator.py --check
python benchmarks/operator-routing-v2/generator.py --update-lock
python -m unittest discover -s benchmarks/operator-routing-v2/tests
```
