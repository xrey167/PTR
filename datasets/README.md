# PTR Datasets

Datasets are organized by provenance and intended use, not only file format.

- `raw/` — immutable/source data
- `processed/` — deterministic transformations
- `generated/` — synthetic/teacher-generated data
- `heldout/` — evaluation-only
- `ood/` — nominal-type, unseen-Pod and composition OOD
- `private/` — never committed if it contains sensitive material
- `schemas/` — canonical record schemas
- `cards/` — dataset cards
- `bundles/` — versioned imported/generated archives

Training and evaluation leakage must be checked before a run is accepted.
