# Dataset Card: own_code_corpus_v0_1

- Version: 0.1 (planned; not yet imported)
- Source / generator: The user's own repositories, plus optionally a permissively-licensed public code dataset subset mixed in
- License / rights: Own repositories: rights held by the user. Any mixed-in public code dataset must carry a permissive license (MIT/Apache/BSD-class) with attribution preserved per its own license; no dataset with a share-alike or non-commercial clause may be mixed in without separate legal review
- Sensitive data: Own repositories must be reviewed for secrets/credentials/PII before inclusion; no automated scrubbing has run yet
- Schema: Raw text corpus, one document per source file, used as `{"text": ...}` for continued pretraining (Axolotl `pretraining_dataset` shape — not the Alpaca/ShareGPT instruction formats)
- Size: Multiple GB from own repositories, plus any mixed-in public code; exact byte count TBD once collected
- Split strategy: train / heldout, with heldout drawn from whole repositories or files excluded from training entirely (not a random line-level split), so held-out completion accuracy reflects genuine generalization rather than near-duplicate leakage
- Contamination controls: MinHash (256 permutations) + LSH clustering + Jaccard similarity (~0.85 threshold) near-deduplication across own-repo and public-code sources before any split is drawn
- Intended training stages: Continued pretraining only (`training/continued-pretraining/`), not instruction-tuning
- Evaluation exclusions: The heldout split and any public benchmark suite referenced by experiment R003 must never be trained on
- Known limitations: Not yet collected or hashed; the `datasets/registry.toml` entry stays `status = "planned"` until a real archive with a verified SHA-256 exists
- SHA-256 / manifest: TBD — populate once `datasets/bundles/own_code_corpus_v0_1.zip` exists and its hash/size are recorded in `datasets/registry.toml`
