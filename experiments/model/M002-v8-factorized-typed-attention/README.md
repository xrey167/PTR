# M002-v8 — fully source-bound factorized typed pair attention

M002-v8 is the only active confirmatory successor. It preserves the exact
M002-v5 development-only selection (`rank=8`, `bias_limit=1`,
`metadata_dropout=0`), v6's FNV-bound confirmatory folds, and v7's explicit
runner-owned pair. It adds the missing complete tracked A0 Git-tree binding
before any execution.

The entrypoint selects `factorized-v2-v8` and `factorized-v2-off-v8`. Both
arms use the same FactorizedV2 graph and weights; `Off` computes that graph
and zeros only its pair bias. The Rust selector rejects every cross-study arm
combination before loading a dataset. The research gate verifies the v5 pilot
archive against its recorded source commit, each current runner file, and the
complete `model/burn-a0` tracked-tree digest.

Five paired seeds each execute the three FNV-bound folds; folds are averaged
inside a seed before the paired five-seed Student-t interval. Any absent or
failed gate is `INCONCLUSIVE/NO-GO`. This study contains no result yet; M009
and Learned Backend qualification remain locked.
