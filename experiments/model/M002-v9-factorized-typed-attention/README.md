# M002-v9 — preflight-validated factorized typed pair attention

M002-v9 is the active confirmatory successor. Before its registration, its
complete freeze gate passed against committed but unregistered candidate files.
It preserves M002-v5's development-only selection (`rank=8`, `bias_limit=1`,
`metadata_dropout=0`), the FNV-bound confirmatory folds, and the FactorizedV2
vs Off protocol.

The entrypoint selects `factorized-v2-v9` and `factorized-v2-off-v9`. The
versioned Rust arm table owns both names; the paired runner consults that table
and rejects any cross-study selection before dataset loading. The research
gate verifies current committed pilot artifacts and selector recomputation,
the source-code scripts against their recorded pilot source commit, all runner
sources, and the complete tracked `model/burn-a0` Git-tree digest.

The five paired seeds execute the three FNV-bound folds, which are averaged
inside each seed before the paired Student-t interval. Any absent or failed
gate yields `INCONCLUSIVE/NO-GO`. M002-v9 has no result yet; M009 and Learned
Backend qualification remain locked.
