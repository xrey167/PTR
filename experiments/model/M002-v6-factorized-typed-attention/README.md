# M002-v6 — FNV-bound factorized typed pair attention

M002-v6 is the confirmatory successor to superseded M002-v5. It reuses the
immutable, development-only v5 pilot selection (`rank=8`, `bias_limit=1`,
`metadata_dropout=0`) without retuning any model, router, dataset, seed or
gate. The sole protocol correction is to bind every confirmatory fold in the
actual Cargo command as `name=FNV64`, as required by the Rust `paired-v5`
interface.

The selected architecture directly compares `FactorizedV2` against `Off`.
Both arms execute the same pair-attention operations; `Off` zeros only the
resulting bias. The unit of inference is a seed: each of the five seeds runs
both arms over all three confirmatory folds, then averages folds before the
paired Student-t interval is calculated.

M002-v6 must not start until its registry entry, source pilot, factorized
contract, decision adapter and FNV fold bindings are commit-bound and the
research gate accepts the freeze. A missing or failed gate is always
`INCONCLUSIVE/NO-GO`; no v5 artifact and no pilot metric is positive evidence.
M009 and Learned Backend qualification remain locked. A committed v6 PASS is
necessary evidence, after which M009 must still be separately preregistered
with an explicit v6 decision and FactorizedV2-contract binding.
