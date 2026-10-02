# M002-v7 — runner-bound factorized typed pair attention

M002-v7 is the only confirmatory successor to superseded M002-v6. It keeps
the exact immutable M002-v5 development selection (`rank=8`, `bias_limit=1`,
`metadata_dropout=0`), the v6 FNV-bound fold tokens, dataset, seeds, model,
router and gates. It changes only the discovered runner identity fault.

The Cargo entrypoint selects `factorized-v2-v7` and `factorized-v2-off-v7`.
Those names are separately registered for M002-v7 in the Rust arm catalog;
they expose `FactorizedV2` and `Off` respectively. The latter executes the
same factorized pair-attention graph but multiplies only its resulting bias by
zero. The runner rejects v5 arm names, v6, and any other cross-study pair
before reading a dataset.

The unit of inference is a seed. Each of five seeds executes both arms across
the three listed FNV-bound folds. Folds are averaged inside each seed before a
paired five-seed Student-t interval is calculated. Any missing gate is
`INCONCLUSIVE/NO-GO`; neither pilot data nor prior superseded records are
positive evidence. M009 and Learned Backend qualification remain locked.
