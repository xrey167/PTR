# F002-weak-supervision

## Hypothesis
Does a verifier-precedence Dawid-Skene label model produce better calibrated and more accurate labels than majority vote at the same annotation budget?

## Primary metrics
accuracy on uniform gold; Brier score; ECE; Unknown and Disputed rates; Krippendorff alpha

## Baseline
majority vote; verifier-only labels

## Falsification
See `experiment.toml`: any label or resolved distribution that contradicts a verifier veto, any disagreement with the independent oracle, no calibration gain over the naive vote-share posterior (paired bootstrap lower bound on the uniform gold sample of the sound-verifier world) or an accuracy loss against the plain majority vote on that world's items, or a failed guard on gold labels and active sets, rejects the model.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Draft status and scope (2026-10-09)
F002 is still `planned`: the harness, the criteria, the aggregator and the mutation plan are built, but nothing is frozen and
no run is evidence. The draft text is [PREREGISTRATION.md](PREREGISTRATION.md) (it becomes the preregistration at the freeze,
together with a `[preregistration]` table in `config.toml` and an entry in `experiments/preregistration.toml`); the decision
rules are [criteria.toml](criteria.toml), applied by [aggregate.py](aggregate.py); the defects the harness must detect are
[tests/mutations.toml](tests/mutations.toml); the non-confirmatory pilot that fixed the form of the gain gates is
[pilot/](pilot/README.md).

Scope (owner decisions of 2026-10-09): a mechanism experiment on synthetic labelling worlds with a known truth; gold is
`GoldSource::Oracle`; nothing is claimed about real annotators or LLM labellers. The harness is
`bins/ptr-bench/src/experiments/f002/` (`ptr-bench weak-supervision <seed>`, one JSON object per seed, exit 0 / 1 / 2 as for
R001). It builds three worlds (sound verifiers, contaminated verifiers, homogeneous control), drives `ptr-labeling` (fit,
resolution with verifier precedence, gold evaluation, acquisition, guards) and the `ptr-analytics` statistics it scores with
through their public APIs, and compares them with an independent oracle (a second Dawid-Skene EM, an expected resolution,
Brier, ECE, Krippendorff alpha, Wilson). The calibration claim is against the naive posterior `(n_c + 1) / (n + C)` and does not isolate Dawid-Skene or verifier precedence (a calibrated vote share, descriptive, is as well calibrated in the homogeneous control and better than the label model when it also respects the vetoes). Stated limits: see PREREGISTRATION.md sections 8 and 9.
