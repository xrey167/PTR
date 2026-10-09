# Q002-evidence-promotion

## Hypothesis
Does exact-source verification prevent search-hit hallucination? `ptr_search`'s typed promotion path (`ExactSourceResolver`, `SearchHit::observe_resolved` against the lifecycle authority, `verify_claim`, `SearchHit::promote_verified_with`) promotes a retrieved hit to `VerifiedKnown` only if its citation is the byte-exact quote at the cited span of an intact raw event that belongs to the capsule generation the hit names, its claim is what that one statement states, and the lifecycle authority holds the hit's generation live. On seeded synthetic worlds with nine classes of poison in the retriever's hits (and hard honest hits that must not be refused) no poison is promoted, no adversarial stage skip at the API boundary is accepted, at least 950 per mille of the honest hits are promoted and the mechanism answers at least as many questions correctly (answer or abstention) as direct retrieval-to-answer, the lifecycle filter alone and the boolean API. The full text is in `experiment.toml`.

## Primary metrics
The 34 metrics of `experiment.toml` (hits and refusals per poison class, the false Known counts of the mechanism and the three baselines, the stage-skip counters, recall in per mille, task success per arm, the label and corpus-digest guards); the wall-clock time is in `timing_ns` and is never judged.

## Baseline
direct retrieval-to-answer (B0: the top hit is the answer); the boolean API fed `true` (B2) is the negative control for today's API, the lifecycle filter alone (B1) a contrast.

## Falsification
See `experiment.toml` and [criteria.toml](criteria.toml): eleven falsifier rules and twenty-nine coverage rules, every seed must pass, and COMPLETED needs complete mutation evidence.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Status of this directory
Q002 is still `planned`: the product (`crates/ptr-search/src/source.rs`, the typed methods of `SearchHit`), the harness (`bins/ptr-bench/src/experiments/q002/`, subcommand `evidence-promotion <seed>`), the preregistration ([PREREGISTRATION.md](PREREGISTRATION.md)), `criteria.toml`, `aggregate.py` and the mutation plan ([tests/mutations.toml](tests/mutations.toml)) are in place, but no `[preregistration]` table is in `config.toml`, the experiment is not listed in `experiments/preregistration.toml` or `experiments/registry.toml` as anything but planned, and no seed has been run for the record. `config.toml`'s `[pinned_thresholds]` holds the choices the harness compiles in (a drift test compares them), among them the corpus digest of each seed, to be re-measured on the frozen commit.

Stated limits: a synthetic structured world of one-statement events in a fixed format (no free text, paraphrase or real retriever); honest hits are exact by construction, so recall only guards against over-refusal; the raw event source, the ownership record (which capsule generation admitted which event) and the lifecycle closure are trust roots taken from the generator's table, and the claim a proof is made for is the caller's; `Disputed` is not a state the runtime can answer, so P8 uses a harness overlay on the real `PtrRuntime::generation_validity`; the tampered-digest class cites corrupt replicas served by a source double because `KnowledgeStore` cannot hold such an event; a Known is false by the generator's label (cross-checked by a naive reading of the corpus, both inside this harness, the reading independent of the code under test and not of the generator); the boolean API is the documented trusted-caller path and is not restricted.
