# A0 evidence repair — 2026-09-29

PTR aims to test whether typed semantic state, uncertainty and operator routing improve a trainable reasoning architecture under controlled comparisons. This synthetic A0 study is an early mechanism probe, not evidence of language-model superiority.

## Repairs

- Reject duplicate Rust final-count rows instead of accepting the first match.
- Compare the dataset fingerprint actually used by Python scoring with the frozen lock for evaluation, contingency and deterministic rerun records.
- Require the preregistered seed 17 for the deterministic rerun.
- Freeze criteria, references, budget, preregistration, design, split lock and codebook byte-for-byte against the original preregistration commit.
- Resolve preregistration from immutable commit `45b2d5491e40171dd242d0bac6cf7d78d9c70746`; an absent local tag is supported, while a conflicting tag is rejected.
- Require G6 evidence to record the exact checks and commands, successful exit codes, matching log hashes, evaluation commit and clean source snapshots before and after checking. The collector buffers outputs until after the second source snapshot. This binds evidence; it is not a cryptographic attestation of execution.

## Re-analysis of archived runs

The locked generator produced 32,000 examples with fingerprint `0ad71688f09b0d0d`. Re-aggregation of the archived runs passes G0–G5 and agrees with the stock aggregate to 1e-12. Full-arm mean IID accuracy remains 0.8839333333333335. No training, prediction or raw correctness log was changed.

G6 fails: the historical `logs/g6.json` contains only pass labels, without source, command or log binding. Those labels cannot certify the evaluated commit. The refreshed `results.json` and `RESULTS.md` therefore mark all gated mechanism hypotheses INCONCLUSIVE. The raw-blind manipulation check remains PASS; sufficiency remains NOT RUN. The two derived FALSIFIED notes are withdrawn because the current evidence no longer supports those verdicts; their historical versions remain in git.

Do not manufacture provenance for historical checks. Restoring a gated conclusion requires genuinely reproducible correctness evidence for the evaluated source and resolution of the remaining study blockers. If that cannot be established for the frozen study, preregister and execute a new study with the corrected evidence collection.

## Verification and next work

The benchmark suite passes all 61 tests. The focused aggregator, driver, configuration and report suites pass 63 tests, with one existing live-Rust configuration comparison skipped because the A0 binary is not built here. The full scripts suite also passes 312 tests with that same one skip; research gates, all 21 experiment manifests and third-party notices validate. The archived-data aggregation and report generation completed successfully; a successful command does not imply that all scientific gates pass.

Remaining review work includes retaining every contingency record, enforcing learning-rate sweep provenance, binding generic runner lockfile and experiment identity, handling missing dirty-source hashes, and integrating the branch with main. Correct those before a new coherent evaluation, then advance model experiments under the repository priorities. This repair does not claim to complete the research roadmap or make PR #32 merge-ready.
