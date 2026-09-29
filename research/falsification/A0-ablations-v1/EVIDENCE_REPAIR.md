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

Remaining study review work includes retaining every contingency record and enforcing learning-rate sweep provenance, followed by integrating the branch with main. Correct those before a new coherent evaluation, then advance model experiments under the repository priorities. This repair does not claim to complete the research roadmap or make PR #32 merge-ready.

## Runner follow-up — 2026-09-29

The generic runner now asks the selected Cargo toolchain for its workspace before launch and records that workspace's `Cargo.lock` path and SHA-256. This handles both the standalone A0 manifest and members of the root workspace, including `--manifest-path=value`; arguments after the executable separator do not select a Cargo manifest. A failed resolution or missing lock refuses launch. Non-Cargo commands record no Cargo dependency claim. Historical run records are preserved, including their known incorrect root-lock attribution; these changes do not repair old provenance retroactively.

Aggregation rejects missing or foreign experiment identities, dirty records without a valid tracked-diff SHA-256 even under `--allow-dirty`, and records with differing dependency locks. Explicitly opted-in legacy records with unknown worktree state retain their existing behavior.

Validation: all 35 runner tests pass; the complete scripts suite passes 321 tests with one existing live-Rust comparison skipped. Ten regression cases fail against the previous runner with assertion failures and no execution errors. A real Cargo 1.85 workspace-location query confirms root-member resolution; A0 toolchain selection and lock persistence are covered by isolated tests, not a new A0 training run. Re-aggregation preserves the corrected study results and the stock cross-check.

A non-mutating merge preview against main at `07241fb` reports 19 conflicting files. Main has a substantially rewritten provenance-aware runner and no stock `aggregate` command, so taking either runner wholesale would discard behavior. Integration must preserve main's execution safeguards, port the aggregation interface and its tests, reconcile CI and registry changes, and regenerate component documentation. No merge has been attempted or declared ready.
