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

## Study completion and integration — 2026-09-29

The remaining study review defects are repaired. Separate contingency arm sets retain their own completed records and retries for every seed. Repeated comparator scores may coalesce only when every independently scored field agrees. The stock aggregate can select each arm configuration separately, preserving its cross-check. Learning-rate selection now validates the frozen manifest, command (including width, steps and arms), seed, commit, clean-tree flag, measured host, toolchain, hardware profile and dataset identity. Duplicate arm/rate cells are rejected. G4 also reproduces the selected rates from sweep records held by the evaluation commit. All nine archived rate selections reproduce exactly.

The merge with main at `07241fb` resolves all 19 conflicts. Main's execution safeguards remain in `run_experiment.py`; the exploratory aggregate implementation and tests move to `aggregate_experiment.py`, reached through the existing `run_experiment.py aggregate` CLI. This aggregate does not publish the lifecycle certification artifact pair. The runner keeps whole-tree metadata for exploratory records and names a non-listed Cargo command's selected lock separately, preserving the existing root-lock field's meaning. Both sets of registry entries, PostgreSQL CI jobs and clippy checks remain, and all 33 component documentation records are regenerated.

The candidate merge passes research gates, metadata freshness, notices, registry and generated-document checks. All 61 benchmark tests, 77 affected root Rust tests and 59 A0 Rust tests pass. The built debug A0 binary agrees with all 12 configured arms. The full Python scripts suite has 672 passes, one release-binary comparison skipped, and one environment-blocked test: this sandbox denies creation of AF_UNIX sockets. That test is unchanged and must pass in GitHub CI; no skip or weaker assertion was added. The standalone debug comparison separately covers the skipped configuration check.

Re-aggregation still passes G0–G5 and the stock cross-check. G6 remains false because historical correctness provenance was never recorded; model hypotheses remain INCONCLUSIVE. These code and integration repairs do not retroactively certify the old study or perform a new training run.


## Admission and review follow-up — 2026-09-29

Admission-aware batches now replace raw attribute tokens of non-live facts with
PAD, using encoded slot identities even when tokens are shuffled. Previously,
admitted slots could attend to those attributes through the raw stream. The
unmasked control retains its raw input. This changes future training behavior;
it does not repair the archived model runs or justify a historical hard-admission
claim. A new frozen study must train the corrected implementation.

Evaluation now stages correctness logs outside the source tree until every model
process (including retries) finishes. G4 requires versioned empty-diff evidence,
checks both learning-rate artifacts and actual per-arm metadata, and compares
measured hosts, toolchains and frozen hardware profiles across sweeps and runs.
Contingency decisions recheck every contrast arm's learnability, including a
comparator that passed at the original step budget. Summary documents now agree
with the INCONCLUSIVE gated verdicts. Archived raw records remain unchanged.

The R2 fallback is now automated after base and R1 fail: it checks and revises
all three independent rule implementations, regenerates and pins data, recomputes
references, rebuilds and self-checks the binary, and calibrates width 48. Failed
G0 bands prohibit an ablation plan even if calibration reaches its target. A
failed calibration removes any old budget so it cannot be reused accidentally.
This fallback has orchestration regression coverage; no new training or R2
capacity finding is claimed here. Existing frozen rule files are unchanged.
