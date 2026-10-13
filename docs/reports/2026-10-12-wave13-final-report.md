# Final report: CPU wave 13 and the local handover (2026-10-12)

Head report for the goal "finish PTR/VCW-MoE" (owner-delegated decisions, conservative and reversible).
Sources: `STATUS.md`, each experiment's `results/run.json`, `results/mutations.json`, `results/STALE.toml`.

## Results

| Experiment | Verdict | Evidence |
| --- | --- | --- |
| F007 vcw-loop | COMPLETED | full loop +7.20 points against a trained gate [6.51, 7.88], +6.35 against best-of-N [5.40, 7.31], margin 3; simulated experts (mechanism check); mutations 33/33 |
| C004 recurrent integration | COMPLETED | +782.6 per mille against a parameter-matched attention stack (lower bound 769.5); counter toy, extrapolation only; 44/44 |
| C007 self-model (CPU) | FALSIFIED | not better than an information-matched probe or the model's own confidence; 50/50 |
| C008 attention schema (CPU) | FALSIFIED | +7.1 [4.8, 9.4] against a router reading the same history, margin 25; 42/42 |
| C001 global workspace | NOT_SHOWN | 20 seeds |
| E007 split-pod-kv | COMPLETED | 5 seeds; 59/59; pilot 1001 equal on 137 keys |
| L002 raft-recovery | COMPLETED | 5 seeds; 32/32; pilot 1001 equal on 79 keys; post-hoc round-5 re-review: no open P0/P1, two deviations disclosed in STATUS.md |
| R013 self-modification authority | COMPLETED | 5 seeds; 32/32; pilot 1001 equal on 57 keys |
| Q003-v2 postgres-hybrid | INVALID | incomplete, superseded |
| Q003-v3 postgres-hybrid | COMPLETED | 5 blind seeds; 28/28; 0 stale hits, recall@10 981-990 per mille; lexical half empty on natural-language queries |

Re-verification on the final integration tree (no records written): pilots of E007, R013 and L002 equal to their
references; mutations E007 59/59, R013 32/32, L002 32/32, Q003-v3 28/28.

## Product and process findings

- ptr-pg's lexical search uses `plainto_tsquery` (AND of all terms) and finds nothing on natural-language questions.
- Ledger (L002 review): failed member refuses work, a pending snapshot is refused, crash-consistent log reset, short foreign-file limit (<= 8 bytes) documented.
- ptr-cluster's cluster-backend wire test asserted the old PTRCS005 snapshot magic (fixed).
- The L002 ledger fixes moved two mutation anchors of the completed L001-v2 (re-anchored, kill verified without a record, noted in its STALE.toml).
- `bindings/ptr-py/Cargo.lock` lacked a dependency line (fixed offline).
- The local `stable` toolchain (1.97) is below the MSRV (1.99); `make ci-*` recipes run with `+1.99.0`.

## Own errors

- Q003-v2 first misdiagnosed as a pipe deadlock (it was torch thread oversubscription).
- An early finish script used `git add -A` (replaced; earlier commits checked).
- The Q003-v3 freeze script set `CARGO_TARGET_DIR`, which `scripts/mutation_check.py` ignores (resumed on the same freeze commit).
- The head's freeze guard restored from the index instead of HEAD (never triggered; fixed in `scripts/freeze/freeze_guard.sh`).
- The first repository freeze-script generator left `target/` before prepare by default; the runner refuses that (found by the E007 replay, fixed with a regression test).

## Freeze tooling validation

`scripts/freeze/` replayed the real E007 freeze in a scratch clone: the generated freeze commit has no diff against the real
one (3c1076b9); 59/59 mutations; pilot equal on 137 entries; preflight, prepare, the five seeds from `config.toml`, the
aggregate (COMPLETED) and `finish_experiment.py` (status, STATUS line, STALE.toml) all ran through.

## Delivery

Waves 13 and 14 as git bundles with sh/ps1 apply scripts, each chain-tested in a fresh clone (check, 3819 tests, vcw-demo,
gates; wave 14 also freeze-tools-test, plugin validation and the training tests). Wave 14 adds AGENTS.md, scripts/freeze,
docs/FREEZE_PROCEDURE.md, docs/LOCAL_START.md, docs/agents/KICKOFF.md and the two-RTX-3090 desktop tooling.

## Open, by design

- GPU stages (GPU_HANDOVER order, VCW-MoE GPU runs), the component measurement study and the ptr-pg lexical follow-up run on
  the owner's Ubuntu desktop with two RTX 3090 (`docs/agents/KICKOFF.md`).
- GPU pinning of frozen seed runs (OD-3): `scripts/run_experiment.py` passes no `CUDA_VISIBLE_DEVICES`; frozen seed runs go one
  at a time until a reviewed runner change settles it.

## Recommended next study

The component measurement study. Prerequisites: a `protocol.toml` per component (`evaluations/PROTOCOL_FORMAT.md`), a pinned
bench workspace under `evaluations/bench/`, one adapter per candidate, and a quiet host.
