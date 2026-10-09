# S002 snapshot cancellation: preregistration

Pre-freeze text of the preregistration of S002; at the freeze it becomes the `preregistration_text` file of `config.toml`'s
`[preregistration]` table (renamed `PREREGISTRATION.md`) and its SHA-256 is frozen with the table, as S001's was. S002 is
`planned`: nothing here is a result and no run is evidence. **Not freezable until the invariant review of the cancellation has been
done** (owner decision, section 3; the open points for it are in section 10). The decisions are made by `criteria.toml` and applied
by `aggregate.py`; this file gives their reasons, the definitions, the modelling choices, what the harness cannot produce and the
limits of the claim. Every criterion is the text of `experiment.toml` (hypothesis, falsification, metrics) and the pins of
`config.toml`'s `[pinned_thresholds]`; the harness is `bins/ptr-bench/src/experiments/s002/` (subcommand
`snapshot-cancellation <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench -- snapshot-cancellation <seed>`

Scope: the resumable model loop of `ptr-runtime` as a steppable run (`PtrRuntime::begin_resumable_run`, `step_resumable_run`,
`crates/ptr-runtime/src/stepped.rs`) and its cooperative cancellation when the semantic revision moved under a run for a reason
other than the run's own writes (`semantic.cancel_stale_revisions`), together with the hard effect boundary that stays the only
safety barrier (`authorize_action`, `prepare_execution_once`, `execute_prepared`; `ptr-security`'s authorization). The claim is that,
driven through their public API by a deterministic cooperative scheduler that interleaves a run with a writer at five positions of
the run's life, (1) no stale action is executed and no `EffectAttempted` record names a stale revision or generation, in the
baseline (cancellation off) and with cancellation on, and the boundary answers the same action identically in both arms, (2)
cancellation is exact (it cancels precisely the runs that a foreign write made stale, at the very next step, hands out no action,
and never cancels a run that only wrote itself), and (3) a cancelled run takes no step after the foreign write, where the
baseline's run goes on. Not claimed: that cancelling is cheaper in total (section 8), anything about a real model backend, or
anything about runs that share a runtime (section 9).

**Claim text (the one the result may print):** "a cancelled run takes no step on a replaced snapshot, and cancellation does not
weaken the hard effect boundary", exercised in a harness. **No shipped host steps runs**: `ptr-server` and `ptrd` call the
blocking entry points, which force `CancelPolicy::Off`, so `semantic.cancel_stale_revisions` has no effect on any production path
today. The result is not "cancellation cuts wasted work" and not "cancellation saves compute": the headline numbers are
`total_units_a0`, `total_units_a1` and `net_units_a1_minus_a0` with its sign (descriptive, ungated; section 8).

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> N" violation `<= N`). COMPLETED iff every seed meets every coverage rule and no falsifier trips **and** the mutation
evidence is complete (`results/mutations.json` present, from this code, checker and plan, with every planted defect killed); FALSIFIED
if a coverage rule (the run is vacuous) or a falsifier fails on any seed, whatever the mutation evidence; INVALID if a record is
missing, does not describe the frozen run, or a counter is missing or mistyped, or the exit status disagrees with the rules, or the
mutation evidence is missing or partial (INVALID says nothing about the hypothesis). The harness exits 0 when every rule holds, 1 when
one fails, 2 when it cannot run (section 7). **A defect of the code under test is never exit 2, and a panic or a hang of the
product is a falsification**: every call into the runtime is inside `catch_unwind` (`product_panics`), and a watchdog thread prints
the run as it stands with `product_hangs = 1` and exits 1 when no case has ended for 60 seconds (`hang_seconds`). A panic in the
harness's own code is exit 101 and INVALID. No rule changes after a run.

Coverage (the run is vacuous if unmet), all counted from events the harness observed, none read from the configuration:
runs >= 1000; stale_runs_by_schedule >= 100; stale_runs_observed >= 100 (the mirror saw the run or its action stale in arm A0);
boundary_refusals_a0 >= 1 (**the baseline met a stale attempt at the boundary**, owner decision) and boundary_refusals_a1 >= 1;
cancelled_runs >= 100; effect_kinds_covered >= 4; fresh_runs_completed >= 100; generation_stale_cases >= 50;
injection_positions_covered >= 5; control_trips >= 5; wasted_units_a0 >= 1 (without work after the write the comparison with the
baseline would be vacuous); noop_writes_observed, retries_replayed, restarted_runs >= 1. Fifteen rules.

Falsifiers: stale_actions_executed > 0; stale_effect_attempts_committed > 0; stale_checkpoint_admissions > 0; fresh_actions_refused > 0;
false_cancellations > 0; missed_cancellations > 0; fresh_runs_starved > 0; cancel_contract_violations > 0; boundary_arm_divergences
> 0; double_executions > 0; executor_vs_ledger_mismatches > 0; mirror_drift > 0; cancel_latency_rounds_max > 1;
wasted_units_a1 > wasted_units_a0; wasted_units_a1 > 0; product_panics > 0; product_hangs > 0. Seventeen rules.

Rules that hold by construction or duplicate another (guards against a broken harness, not evidence): `runs >= 1000` (the number of
scheduled cases), `stale_runs_by_schedule` (the schedule's own count; `stale_runs_observed` is the mirror's),
`injection_positions_covered >= 5` and `effect_kinds_covered >= 4` (observed, but the schedule draws every position and effect in
every seed), `control_trips >= 5`. `wasted_units_a1 <= wasted_units_a0` and `wasted_units_a1 <= 0` are **named duplicates**: they restate the cancellation contract
(a cancelled run takes no step on a replaced snapshot), `missed_cancellations` is the independent check of the same contract, and
they are not independent evidence and not a compute saving. The first is kept because it is the owner's wording. `cancel_latency_violation` and
`wasted_work_violation` restate three rules as counts for the mutation checker and are not rules of their own.

## 2. Cases

1500 cases per seed (`cases_per_seed`), drawn from the seed alone (`plan.rs`). A case is one run of 2 to 6 rounds of Pod requests
ending in an action whose effect cycles through Read, Mutation, External, Irreversible by case index, and at most one writer:

| class (per mille) | writer | where it acts |
|---|---|---|
| never (150) | none | |
| bump-early (300) | a host write of a key nobody wrote: the revision moves | before round 1; mid-run (before round k, 1 <= k < rounds); just before the action (before the step that hands out the action) |
| bump-late (200) | the same | between ActionReady and `authorize_action`; between `prepare_execution` and `execute_prepared` |
| noop (100) | a host write of a key to the value it holds: the revision does not move | any of the five positions |
| lifecycle (250) | the target's generation 1 is superseded by 2 or revoked (half each): no semantic revision moves | any of the five positions |

Each case runs once in arm A0 and once in arm A1, each on a fresh `PtrRuntime` (a host semantic grant that allows host writes, the
target capsule at generation 1, the capability and all three effects permitted, one execution session whose single grant covers the
case's action scope, the Pod and the verifiers), after one base host write (so that the no-op writer has a key to rewrite).

The model backend is scripted (`world.rs`): the case's rounds of Pod requests, then an `ActionReady` of the case's effect stamped
with the revision of the context the backend was last shown (a model that re-reads its snapshot every round: what
`ModelResumeRequest.revision()` makes possible), then `Finished`. The Pod echoes its input and counts its invocations; the verifiers
pass. Nothing about a real model is measured.

The scheduler is a deterministic virtual-time event queue (`vtime.rs`, ordered by `(tick, seq)`): the run's steps are 10 ticks
apart, a writer acts 5 ticks before the event it precedes, and the three boundary checkpoints follow the last step 10 ticks apart. It
is cooperative: the runtime does one step, then the scheduler's next event runs. No clock is read.

**Host policy** (the same in both arms): a run that is cancelled, or whose action the boundary refuses as stale (revision or permit)
while the generation is live, is begun again once on the current revision (a fresh request) and its action goes through the boundary
under the case's at-most-once key; the boundary pass calls `authorize_action`, `prepare_execution_once` and `execute_prepared`
without stopping at the first refusal (as a careless host would, so every layer is asked about every stale action); after an
execution the same key and action are offered once more and must be answered from history. A host does not begin again on a
generation that is gone.

## 3. Owner decisions of 2026-10-09 (binding; verbatim)

- S002: refactor run_resumable_inner into a steppable run (existing entry points stay wrappers, behaviour unchanged with cancellation off); coarse revision-granular cooperative cancellation before the effect boundary only; boundary remains the safety barrier; gated claim: cancellation never weakens the boundary and cuts wasted work; coverage also requires the baseline to show >= 1 stale attempt reaching the boundary; invariant review before freeze; do LAST.

## 4. Arms and oracle

- **Arm A0 (the baseline).** `semantic.cancel_stale_revisions = false`: the run goes on with its next snapshot whatever
  happened, and the hard boundary alone refuses what is stale. The existing blocking entry points are this arm's loop (a unit test
  in `ptr-runtime` compares them with a verbatim copy of the loop they replaced over 14 scripted backends x 4 round budgets x 5
  Pod/verifier outcomes x both settings).
- **Arm A1 (cancellation on).** The default setting: before a step does any work it compares the store's revision with the revision
  the run is bound to (the revision its own latest write produced) and, if they differ, ends the run as `RunStep::Cancelled`.
- **Control (bypass, not an arm).** For each effect kind the executor's log is fed a stale action directly (a revision one behind, a
  generation that is not live) and a fresh one; the mirror must flag them stale (revision), stale (generation) and fresh. And a
  stale `EffectAttempted` is committed directly and the ledger scan must count it, while it counts none on a clean ledger. Six
  checks, five trips (`control_trips`). A control that does not trip is exit 2. The harness cannot hand the real executor a
  dispatch, so the bypass feeds the same logging function the real executor calls; it tests the oracle, not the boundary.
- **The revision mirror** (`oracle.rs`) holds the harness's own count: it starts at 0 for a fresh runtime, adds one for the run's
  ingest, one for each promoted Pod output whose bytes differ from what its key holds, one for each host write of a key that holds
  another value or none, and nothing for a lifecycle record or a rewrite of the same value; it also holds the revision the active run
  is bound to (its own latest write) and the target's live and revoked generations. It never reads the runtime's revision into its model.
  The runtime's revision is compared with it after the run's ingest, after every step, after every injection. Before every step the
  oracle decides from the mirror alone whether the run is stale (`run_bound != revision`) and so whether arm A1 must cancel; the step's
  result is compared with that decision (`missed_cancellations`, `false_cancellations`). Arm A0 cancelling at all is a false
  cancellation.
- **Stale at a checkpoint**, from the **committed ledger replayed to that point** (`oracle::LedgerState`): the action's revision is not
  the revision of the last committed delta (checked first), or its generation is not the live one there or was revoked. The mirror's
  verdict is compared with it at every checkpoint and a difference is `mirror_drift`; so `stale_actions_executed` and
  `stale_checkpoint_admissions` are never judged against a lagging mirror, and a runtime write outside the harness' view is still
  caught. A step that returns `Cancelled` after promoting a Pod output (a defective runtime) is applied to the mirror from the Pod's
  invocation count, so one miscounted step does not offset the rest of the case.
- **The ledger scan** (`oracle::scan`) replays the committed records in order, tracking the revision (each committed delta's own
  `revision`) and the target's generations (committed, superseded, revoked) from the records alone, and counts the `EffectAttempted`
  records whose revision or generation was not the live one at their place in the replay. It uses no runtime state.

## 5. Metrics (all JSON integers)

| metric | definition |
|---|---|
| runs | cases executed (each in both arms) |
| stale_runs_by_schedule / stale_runs_observed | cases the schedule labels stale (a revision-moving writer) / cases in which the mirror saw a run or action stale (arm A0) |
| stale_actions_executed (hard) | executor calls made on an action the ledger replay judged stale (revision or generation) just before the call, both arms |
| stale_effect_attempts_committed (hard) | `EffectAttempted` records the ledger scan judges stale, both arms |
| stale_checkpoint_admissions (hard) | checkpoints (authorize, prepare) that answered Ok for an action the ledger replay judged stale |
| fresh_actions_refused (hard) | checkpoints that refused an action the ledger replay judged fresh |
| stale_boundary_attempts_a0/a1, stale_action_rate_permille_a0/a1 | boundary passes that met a stale action, per arm; 1000 x stale executions / stale passes |
| boundary_refusals, _a0, _a1 | stale passes that ended in a refusal (once per pass), per arm |
| cancelled_runs | steps that returned `Cancelled` (arm A1; any in arm A0 is also a false cancellation) |
| false_cancellations (hard) | cancellations the oracle did not expect (mirror says the run is not stale, or arm A0) |
| missed_cancellations (hard) | steps of arm A1 on a run the mirror says is stale that did not cancel |
| fresh_runs_completed / fresh_runs_starved (hard) | runs the schedule leaves fresh (never, noop; and every restart on a live generation) that ended executed / did not |
| cancel_contract_violations (hard) | a cancelled run that carries an action, keeps a number of observations other than the rounds completed, reports another request id, bound or current than the mirror's, whose step invoked a Pod, wrote to the ledger or moved the revision, whose `RuntimeEvent::RunCancelled` was not emitted exactly once, or whose `into_error` is not `RunCancelled{bound, current}` |
| boundary_arm_divergences (hard) | cases in which both arms handed the boundary an action and the actions or the three checkpoint outcomes differ |
| cancel_latency_rounds_max | steps started since the writer acted, up to and including the cancelling step (1 = cancelled by the very next step) |
| cancel_latency_violation (hard) | 1 iff the latency exceeds 1 |
| wasted_units_a0/a1 | steps (a Pod round; the step that hands out the action) a run takes after a bumping foreign write replaced its snapshot, until the run ends, from the Pod's invocation count and the step results; a cancelled step that did work counts its work |
| wasted_work_violation (hard) | 1 iff wasted_units_a1 > wasted_units_a0 or > 0 (a named duplicate of the cancellation contract, not independent evidence) |
| total_units_a0/a1, net_units_a1_minus_a0 | every step of every run of the arm, restarts included, and arm A1's total minus arm A0's with its sign (headline numbers, descriptive, ungated) |
| double_executions (hard) | executor calls beyond one per case and arm |
| executor_vs_ledger_mismatches (hard) | per case and arm, the difference between the executor's calls and the `EffectAttempted` records |
| mirror_drift (hard; `stale_actions_executed` and `stale_checkpoint_admissions` are interpretable on their own, but read `mirror_drift` first) | the runtime's revision differs from the mirror's; the mirror's staleness verdict differs from the ledger replay's; a run that does not take the shape the scripted chain implies; a run the runtime refuses; a case in which the schedule's label (stale / generation dead) and the mirror's observation disagree |
| effect_kinds_covered, injection_positions_covered | distinct effects at a boundary checkpoint; distinct positions at which a writer acted |
| generation_stale_cases | cases (arm A0) in which the target's generation was superseded or revoked in the runtime |
| control_trips, noop_writes_observed, retries_replayed, restarted_runs | coverage counters (observed) |
| product_panics, product_hangs (hard) | panics caught around a runtime call; 1 if the watchdog fired |

Timings (`timing_ns`) are never read by a rule. `hard_failures` is the sum of the sixteen hard counters named in
`tests/mutations.toml`.

## 6. Negative controls and mutation plan

The two controls are in section 4. `tests/mutations.toml` (run with `python3 scripts/mutation_check.py S002`; `--check` verifies the
anchors) plants twenty-one defects one at a time: eight in the hard boundary (revision compare removed; inverted; generation check
removed; the re-check of prepare/execute not requiring the current revision for reads; permit epoch and admission re-check both
removed; authorize_action admitting a stale revision when cancellation is off; the executor called twice; a settled key not answered
from history), eleven in the cancellation (check removed; inverted; only once the action is ready; the bound not rebased after the
run's own write; one step late; after the Pod ran; a cancelled run keeping its action; bound and current swapped; the setting
ignored; a panic; a hang) and two in the harness (the mirror not told of the writer's writes; the oracle expecting the opposite
cancellation). Every hard counter is expected by at least one. Equivalent mutants (not listed): removing only the permit epoch check
or only the re-check in `admit_effect` (defense in depth: each commit retires the permits prepared before it, so the re-check sees
what the epoch already guarded; the pair is listed). The two equivalent mutants, by name: `permit-epoch-check-removed` (the
`Arc::ptr_eq(&self.execution.epoch, &permit.epoch)` test in `admit_effect`) and `admission-recheck-removed` (the
`check_execution_action` call in `admit_effect`).

## 7. Exit 2 (the harness cannot run), and nothing else

A host write, a lifecycle commit or the setup that the runtime refuses (the writer's path is not what S002 tests); a negative
control that does not trip; the scheduler running dry or scheduling in the past; the watchdog thread failing to start; an argument that
does not parse. Everything the code under test does wrong (a run shape it does not take, a refused step, a wrong revision, a stale
execution, a panic, a hang) is a counter and a falsifier.

## 8. Weak metrics and what is not measured

- **Compute is a descriptive headline, not a claim of saving.** `wasted_units` counts the steps a run takes after a bumping foreign
  write replaced its snapshot, until it ends. A cancelled run takes none (it is cancelled by the very next step), so
  `wasted_units_a1 = 0` restates the cancellation contract and is the same condition `missed_cancellations` checks; the two wasted rules
  are named duplicates, not evidence of a saving. **Total compute is not claimed to fall**, and the dry run says the opposite:
  `net_units_a1_minus_a0` is positive (arm A1 discards the steps taken before the write and begins again, 1226 restarts in the seed-17
  dry run), because in the baseline a mid-run write usually still ends in an accepted action (the stub backend re-reads its snapshot and
  stamps the action with the revision it last saw; a model that carries its first snapshot's revision would have its action refused and
  the whole run wasted in the baseline). What cancellation buys here is that no step runs on a snapshot already known to be replaced, and
  that the run whose action *would* have reached the boundary stale (`just-before-action`) never does.
- **The boundary is exercised, not weakened, in both arms.** Stale attempts reach it in arm A0 for the writers acting after the
  last step, and in both arms for the writers acting after the run (between ActionReady and authorize; between prepare and execute)
  and for the lifecycle writer. `boundary_arm_divergences` compares the three checkpoint outcomes of both arms for the same action.
- **`stale_runs_by_schedule`, `runs`, `control_trips`, `injection_positions_covered` and `effect_kinds_covered`** hold by
  construction or nearly so (section 1).
- **The bypass control tests the oracle's detection**, not the boundary (section 4).
- Wall-clock time is recorded and never judged.

## 9. Stated limits (what this experiment does NOT show)

- **S002 makes no multi-run or interleaving claim.** It runs one run at a time per runtime. Runs that share a runtime cancel one
  another and a host that keeps beginning runs can keep every older run from finishing (livelock); `fresh_runs_starved` is about runs
  nobody else writes to.
- **No shipped host steps runs** (`ptr-server`, `ptrd`): the blocking wrappers force `CancelPolicy::Off`, so
  `semantic.cancel_stale_revisions` has no effect on any production path today. Generation staleness is deliberately not a
  cancellation trigger.
- The cancellation is **coarse and revision-granular**: any write the run did not make cancels it, including another run's output on the
  same runtime. Two runs interleaved on one `PtrRuntime` cancel each other (the older one is cancelled; shown by a runtime test, not by
  this harness, which runs one run at a time per runtime). A host that interleaves runs gets mutual cancellation and, with
  continuous arrivals, can starve all of them; `fresh_runs_starved` here is about runs nobody else writes to.
- The cancellation is **cooperative and before the effect boundary only**: a run that is never stepped is never cancelled, the check
  is at the top of a step and nowhere else, a write after the last step returned the run's action is the boundary's to refuse, and
  nothing is recalled after `EffectAttempted`. A step is not interruptible.
- A **lifecycle record moves no semantic revision**, so a superseded or revoked generation cancels nothing; the boundary refuses the
  action. The harness counts that case (`generation_stale_cases`) and the boundary's refusal; it does not claim the cancellation
  covers it.
- A cancelled run writes **no semantic state and appends no cancellation record**, only the in-memory `RuntimeEvent::RunCancelled`; its
  committed observations stay in the store. For input whose KV-reuse taint is Approximate the cancelled step records that taint, as any
  returned run does, which appends to the ledger; the harness uses Exact input only, so it does not exercise that.
- The mirror encodes "an identical re-promote does not bump the revision" (`Mirror::run_promotes`), a model of product behaviour; a change
  of that rule is caught only through `mirror_drift`.
- Synthetic: a scripted backend, one echoing Pod, passing verifiers, small semantic state, one process, one machine profile, one
  target and capability, a single writer per case. Real model latency, real Pods and concurrent writers are not exercised.
  `ptr-server` and `ptrd` still call the blocking wrappers and do not interleave or cancel runs.
- Revisions are checked against a mirror of "bumps iff the state changes"; the absolute numbers are not independently specified
  anywhere else. The base write, the run's ingest and the promoted outputs are the only own writes the mirror models.

## 10. Open points for the invariant review (before the freeze)

1. Invariant 2 ("a reasoning run is bound to one immutable semantic revision unless explicitly resumed on a new snapshot"): the
   blocking loop has always resumed on the new snapshot every round. Is cancelling a run that would have resumed on the new snapshot
   the intended reading, or only an opt-out (`semantic.cancel_stale_revisions = false`) of a default that is stricter than before?
   The default is `true`, but the blocking wrappers pass `CancelPolicy::Off`, so no existing entry point changed.
2. The bound is rebased from the revision `promote_verified_pod_output` returns. Under `&mut self` that revision is the current one;
   a future concurrent writer inside a step would be absorbed silently.
3. A cancelled run writes no semantic state and appends no cancellation record (for Approximate-taint input it records the taint, as any
   returned run does). Should cancellation be a ledger record (replay, audit)? It would need a new `LedgerEvent` variant.
4. The coarse check lets one run's output cancel another's (section 9); a keyed or key-granular check would need dependency
   tracking the semantic host exposes but this change does not use.
5. The effect boundary is untouched (no change to `admit_effect`, `check_execution_action`, `ptr-security`); the review should confirm
   that nothing in `stepped.rs` is reachable after `EffectAttempted` and that `authorize_action` and the execution path do not
   consult the policy.

## 11. Proposed INVARIANTS note (text only; docs/INVARIANTS.md is not edited here)

Proposed addition under invariant 2, for the owner to accept or reject at the review: "A stepped run (`PtrRuntime::begin_resumable_run`,
`step_resumable_run`) with `semantic.cancel_stale_revisions` on is ended before its next round when the semantic revision moved for a
reason other than its own writes: it is cancelled instead of silently resuming on the new snapshot. The blocking entry points keep the
older behaviour (they resume on the new snapshot every round) because nothing else can write while they hold the runtime. The
cancellation is a saving of work and never a safety check: invariant 7 is enforced by the effect boundary alone, which refuses a stale
action whatever the policy, and nothing in the cancellation is consulted after a run has returned its action."
