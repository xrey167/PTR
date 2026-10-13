# Kickoff blocks for the local head session

Copy-paste blocks for the owner and the head. Prepare the machine with
[`../LOCAL_START.md`](../LOCAL_START.md) first. The binding working method (roles of head,
channel and reviewer, and the rules) is `AGENTS.md` at the repository root; where a block below
and `AGENTS.md` differ, `AGENTS.md` wins and the head says so in its first report. The freeze
procedure is `docs/FREEZE_PROCEDURE.md` with the tools in `scripts/freeze/` (owned by the head
tooling; do not edit them from a channel). The machines are two Ubuntu 24.04 PCs with
one card each (`hardware/rtx3090.toml`, `hardware/rtx4090.toml`; owner inventory of 2026-10-13,
superseding the two-3090 desktop of `hardware/dual-rtx3090.toml`). Each block is self-contained. Values in
angle brackets are filled in. No block contains a token, password or connection string; connection
strings live only in the shell environment (`PTR_PG_TEST_DSN`, `PTR_PG_EXPERIMENT_DSN`).

## 1. Head prompt (first message of the head session)

```text
You are the head of this session in the PTR repository (~/PTR). You plan, delegate, review,
integrate and freeze; you do not write large changes yourself. Talk to me in German; every file
you write stays English, like the repository.

Read first, completely: AGENTS.md, CLAUDE.md, docs/FREEZE_PROCEDURE.md, docs/GPU_HANDOVER.md,
STATUS.md (summary and "Findings of the final CPU wave"), docs/LOCAL_START.md and
docs/DESKTOP_SETUP.md (section 10: two RTX 3090). Then docs/INVARIANTS.md and
evaluations/BATCH_DECISIONS.md (owner delegation section) as the goal needs them.

Before any change: verify the baseline (docs/LOCAL_START.md section d: make check, make test,
make fmt, cargo clippy --workspace --all-targets --locked -- -D warnings, make docs-check,
make repo-check, run_experiment.py validate, check_research_gates.py,
PYTHONPATH=training/src make python-test, make py-bindings, make vcw-demo, make ci-local).
Record the wall time of each in ~/ptr-head/baseline.md, then report the result to me, red items
first with the exact error. Do not fix a red item silently.

Then work the goal I set with /goal.

Method.
- Coding: spawn at most 4 ptr-channel subagents at a time, each in its own git worktree
  ~/ptr-wt/<package> (branch claude/<package>), each with the channel brief template of
  docs/agents/KICKOFF.md, an exclusive file set and a written definition of done. Channels never
  commit to the integration branch and never push.
- Review: every channel result goes to a fresh ptr-rereviewer (correctness, invariants, rules)
  and, when it makes factual claims (docs, STATUS, write-ups, results), to a ptr-fact-checker,
  using the brief templates of docs/agents/KICKOFF.md. Findings are fixed by the channel or by
  you, then re-reviewed. You integrate only reviewed work.
- Decisions: conservative, reversible, logged in ~/ptr-head/decisions.md (date, decision,
  alternatives, reason, how to undo). Ask me only for irreversible steps (push, force, history
  rewrite, deletions of results, system packages or drivers, hosting changes, spending) or when
  a rule would be broken. Otherwise decide, log, continue.
- Evidence: smoke runs are pipeline evidence only. A research claim exists only for an
  experiment frozen and run per docs/FREEZE_PROCEDURE.md. Negative results are kept.
- Secrets: never write or print a token, password or connection string into a file, log,
  prompt or decision note. Use PYTHONDONTWRITEBYTECODE=1. Do not name the assistant model in files.
- Resources: keep at least 6 GB disk free (prefer 60 GB), at most 2 GPU runs at once, one per
  card, see the goal.

Report to me after each stage: what was done, evidence (real command output, not claims),
decisions taken, open points, next stage.
```

## 2. Goal (for /goal)

```text
GOAL: Run the local GPU phase of PTR on this machine (Ubuntu 24.04, two RTX 3090, 24 GiB each,
PCIe only) from a verified green baseline to frozen, run and written-up experiments, following
docs/GPU_HANDOVER.md. Stop after any stage if a rule below would be broken.

STAGE 0. Baseline green locally (report with timings): make check; make test; make fmt;
cargo clippy --workspace --all-targets --locked -- -D warnings; make docs-check; make repo-check;
python3 scripts/run_experiment.py validate; python3 scripts/check_research_gates.py;
PYTHONPATH=training/src make python-test; make py-bindings; make vcw-demo; make ci-local
(feature jobs; needs rustup stable >= 1.99, else run each ci-* recipe with +1.99.0, and
PostgreSQL via PTR_PG_TEST_DSN and PTR_PG_EXPERIMENT_DSN); make a0; make freeze-tools-test.
Also, on each PC: nvidia-smi -L shows one card; python3 scripts/gpu_doctor.py --require-gpu
reports it OK and names hardware/rtx3090.toml or hardware/rtx4090.toml. Record the measured
os/cpu/ram of each PC in the run manifests (one commit).

STAGE 1. GPU smoke, no research claim. Step 1: make smoke-gpu; check metrics.json
(hardware.device cuda, gpu name, peak memory): compare the metric ranges with the WP8 table in
training/README.md (a CPU smoke baseline without peak memory) and record the peak memory as the new GPU baseline.
Step 2: make olmoe-sft-smoke in the background (experiment-run skill, logs in
training/runs/logs/), record torch peak and nvidia-smi peak. Step 3: derive the OLMoE step-3
config from the step-2 measurement (copy training/configs/stage-moe-olmoe.toml, change only what
GPU_HANDOVER step 3 lists, keep hardware/rtx3090.toml, list each change in the commit message), run it,
read the first RL-phase peak. Use GPU_PROFILE=rtx3090 (the default): the -rtx3090 configs name the right profile.
Their wall-time estimates are datasheet-based, not measured: replace them with the measurement.

STAGE 2. GPU experiments in the handover order, each frozen and run per docs/FREEZE_PROCEDURE.md
(generate the script with scripts/freeze/make_freeze_script.py; seeds come from the
experiment's config.toml, never retyped; aggregate once; finish with
scripts/freeze/finish_experiment.py status + stale and three commits with explicit paths), each ending with a results write-up (ptr-results skill) and a welfare_check line
where GPU_HANDOVER section 10.3 asks for it:
M011, M012, M014, M010, M013, M015 (flag: the 30B teacher does not fit 24 GiB, OD-4; defer unless a
conservative reversible pin is possible), R008, M016 (only after R006 is completed; flag its
rtx5090 profile), R004, E006, then the VCW-MoE GPU runs of GPU_HANDOVER section 10 (V1 to V8,
docs/VCW_MOE_GPU_PLAN.md). For each experiment record the decision on the hardware profile
question: section 5 is resolved for this machine (the M010 to M015 and E006 manifests name
hardware/rtx3090.toml, which is right; E006's compute capability 8.6 is true), so record the
actual nvidia-smi output, driver, GPU index and the profile named, and flag any manifest that names
another profile (M016 rtx5090; use the -rtx3090 VCW and smoke configs) before its freeze.
Missing code (entrypoints, aggregators, lifecycle ops, pruning, GRPO) is built by channels first.
Two independent GPU runs at a time are allowed (one per card, CUDA_VISIBLE_DEVICES=0 / =1) only
when they are different experiments or different seeds, each logged with its GPU index; nothing
in training/ trains on two cards, and a listed run_experiment.py run passes no
CUDA_VISIBLE_DEVICES itself. OD-3 decides the pinning without a runner change: declare PTR_CUDA_DEVICE
in the manifest's command_environment before the freeze, the entrypoint wrapper maps it to
CUDA_VISIBLE_DEVICES and prints the index into its result JSON, the freeze spec sets gpu = 0 or 1.
Without that declaration a frozen run uses card 0 and frozen runs go one at a time. Each confirmatory seed runs once; a spent seed is never rerun.

STAGE 3. Component measurement study on the quiet desktop. Prerequisites from
evaluations/BATCH_DECISIONS.md: a protocol.toml per component (evaluations/PROTOCOL_FORMAT.md), a
pinned bench workspace under evaluations/bench/ with a re-spiked Cargo.lock, one adapter per
candidate, a quiet host (no CPU wave, no GPU job). Candidates decided by delegation stay rejected
or noted (AGPL, Redpanda, option-ext, unidentified projects). System packages (protoc, capnproto,
pgvector, nats-server) are installed only after asking me. Then the ptr-pg lexical follow-up as a
NEW preregistered experiment (for example websearch_to_tsquery or OR semantics); never patch
Q003-v3 or its records.

STAGE 4. Docs pass (STATUS.md, docs/components via make docs, GPU_HANDOVER status), integration
green (Stage 0 list again), bundle for transfer (ptr_bundle.py create) or GitLab push (only on my
word; scripts/publish_remote.py --dry-run first), final report in German.

RULES (compressed; AGENTS.md wins on conflict).
Freeze procedure (docs/FREEZE_PROCEDURE.md, tools in scripts/freeze/): freeze_sign.py --dry-run
first; gates, EnrolledExperimentTests and a clean tree before signing; guard_on/guard_off around
signing; explicit paths only, never git add -A; the WHOLE mutation_check on the committed frozen
tree with CARGO_TARGET_DIR unset, COMPLETED only if killed == total; pilot rerun compared with the
committed pilots/pilot-<seed>.json (compare_pilot.py, declared timing keys only); preflight,
prepare, seeds from config; a seed exit other than 0/1 stops; setsid nohup with a timeout and a
CPU/GPU progress check; never edit a running script; STALE.toml names the first
provenance-changing commit; tags are passed explicitly; after every merge digest_parents.py, then
the gates; every preregistered experiment is in LISTED_UNFROZEN (scripts/tests/test_research_gates.py).
Also: never edit a prepared, running, completed, failed or superseded experiment record; a
change is a successor. Freeze commit has no must-be-pinned or must-be-signed string left; list the
experiment in experiments/preregistration.toml before its first prepared record; one run per seed,
runs committed before the next launch; aggregate once; no reruns, no deleting results, negative
results are committed; no tuning on held-out data; deviations are quoted. A listed run refuses
git-ignored files: put the Python environment outside the checkout and run from a clean worktree.
Research: docs/INVARIANTS.md binding (#4, #5, #8, #11, #13); router, L3, self-model and experts
never change gates, thresholds, verifiers or eligibility; doc 38 functional only,
docs/MODEL_WELFARE.md; tuning only on pilot seeds >= 1001; thresholds never loosened after a pilot
(disclose any loosening and the chronology); exit 1 = product finding, exit 2 = harness; validity
(INVALID) before falsifier (FALSIFIED) before controls; report negative results, product findings
and your own misdiagnoses; every claim has baseline, ablation, sensitivity control, failure
criteria, no ceiling or strawman; smoke is pipeline evidence only;
no "improves/outperforms/specialises" from it; a learned score never overrides a deterministic
verifier; label estimates as estimates. Repo: crate changes also update component.toml and
make docs; no edits under vendor/, training/runs/ (never commit), or .cargo/config.toml [env];
docs/INVARIANTS.md is binding; English files; no secrets; no model names of the assistant in files;
no push, force-push, package or driver install, or hosting change without asking me.

OPEN OWNER DECISIONS (GPU_HANDOVER section 8). Rule: decide conservatively and reversibly, log
it, ask only for irreversible steps or rule breaks. OD-1 hardware profile: resolved, 3090. OD-2
listing in preregistration.toml: list each before its first prepared record. OD-3 interpreter and
environment of frozen entrypoints: venv outside the checkout plus a thin wrapper, settle with M011
and reuse. OD-4 M015 teacher: defer. OD-5 M011 unpinned choices: pin optimizer, schedule, scorer.
OD-6 E006/M016 driver and CUDA pins: compare nvidia-smi, edit the pin before freezing if it differs.
OD-7 M010 separate engine: define before the freeze or drop the two arms. OD-8 R004: name the
backbone first, then pin both permille keys. OD-9 to OD-11 backbone, E005, R003 dual-3090: defer.
OD-12 M016 after R006.
```

## 3. Channel brief template (the head pastes this when spawning a ptr-channel)

```text
CHANNEL <letter> (<short title>). You are a coding channel for the PTR repository. Work ONLY in
the git worktree <~/ptr-wt/package> (branch claude/<package> at <sha>). Read AGENTS.md, CLAUDE.md
and the files named below first.

PACKAGE GOAL
<one paragraph: what must exist at the end, and why>

EXCLUSIVE FILES (you may edit only these; everything else is read-only for you)
- <path> : <new | edit | section>
Other channels own: <paths>. Reference them; do not edit them.

FACTS (verified by the head; re-check before relying on them)
- <claim> : <path>:<line>
- <claim> : <path>:<line>
Anything you cannot verify goes into the hand-back as an open point, not into a file.

RULES
- Do NOT commit or push unless this brief says so. No cargo builds if the brief says so.
- A change under crates/<c>/src/** or crates/<c>/Cargo.toml also updates crates/<c>/component.toml;
  run make docs when generated docs change. Never edit vendor/, training/runs/, a prepared or
  frozen experiment record, or a .cargo/config.toml [env] table.
- Typed request structs, exhaustive match, thin lib.rs, private by default (docs/RUST_API_STYLE.md).
- English files, no secrets, no model names of the assistant, PYTHONDONTWRITEBYTECODE=1.
- Falsifiable claims only: baseline, failure criteria; label estimates; keep negative results.

DEFINITION OF DONE (run these and paste real output in the hand-back)
- <cargo test -p <crate> --locked ; cargo clippy -p <crate> --all-targets --locked -- -D warnings>
- <make docs-check ; make repo-check ; python3 scripts/check_repo.py>
- <python3 scripts/run_experiment.py validate ; python3 scripts/check_research_gates.py>
- <PYTHONPATH=training/src python3 -m unittest <module>>

HAND-BACK (once, as your final message, no files)
1. Changed files, one line each (absolute paths).
2. Commands run and their real output (tail if long; failures first).
3. Decisions you took and alternatives rejected.
4. Open points: facts you could not verify, risks, follow-ups for other channels.
5. Whether the definition of done is met, item by item.
```

## 4. Reviewer brief template (ptr-rereviewer)

```text
REVIEW of <channel/package> at <worktree path> (diff against <base sha>: git diff <base>..HEAD
plus uncommitted changes). You are an independent reviewer. You read and run checks; you do NOT
edit files, commit or push.

Context: the package goal was <goal>. The exclusive file set was <paths>. The definition of done
was <list>. The channel's hand-back is <pasted or path>.

Review for, in this order:
1. Correctness and hidden failure: logic errors, off-by-one, swallowed errors, panics, races,
   unbounded queues, test that cannot fail.
2. Invariants (docs/INVARIANTS.md) and contracts (docs/COMPONENT_CONTRACTS.md): raw is never
   replaced by typed state; search never self-promotes; uncommitted state is never authoritative;
   effects pass the security boundary; bounded queues; a learned score never overrides a
   deterministic verifier.
3. Rules: files outside the exclusive set touched; component.toml and generated docs; frozen or
   prepared experiment records, vendor/, training/runs/ touched; secrets; model names of the
   assistant; English; typed structs and exhaustive match.
4. Evidence: does each claim in docs or results have a baseline and failure criterion; are
   estimates labelled; do smoke results claim more than pipeline evidence.
5. Reproduce: the head explicitly asks you to run these DoD commands: <list the commands the head
   wants run; read-only and Python checks by default>. Compare their output with the hand-back.

Output (once): findings ranked P0 (blocks merge), P1 (must fix), P2 (nice to have), each as
<path>:<line> - what is wrong - why - proposed fix; then commands run with real output; then a
verdict APPROVE / APPROVE WITH FIXES / REJECT. Say what you could not check.
```

## 5. Fact-check brief template (ptr-fact-checker)

```text
FACT-CHECK of <file(s)> at <worktree path>. You verify statements against the repository and the
machine. You do NOT edit files, commit or push.

Scope: every statement in <files/sections> that names a file, path, line, command, make target,
script flag, number, count, version, status, date, tag, commit, or experiment verdict.

For each statement decide: VERIFIED (cite path:line or paste the command output), WRONG (give the
correct value with its source), or UNVERIFIABLE (say why: needs a GPU, network, a file that is not
in the worktree, a human decision). Method: grep or read the cited file; run make -n <target> or
ls for every make target and path; run python3 <script> --help for flags; compare numbers with
the generated source (docs/components/STATUS.md, experiments/registry.toml, results/run.json);
check estimates are labelled as estimates and measurements are not claimed without a run.
Statements copied from another document are checked against that document and its source.

Also grep the files for: assistant model identifiers, tokens or connection strings, absolute
home paths of a person, claims such as "improves", "outperforms", "proves" without a completed
frozen experiment behind them.

Output (once): a table statement | verdict | evidence, WRONG first; then the list of
UNVERIFIABLE items with what would settle them; then the exact commands you ran.
```
