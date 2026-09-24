# Recommendations — what to do next, and what to do better (2026-09-24)

This document comes from a full mapping pass over the repository at `e93ed99`.
The pass read every workspace area. Every test suite that CI runs was executed
locally (see "Evidence base" at the end), and the load-bearing claims below were
re-checked against the code. It follows the conventions of
[`OPEN_ITEMS_PLAN_20260921.md`](OPEN_ITEMS_PLAN_20260921.md): every defect names
the file and line where it can be seen. A companion document,
[`PROJECT_MAP.md`](PROJECT_MAP.md), maps the whole repository.

*Revised the same day, after an independent verification pass:*
- items 1.2, 2.4, 3.5 and 3.7 were corrected;
- rows 4.15–4.20 were added;
- the owner decisions were renamed O1–O5.

## 1. The diagnosis in one paragraph

PTR has built a strong **hard shell** and has barely started on its **soft core**.
About three quarters of the Rust code (roughly 25.7k of 33.9k lines under `crates/`)
lives in three places:

- the ledger (`ptr-ledger`, 8.4k lines)
- the runtime's authority, audit and snapshot machinery (`ptr-runtime`, 10.2k lines)
- the network wires (`ptr-podwire`, `ptr-execwire`, `ptr-cluster`, `ptr-net`, 7.1k lines)

That code is real, carefully reasoned and well tested. The ten cognitive crates that
carry the project's actual thesis add up to about 2.5k lines, roughly 7%. Only
`ptr-semdb` among them is substantial. Five of the ten (`ptr-router`, `ptr-ingress`,
`ptr-memory`, `ptr-search`, `ptr-feedback`) have no consumer at all.

The thesis is that typed semantic state, epistemic metadata and operator routing
make a model reason better. Nothing has tested that thesis yet:

- no model has been trained
- the only model backend echoes its input
- the only neural code, `model/burn-a0`, is a single-layer probe that no runtime
  path reaches
- 20 of 21 experiments are `planned`
- no component evaluation has compared two candidates

The repository's own [priority ledger](PRIORITIES.md) already says so:

> The next scientific gate is not a larger model. It is a matched, reproducible
> M001/M002/M003/M004 experiment … Do not grow infrastructure indefinitely before
> testing that architecture.

The git history since then has mostly grown infrastructure:

- PR #15 to #24: anchors, compaction, erasure, peer admission, cluster integrity,
  PodWire, ExecWire and address authority
- PR #27 to #31: a continued-pretraining scaffold for a model that has no corpus yet

**The single most important recommendation is to follow that sentence now: freeze
infrastructure growth and run the first real falsification test of the core
thesis.**

## 2. What is already good (keep doing it)

- **Test discipline.** 425 default-feature tests, 155 feature-gated backend tests,
  43 Burn A0 tests and 233 Python tests. All of them pass on `e93ed99`.
- **Honest status vocabulary.** `STATUS.md` says "not yet evidence of superiority".
  Experiments carry falsification criteria. `check_research_gates.py` stops M001–M005
  and E002 from being *declared* running or completed while their baselines are
  unpinned. It is a CI check on the manifest, and it does not stop an execution.
- **Docs-as-code.** `component.toml` feeds generated README blocks and
  `docs/components/STATUS.md`, and CI fails when they are stale.
- **Supply-chain hygiene.** Vendored crates are hash-inventoried and each has a
  retirement record. Notices are diffed against every lockfile, and cargo-audit and
  cargo-deny run over every owned workspace.
- **Precise engineering on durability.** Examples are the hash-chained PTRLOG02 log,
  anchored recovery, at-most-once execution keys and the fencing token.

The problem is not quality. It is **where the effort has gone**, and how far the
prose has run ahead of the code.

## 3. What to do next — in order

### Step 1 — Test the thesis: make M001–M004 real experiments

The goal is the first piece of evidence for or against the core idea. Everything
in this step runs on the CPU `flex` backend that A0 already uses.

| # | Task | Why / evidence |
|---|---|---|
| 1.1 | Implement the ablation switches in A0: no semantic slots, no typed attention bias, no latent recurrence, no router. Read them from `model/configs/ablations.toml`. | That file lists five ablations, but no code reads it. A0 exposes only `latent_steps`, which defaults to 0 (`model/burn-a0/src/lib.rs:349`). |
| 1.2 | Replace the M001 pilot with a real M001–M004 run: at least 5 seeds, matched parameter count, the two imported hash-verified bundles plus a synthetic set. Use a task that the ablated arm *could* in principle learn, and give both arms the same payload information. | The recorded pilot (`experiments/model/M001-semantic-slots/results/pilot_run.json`) was re-recorded in `ebe475c`. Since `0fcf7ab`, `examples/m001_pilot.rs` gives distinct per-row slot payloads to *both* arms, so a re-run is confounded. By construction, 0.25 is the most the ablated arm can reach and the typed arm is handed the label (`m001_pilot.rs:15-38`). The pilot is a plumbing check, as its own README says. |
| 1.3 | Run the experiments through `scripts/run_experiment.py`, and add an aggregation step that turns per-seed records into `metrics.json`. | No committed evidence has come through the runner so far. Per-experiment scripts overwrite fixed file names (`experiments/lifecycle/L001-revocation-crash/aggregate.py:21,38`), and `required_artifacts` expects a `metrics.json` that the runner never produces (`scripts/run_experiment.py:241`). |
| 1.4 | Fill in `hardware/default.toml` with measured values, and record the profile *contents* in the evidence, not just its path. | 20 of 21 experiments point at a profile whose fields are all `unspecified`. |
| 1.5 | Make one benchmark suite real, either `semantic-typing` or `operator-routing`: data, scorer, harness, and a reference from the experiment manifest. | All 10 suites under `benchmarks/` are metric-name lists that nothing references. |
| 1.6 | Pin the plain-model baseline's backbone revision. | `research/baselines/plain_model` is unpinned. Until it is pinned, no M00x comparison against a plain model is valid. |

**Result of Step 1:** a statement such as "semantic slots improve operator-routing
accuracy by X ± Y over the ablation across 5 seeds", or an honest negative result
recorded in `research/falsification/`. Either one is worth more than any further
infrastructure.

### Step 2 — Connect the path end to end (open item C1)

`OPEN_ITEMS_PLAN_20260921.md` names C1, "connect actual semantic payloads to the
model", as "the largest open piece in the repository". It is still open.

| # | Task | Why / evidence |
|---|---|---|
| 2.1 | Add one real `InferenceBackend` behind `ptr-model-api`. An OpenAI-compatible HTTP client covers vLLM, SGLang, llama.cpp server and Ollama with one adapter. | The only backend is `ReferenceEchoBackend`, and it is what `ptrd` serves (`crates/ptr-server/src/lib.rs:80`). |
| 2.2 | Make `ModelRequest` carry the snapshot's semantic content (slot values), not only a revision number and raw text. | `crates/ptr-model-api/src/request.rs:4-9`. Invariant 2, "a reasoning run is bound to one immutable semantic revision", is only nominal while the model never sees the snapshot. |
| 2.3 | Wrap `ptr-burn-a0` as an `InferenceBackend`. This joins the A0 seam in production rather than in two half-tests in two workspaces. | No crate depends on burn-a0. The seam exists only in `crates/ptr-runtime/tests/neural_admission.rs` and `model/burn-a0/tests/semantic_payload.rs`. |
| 2.4 | Give `ModelEvent::ActionReady` a typed `ActionIr` payload, then consume `ActionReady` and `OperatorRequested` in the runtime loop and route actions into the existing execution gateway. | The runtime reacts only to `PodRequested` and `Finished` (`crates/ptr-runtime/src/lib.rs:293-301`). The gateway only ever receives an `ActionIr` built by host code. `ActionReady` carries only `operation: String` (`crates/ptr-model-api/src/event.rs:26-28`), so it cannot express an action. This is a protocol and type change, not only runtime glue. |
| 2.5 | Give `ptrd` a durable mode (`open_durable`), and make it fail closed on an unreadable config file. | `ptrd` always builds `PtrRuntime::new` in memory and falls back to defaults when the file fails to parse (`bins/ptrd/src/main.rs:29-32,44`). Restarting `ptrd` loses all state. |

### Step 3 — Make the documents tell the truth again

This step is cheap and mostly mechanical, and it improves trust in everything else.
The mapping pass recorded about 190 places where documentation and code disagree.
Most fall into a few patterns, and the first three items below fix a pattern rather
than a single instance.

| # | Task | Evidence |
|---|---|---|
| 3.1 | Count `#[tokio::test]` as well as `#[test]` in the docs generator. | `scripts/update_component_docs.py:41-44` counts only the literal `#[test]`, so 41 async tests are invisible. For example, `ptr-server` shows 1 test and has 4; `ptr-podwire` shows 24 and has 34 with its feature enabled. |
| 3.2 | Generate each README's "Upstream / Downstream / telemetry" lines from `cargo metadata` instead of hand-writing them, and add a gate for it. | Almost every crate README and diagram claims links that the dependency graph does not have. The most common is a telemetry edge to `ptr-observe`, and no crate depends on `ptr-observe`. |
| 3.3 | Replace the template header "Maturity: architecture + contract scaffold" with the generated maturity. | This header contradicts the generated block in the same file, for example `crates/ptr-types/README.md:4` against `:11`. |
| 3.4 | Update the root `README.md` component map and `docs/components/README.md` from 24 to 27 crates. | `ptr-cluster`, `ptr-execwire` and `ptr-podwire` are missing from both. |
| 3.5 | Refresh `STATUS.md`. | Its "19 architecture experiments remain planned" is now 20. Line 14 says raft-rs runs single-node, and line 39 says "multi-node consensus … incomplete", but a three-member raft group with elections, partitions and snapshot transfer is tested on loopback. Line 38 says "network authentication, scoped Pod access and durable audit/idempotency remain open", and all three exist. Line 11 calls the allow receipts "audit-ready", but every consumer discards them (`crates/ptr-runtime/src/lib.rs:501`, `execution.rs:1162`). |
| 3.6 | Mark the L001 numbers as historical v1-format evidence in the L001 README itself, then re-run L001 on PTRLOG02. | `results/run.json` and `results/process_crash_run.json` were recorded on 2026-09-18, before PTRLOG02 (`069d4b0`, 2026-09-19). Only `STATUS.md` says so. |
| 3.7 | Fix the stale single documents listed below the table. | Each was checked against current code. |

The stale single documents for 3.7:

- `docs/CONFIGURATION.md:5` says "only the file layer is implemented". The
  environment and CLI layers are also implemented and used.
- `model/burn-a0/config.toml` still says Burn 0.18, ndarray, MSRV 1.85 and
  validity embeddings. The code uses Burn 0.22.0-pre.3, flex, 1.95 and an
  admission mask.
- The `burn-0.18-ndarray-a0` evaluation candidate has a stale id.
- `sdk/` still says "contract-only", although the server implements both routes.
- `training/README.md:19` says ADR-0015, R003 and the training-backend evaluation
  "don't exist yet". All three exist.
- `docs/architecture/27-neural-state-admission.md:194-202` says "no real
  checkpoint". A real A0 checkpoint is bound in `checkpoint_binding.rs`.
- `OPEN_ITEMS_PLAN_20260921.md` still lists C3, C5, C6 and C8 as open.
  - C3 (Pod access over the network) and C5 (membership change) are implemented.
  - C8 is half closed: address authority exists (`PeerBook`), discovery does not.
  - C6 has an API (`take_installed_snapshot_matching`), but the host must supply
    the independent anchor.
- `docs/architecture/26-cognitive-codebook.md:333-344` lists 11 refusal codes.
  The code has 13.

### Step 4 — Close the concrete defects found

These are small and local, and each one can be verified on its own.

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 4.1 | The six `raft_fence.rs` tests never run in CI. They pass locally, 6 of 6. | The file is gated by `#![cfg(feature = "raft-rs-backend")]` (`crates/ptr-ledger/tests/raft_fence.rs:20`) and has no `[[test]]` entry. The only CI job with the feature runs `--test raft_rs` and `--test raft_cluster` (`.github/workflows/ci.yml:114-115`). | Add a `[[test]]` entry with `required-features` and a CI step. |
| 4.2 | A change to `ptr-types` does not trigger A0 CI, although A0 depends on it by path. | The `burn-a0.yml` path filter lists only `model/burn-a0/**`, `vendor/**` and itself. `model/burn-a0/Cargo.toml:12` depends on `../../crates/ptr-types`. | Add `crates/ptr-types/**` to both filters. |
| 4.3 | `datasets/private/` is not gitignored, although the docs promise it is never committed. | `git check-ignore datasets/private/x` matches nothing. The `.gitignore` covers only raw, processed and generated. | Add `/datasets/private/*` with a `.gitkeep` exception. |
| 4.4 | `Secret<T>` prints its plaintext through `Debug`. | `crates/ptr-inspect/src/lib.rs:19-20`: `#[derive(Debug)] pub struct Secret<T>(pub T)`. | Write a manual `Debug` that prints `[redacted]`, and make the field private. |
| 4.5 | The runtime silently drops gap, out-of-order and duplicate outcomes. | `MaterializedState::apply` does `let _ = self.try_apply(..)` (`crates/ptr-state/src/lib.rs:38-40`), and that is the entry point the runtime uses. | Call `try_apply` from the runtime and surface the outcome. |
| 4.6 | `FileLedger` has a **lifetime** cap of 100,000 commits, and compaction does not raise it. | `crates/ptr-ledger/src/integrity.rs:85` checks the absolute index against `MAX_RECORDS`. | Document it as a release blocker, or make the bound relative to the compaction floor. |
| 4.7 | Raft storage has crash windows. A crash mid-`apply_snapshot` can pair new snapshot bytes with old metadata, and a torn log tail leaves the member unopenable. | `crates/ptr-ledger/src/raft_storage.rs:362-375`, `:657-690`. | Reorder writes so state is persisted last and atomically, bind the snapshot to the state by digest, and add an anchored tail-recovery path like `FileLedger`'s. |
| 4.8 | `new_experiment.py` produces a manifest that fails the schema and is never registered. | `scripts/new_experiment.py` writes only `id`, `status` and `hypothesis`. `experiments/schema.toml` requires metrics, seeds, results_dir, baseline, falsification and hardware_profile. | Generate the full schema, register the experiment in `registry.toml`, and add a unit test. |
| 4.9 | The `training-backend` evaluation slot is missing from `evaluations/registry.toml`. | The registry has 26 components and there are 27 directories under `evaluations/components/`. | Add it, so a crate that references it does not fail with "unknown evaluation". |
| 4.10 | Fuzzing covers only an unused decoder, and no CI job runs it. | `fuzz/fuzz_targets/podwire.rs` targets the prost `PodCall` conversion that nothing speaks. | Add targets for the network-facing decoders (PTRPWREQ, PTREXREQ/PTREXRCP, PTRRAFTW) and PTRLOG02/PTRANC01, plus a CI build step. |
| 4.11 | Feature-gated ledger and state backends are tested but never clippy-linted or checked at MSRV. | `.github/workflows/ci.yml:97-121`. | Add clippy steps next to the test steps. |
| 4.12 | The local check lists are a subset of CI, so a contributor can pass every documented command and still fail CI. | `CONTRIBUTING.md:33-48`, the PR template and the `Makefile` all omit vendor, notices, codebook and research-gate checks. `make a0` uses the 1.85 toolchain for a crate that needs 1.95. | Add one `make ci-local` target that mirrors `ci.yml`, and pin `make a0` to `+1.95.0`. |
| 4.13 | The release archive ships no licence or notice files, and it does ship the `ptr-worker` stub. `gh release create … \|\| true` hides failures. | `.github/workflows/release.yml:22-30`, `bins/ptr-worker/src/main.rs:1-3`. | Package `LICENSE-*`, `NOTICE` and `THIRD-PARTY-NOTICES.md`, drop `ptr-worker` until it does something, and remove `\|\| true`. |
| 4.14 | A dormant workflow can still push to `main`. | `.github/workflows/one-shot-sync.yml` (`contents: write`, regenerates `Cargo.lock`, commits to main). `docs/VENDOR_PATCH_POLICY.md:105-107` says such workflows are removed. | Delete it, as its raft-engine predecessor was deleted in `ba1eb0e`. |
| 4.15 | Pod output is promoted on any `Pass`, even at level `Unverified` or with a hard finding. The action gateway refuses both. An empty `effects` list also passes the Pure/Read gate as if it were pure. | Pod loops check only `status == Pass` (`crates/ptr-runtime/src/lib.rs:332-342`, `:444-455`), and so does `ptr-podwire` (`access.rs:189`). The gateway checks level and hard findings (`execution.rs:1044-1053`). | Apply the gateway's `RequiredVerification` rule to Pod output, and require a non-empty effect declaration. |
| 4.16 | A mistyped journal path silently creates a new, empty journal. | `PtrRuntime::open_durable` → `FileLedger::open` uses `create_new` when the file is absent (`crates/ptr-ledger/src/file.rs:125-137`). `ptrctl seal` uses this unanchored open (`bins/ptrctl/src/seal.rs:105`). | Give `open_durable` an explicit "must exist" mode and use `open_durable_at` with an anchor in `ptrctl seal`. |
| 4.17 | One panicking action executor takes the execution host down for good. | No `catch_unwind` exists in `ptr-runtime`, `ptr-execwire` or `ptr-server`. `ptr-execwire` holds the runtime in a `std::sync::Mutex` and every later request calls `.lock().expect(..)`, so after one poisoning every request panics. | Contain executor panics at the gateway, and map a poisoned lock to a fenced, reportable state. |
| 4.18 | `ptr-bench` exits 0 even when `false_accepts`, `recovery_errors` or `tail_trim_errors` is non-zero, and the runner marks a run "completed" from the exit code alone. | `bins/ptr-bench/src/main.rs:176-186`, `:246-257` only print the counters. `scripts/run_experiment.py:229-235`. | Exit non-zero on any hard-invariant violation. |
| 4.19 | Compaction stops working once SemDB state exceeds the codec bounds. The whole state is exported as one `SemanticDelta`, capped at 16,384 items per list or 4 MiB. | `crates/ptr-runtime/src/compacted.rs:501-505`; `crates/ptr-semdb/src/codec.rs:5-6`. | Chunk the export, or give snapshots their own bounded format. Treat it as a release blocker together with 4.6. |
| 4.20 | The release job does not wait for CI, and actions are pinned by mutable tag. `dtolnay/rust-toolchain@stable` is a moving branch, and it runs in the release job, which holds `id-token: write`. | `.github/workflows/release.yml` has no `needs` or status gate. `grep uses: .github/workflows/*.yml`. | Gate the release on CI success, and pin actions by commit SHA at least in `release.yml`. |

### Step 5 — Reduce scope honestly

These are decisions, not code. They stop the documentation from promising what no
one is working on.

- **Eight crates have no consumer:** `ptr-router`, `ptr-ingress`, `ptr-memory`,
  `ptr-search`, `ptr-feedback`, `ptr-inspect`, `ptr-storage` and `ptr-observe`.
  A grep for `<crate>::` outside each crate finds nothing. Freeze them at their
  current size, and mark them "reserved: contract only, no consumer" in
  `component.toml`. Wire one in only when an experiment from Step 1 needs it.
  Recommendation: no new code in these crates until then.
- **ADR-0006 (typed isolate runtime) is Accepted but not implemented.** `ptr-runtime`
  lists `ptr-exec` in `Cargo.toml` but never imports it. The runtime is one
  synchronous `&mut self` object, and both network hosts wrap it in a global
  `Mutex` (`crates/ptr-server/src/lib.rs:35`,
  `crates/ptr-execwire/src/endpoint.rs:181`). That currently breaks
  `docs/INVARIANTS.md` rule 9 ("shared mutation is not the default"). Rule 8 is
  broken too, because the runtime event log is unbounded
  (`crates/ptr-runtime/src/lib.rs:781-787`). Either amend ADR-0006 to "deferred"
  and state the current model, or implement it. Recommendation: amend now, and
  implement only when a real model backend makes concurrency matter.
- **Configuration that does nothing.** 9 of 13 config keys are parsed and validated
  but read by nothing. For example, `runtime.mode = "cluster"` changes nothing, and
  `observability.*` has no subscriber. The other four behave in a narrower way than
  their names suggest:
  - The three `action_boundary.*` flags feed only the diagnostic `authorize_action`.
    The execution gateway forces all three checks on
    (`crates/ptr-runtime/src/execution.rs:1157-1159`).
  - So in `ptrd`, only `server.bind` changes behaviour.

  Remove or clearly mark inert keys so operators are not misled.
- **Integration and evaluation stubs.** 33 of 36 integration documents are
  identical 6-line stubs, and 74 of 81 evaluation candidates have no evidence.
  Do not add more until existing ones have measurements.

### Step 6 — Decisions only the owner can make

Each decision states my recommendation. None of them blocks Steps 1 to 4. They are
numbered O1–O5 so they do not collide with the D1–D6 decisions in issue #23.

| # | Decision | Recommendation |
|---|---|---|
| O1 | How should A0 grow after the M001–M004 run? It has no LM head, one layer and single-head attention. | If M001–M004 show an effect, test the mechanisms as an adapter on a small pretrained backbone rather than scaling A0 from scratch. If they do not, record the negative result before changing the design. |
| O2 | Which backbone for the plain-model baseline and for continued pretraining (ADR-0015)? | Pick one small open model, pin its revision once, and use it for both. It unblocks E002, the M00x gates and R003 together. |
| O3 | Continued pretraining for a coding Pod (ADR-0015, Proposed) | Pause it until Step 1 has a result. The corpus `own_code_corpus_v0_1` does not exist, so the dry-run manifest cannot even be built (`training/configs/run-coding-pod-cpt.toml:16`). The R003 trust gate is procedural only, because `check_research_gates.py` has no R003 rule. |
| O4 | Tier B items B1–B5 in the open-items plan, renumbered D1–D6 in issue #23 (runtime→net dependency, signed receipts, authenticated codebook, accept-loop location, admission-policy journaling, authenticated reconciliation) | Defer them all. They matter for a multi-host deployment, and no binary deploys the network crates yet. `IrohTransport` binds only `127.0.0.1` and gets a new key on every bind (`crates/ptr-net/src/lib.rs:49-52`). |
| O5 | The eight unconsumed crates (Step 5) | Freeze, as described in Step 5. |

## 4. What *not* to do now

- **Do not add more cluster or wire features.** That covers C2 to C8, accept loops
  and session reuse. First the model path must exist (Step 2) and the thesis must
  have evidence (Step 1).
- **Do not start continued-pretraining runs.** There is no corpus, no pinned base
  model, no held-out code-eval suite and no mechanical trust gate.
- **Do not add new evaluation slots, integration documents or architecture
  documents.** The ratio of prose to evidence is already the project's biggest risk.

## 5. Evidence base

Everything below was run in a clean checkout at `e93ed99` on 2026-09-24.

| Suite | Command | Result |
|---|---|---|
| Workspace, default features | `cargo test --workspace --locked` | 425 passed, 0 failed (31 packages) |
| ptr-observe tracing adapter | `cargo +stable test -p ptr-observe --features tracing-adapter` | 12 passed |
| Ledger failpoints | `… -p ptr-ledger --features failpoints --test failpoints` | 2 passed |
| Ledger raft-engine | `… --features raft-engine-backend --test raft_engine` | 1 passed |
| Ledger raft-rs | `… --features raft-rs-backend --test raft_rs` / `--test raft_cluster` | 13 + 21 passed |
| Ledger raft fence (**not in CI**) | `… --features raft-rs-backend --test raft_fence` | 6 passed |
| State Turso | `… -p ptr-state --features turso-backend --test turso` | 1 passed |
| Net Iroh | `… -p ptr-net --features iroh-backend` | 8 passed |
| Cluster wire | `… -p ptr-cluster --features cluster-backend` | 15 passed |
| Execution wire | `… -p ptr-execwire --features execwire-backend` | 28 passed |
| Pod wire | `… -p ptr-podwire --features podwire-backend` | 34 passed |
| Crate template | `… --manifest-path templates/rust-crate/Cargo.toml` | 19 passed |
| Burn A0 (Rust 1.95.0) | `cargo +1.95.0 test --manifest-path model/burn-a0/Cargo.toml` | 43 passed |
| Script unit tests | `python3 -m unittest discover -s scripts/tests` | 201 passed |
| Training unit tests | `python3 -m unittest discover -s training/tests` with `training/src` on the path | 32 passed |
| Repository gates | `check_repo`, `check_msrv_alignment`, `check_contract_citations`, `update_component_docs --check`, `run_experiment validate`, `run_component_eval validate` | all OK |

Per-package default-feature test counts: `ptr-runtime` 153, `ptr-types` 73,
`ptr-ledger` 62, `ptr-semdb` 24, `ptr-podwire` 24, `ptr-execwire` 14,
`ptr-observe` 11, `ptrctl` 8, `ptr-security` 8, `ptr-pods` 7, `ptr-config` 6,
`ptr-cluster` 6, `ptr-server` 4, `ptr-model-api` 3, `ptr-protocol` 3; every other
package 0 to 2.
