# Component protocol format

`DECISION_PROCESS.md` says a component moves from `open` to `preferred` only after
gate tests pass and evidence compares at least two credible candidates, and that a
benchmark must be able to show a candidate worse. A **protocol** is what makes that
checkable: the component's comparison, written down and committed **before the first
evidence run**, so that nothing about the gates, workload, metrics, repetitions or
scoring can be chosen after the numbers are known.

One file per component: `evaluations/components/<component>/protocol.toml`
(template: `evaluations/protocol.template.toml`). `scripts/check_component_protocols.py`
(CI: `repository-invariants`, `make evals-protocols-check`) enforces the rules below.
Unit tests: `scripts/tests/test_check_component_protocols.py`.

## What a protocol fixes

| Table | Content |
|---|---|
| top level | `version = 1`, `component` (the directory name), `protocol_id` (changes whenever the protocol changes), `hardware_profile` (a `hardware/*.toml`), `bench_workspace` (the excluded workspace under `evaluations/bench/`), `lockfile` (its `Cargo.lock`, which pins the candidate crate versions) |
| `[repetitions]` | `warmup >= 1` discarded runs and `measured >= 5` kept runs; the reported value of a metric is the **median over the measured runs** |
| `[workload]` | `description`, `seeds` (generated data is a pure function of a seed; the kit's SplitMix64 is copied from `ptr-bench` so a seed means the same on every machine), `samples` (>= 10000 whenever a latency metric exists, so p99 is an observed order statistic of at least 100 samples) |
| `[[gate]]` | `id`, `criterion` (`correctness` or `failure_semantics`), `description`. Gates are pass/fail correctness and safety properties every candidate must satisfy, such as digest-identical outputs, hostile input gives an error and never a panic, bounded queues, revoked data never returned |
| `[[metric]]` | `id`, `criterion` (`latency`, `throughput`, `memory`, `operability`), `unit`, `direction` (`lower`/`higher`). A metric exists so that a candidate can be shown worse |
| `[scoring]` | `weights_from = "config.toml"`, `excluded = ["research_flexibility"]`, `method = "ratio-to-best"`, and the three proxies below |
| `[[candidate]]` | `id` (equal to the id in `candidates.toml`), `kind`, `version`, and the pin: for `crate`, `lock_package` (the lockfile must contain exactly `version`); for `binary`, `distro-package`, `python-package`, a 64-hex `sha256` (and an `https` `url` for a binary); `incumbent` is the in-repo implementation |

The checker verifies the schema, that each `crate` pin is in the lockfile, that every
protocol candidate is in `candidates.toml`, and that the hardware profile exists.

## Evidence

A benchmark writes one `evidence/<started_at>-<candidate>-run.json` per candidate and
run. The record has **exactly the fields `scripts/run_component_eval.py` stores** for a
run (`schema_version`, `component`, `candidate`, `criteria`, `git_sha`, `platform`,
`python`, `candidate_manifest_sha256`, `cargo_lock_sha256`, `status`, `started_at`,
`command`, `finished_at`, `exit_code`, `stdout`, `stderr`, `launch_error`,
`duration_ns`) plus two extension fields:

* `protocol_sha256`: SHA-256 of `protocol.toml` at the time of the run. A record that
  names any other hash is stale evidence and is rejected.
* `bench`: `candidate_version`, `gates` (`{gate id: bool}`, exactly the protocol's gate
  ids), `benchmark_ran`, `repetitions` (`warmup`, `measured`), `decision_grade`,
  `decision_note`, `metrics` (`{metric id: median}`), `proxies`, `environment` (git sha
  and dirty flag, rustc, CPU model, threads, memory, kernel, filesystem type, CPU
  governor).

The Rust kit (`evaluations/bench/kit`) writes these records
(`evalkit::evidence::EvidenceRecord`); a unit test pins its field list to the Python
runner's and to this checker's. The legacy runner records and smoke files that predate
protocols have no `bench` object and are not benchmark evidence.

`decision_grade` is true only if the filesystem is not tmpfs/overlay-like **and** the
hardware profile says `decision_grade = true`. The checker rejects a record that claims
decision grade on a profile that does not. `hardware/cpu-4core-15gib.toml` is a shared,
heavily loaded sandbox and says `false`: durable-write numbers taken there are "not
decision-grade" and cannot make a candidate `preferred`.

## Order: protocol first

The protocol is committed before the first evidence. The checker compares git history:
the commit that first added `protocol.toml` must be a strict ancestor of the commit that
first added any evidence naming it (same commit is refused; untracked evidence is not
yet history). A change to the protocol after evidence exists changes its hash and
invalidates that evidence; the remedy is a new `protocol_id` and a re-run.

## Status rules

Per component, using the latest benchmark record of each candidate:

1. **A failed gate means `rejected`.** Any `false` in `bench.gates` with a status other
   than `rejected` is an error. A `rejected` status without gate-failure evidence (for
   example a candidate blocked by a security advisory, not obtainable, or dropped on
   license) must carry a `note` naming the reason.
2. **`preferred` needs at least two candidates that passed every gate, ran the
   benchmark with the protocol's repetitions, and are decision-grade**, and the
   preferred candidate must be one of them. `preferred` without a protocol is an error.
3. **A lone survivor stays `evaluating`** and carries a `note` that says so.
4. Evidence that is not a benchmark record (existing test files, smoke JSON) is
   ignored; a component without benchmark evidence needs no protocol yet.

`locked` remains reserved as `DECISION_PROCESS.md` says; this tooling never sets it.

## Scoring rule

`criteria` weights come from the component's `config.toml`/`candidates.toml`
`[criteria]` (correctness .3, failure_semantics .15, latency .12, throughput .1,
memory .08, portability .07, operability .06, integration_cost .05, license_fit .04,
research_flexibility .03). Only **survivors** (gates passed, benchmark ran) are scored,
by `check_component_protocols.automated_score`:

* `correctness`, `failure_semantics`: gate criteria. A survivor scores 1.0 on each. They
  decide survival; they do not rank.
* `latency`, `throughput`, `memory`, `operability`: **ratio-to-best** per metric among
  survivors (`best / value` when lower is better, `value / best` when higher is better;
  1.0 is the best candidate), averaged within a criterion.
* `license_fit`: the **cargo-deny verdict** of the candidate's graph in the bench
  workspace under the repository `deny.toml`: 1.0 clean, 0.0 not. A candidate with a
  security advisory in its graph is *blocked* unless a cheap vendor patch exists (see
  `docs/VENDOR_PATCH_POLICY.md`); there are no advisory ignores.
* `integration_cost`: mean ratio-to-best (lower is better) of three measurable parts:
  adapter lines of code (`tokei`-style count of the candidate's adapter module),
  transitive crate count (`cargo tree --prefix none | sort -u | wc -l` of the candidate's
  feature set), and clean build seconds (median of the protocol's repetitions).
* `portability`: the fraction of the protocol's target triples on which `cargo check`
  of the candidate's adapter succeeds.
* `research_flexibility`: **owner-scored and excluded** from the automated score. The
  weight is not redistributed to it under any circumstance; changing it changes nothing.

The total is `sum(w_c * s_c) / sum(w_c)` over the criteria that have a basis for that
component; criteria without one are listed as `unscored` instead of being counted as
zero. The score ranks survivors; it is not itself a decision. A `preferred` verdict is a
human decision recorded in `candidates.toml` after the checker agrees it is allowed.

## Proxies are measurable on purpose

`license_fit`, `integration_cost` and `portability` are scored from things a script can
measure so that the score is reproducible without judgement. They are proxies and say
so: a low adapter LOC does not mean an API is pleasant. `research_flexibility`, the one
criterion that is judgement, stays with the owner.

## Workflow

1. Run the spike (`evaluations/SPIKE.md`): is the candidate clean, blocked or open?
2. Write `protocol.toml`, set `status` of blocked/non-obtainable candidates to
   `rejected` with a `note`, commit.
3. Run gates and benchmark with the kit; commit the evidence records.
4. `python3 scripts/check_component_protocols.py` must pass; then the human decision.
