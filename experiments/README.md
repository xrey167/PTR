# PTR Experiments

Every experiment has an ID, hypothesis, baseline, seeds, metrics, falsification criterion and immutable result directory.

Categories:
- `model/`
- `semdb/`
- `runtime/`
- `lifecycle/`
- `retrieval/`
- `feedback/`
- `system/`

Do not silently rewrite failed results. Supersede them with a new experiment/version and record the decision in `research/decisions/`.


## Runner

Validate manifests:

```bash
python scripts/run_experiment.py validate
```

Execute a declared experiment entrypoint with an explicitly listed seed:

```bash
python scripts/run_experiment.py run L001 --seed 17 --set iterations=100
python scripts/run_experiment.py run L001 --entrypoint process_entrypoint --seed 17 --set iterations=50
```

The runner never invokes a shell. It tokenizes the versioned `entrypoint`, rejects undeclared seeds and unresolved placeholders, and writes a unique immutable JSON record containing the Git revision, manifest/lock hashes, exact argv, duration, exit status, stdout/stderr and launch errors. The record appears whole or not at all, so a write that fails leaves no partial record. Failed executions remain evidence rather than being overwritten.

The Git revision is the commit HEAD was at when the run started. The runner refuses to start from a working tree whose provenance files (the Rust, SQL and protobuf sources, the Cargo and toolchain files, the recording scripts, the experiment's own files apart from its results) HEAD does not hold, counting files only the clone's own ignore rules hide, files that HEAD's `.gitignore` files would show but one HEAD does not hold hides (named by that `.gitignore`), files git is told not to look at, symlinks named as sources, and symlinked directories, nested repositories and submodules where the build reads, such as a `crates/ptr-*` crate linked or cloned in, and writes no record, exiting 2, when HEAD moved or one of those files was written, created or removed while the command ran, even if it was put back (`ProvenanceWatch` in `scripts/experiment_records.py`).

Every experiment's `aggregate.py` (L001, L003 and L004 today) publishes `results/run.json` and `results/metrics.json` as one aggregate: `run.json` names the SHA-256 of the `metrics.json` written with it, and an interrupted publish leaves no `run.json` rather than the previous one beside new metrics. The L003 and L004 aggregators refuse to aggregate in a checkout holding provenance files HEAD does not hold, and `scripts/check_research_gates.py` fails a completed experiment whose `run.json` is not one aggregate with the `metrics.json` and `mutations.json` beside it (`publish_aggregate` and `aggregate_errors` in `scripts/experiment_records.py`). A current `run.json` must name that SHA-256. One aggregated before `run.json` named it, as the archived L003 and L004 results of ad2f8d1 were, passes only while it is stale and while it and the `metrics.json` beside it are the pair the last commit that changed `run.json` holds.
