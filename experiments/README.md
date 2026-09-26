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

The runner never invokes a shell. It tokenizes the versioned `entrypoint`, rejects undeclared seeds and unresolved placeholders, and writes a unique immutable JSON record containing the Git revision, manifest/lock hashes, exact argv, duration, exit status, stdout/stderr and launch errors. Failed executions remain evidence rather than being overwritten.

The Git revision is the commit HEAD was at when the run started. The runner refuses to start from a working tree whose provenance files (the Rust and SQL sources, the Cargo and toolchain files, the recording scripts, the experiment's own files apart from its results) HEAD does not hold, and writes no record, exiting 2, when HEAD moved or one of those files was written, created or removed while the command ran, even if it was put back (`ProvenanceWatch` in `scripts/experiment_records.py`).
