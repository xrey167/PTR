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

The runner never invokes a shell. It tokenizes the versioned `entrypoint`, rejects undeclared seeds and unresolved placeholders, and writes a unique immutable JSON record containing the Git revision, manifest/lock hashes, exact argv, duration, exit status, stdout/stderr and launch errors. Failed executions remain evidence rather than being overwritten. Since record version 2 it also holds whether tracked files differed from that revision (`git_dirty`, with a hash of the diff), the measured host (CPU model, logical CPUs, memory), the declared hardware profile's hash and contents with the fields it still leaves `unspecified`, and the `rustc` version of the toolchain a cargo entrypoint runs on.

Summarize one entrypoint's records across seeds:

```bash
python scripts/run_experiment.py aggregate L001
python scripts/run_experiment.py aggregate M001 --entrypoint ablation_entrypoint --git-sha <sha>
```

Every JSON object a run printed on its own line is a row: its string fields name it (for example `{"arm": "typed", "split": "ood"}`) and its numeric fields are summarized across seeds (n, mean, sample standard deviation, min, max, and each seed's value). The aggregate is written as a new immutable `aggregate-<timestamp>-<entrypoint>.json` that lists the SHA-256 of every record it read. It refuses, writing nothing, records from more than one commit (choose one with `--git-sha`), records from a dirty or unrecorded worktree (unless `--allow-dirty`, which it records), records that differ in manifest, parameters, toolchain or host, a seed the manifest does not declare, two completed runs of one seed, a row printed twice by one run, and a malformed record. Failed runs, declared seeds with no completed run, rows or metrics some seeds did not report, and non-finite values (NaN, infinity) are listed, never dropped or averaged; any of them makes the aggregate `incomplete` and the command exit 1.
