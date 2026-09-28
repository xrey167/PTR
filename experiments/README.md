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

## Preregistration

An experiment listed in `preregistration.toml` leaves `planned` only with a frozen preregistration. Once its status is `prepared`, `running`, `completed` or `failed`, `scripts/check_research_gates.py` fails CI unless:

- its `config.toml` holds a `[preregistration]` table with every key the list requires, each a pinned value of its declared type (`int`, `str`, `bool`, or a non-empty `int-list` or `str-list`), and no other key of the table holds a placeholder either;
- every value of that table is an integer, a boolean, a string of printable ASCII or a list of only integers or only such strings, and every key is printable ASCII, so that its canonical text (JSON with sorted keys and no whitespace, `preregistration_canonical` in `scripts/experiment_records.py`) escapes only `"` and `\` and is the same whoever writes it; `experiment.toml`'s `preregistration_sha256` is the SHA-256 of that text;
- a `seeds` key of the table, where there is one, names the manifest's seeds;
- every baseline the list names is pinned at each key the list names for it, and its status is pinned and not `blocked-*`.

A `must-be-pinned-…` string marks a value still to be chosen and a `must-be-signed-…` string an owner decision still to be taken, which the owner signs by committing the decided value in its place. Either blocks whatever the key's type, and whether or not the list requires the key, as do an empty string, `unconfigured` and `none`, in any case and with any surrounding space, so an owner decision cannot pass as a default. A superseded experiment is not gated: another experiment replaced it, and that one's preregistration counts. S003, F003, Q003, R004, M008 and E005 are listed, and all of them are still `planned`.


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
