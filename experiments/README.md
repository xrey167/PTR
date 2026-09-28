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

- its `config.toml` holds a `[preregistration]` table with every key the list requires, each a pinned value of its declared type (`int`, `str`, `bool`, a non-empty `int-list` or `str-list`, or `file`), and no other key of the table holds a placeholder either;
- every value of that table is a boolean, an integer of magnitude at most 2^53 − 1, a string of printable ASCII or a list of only such integers or only such strings, and every key is printable ASCII, so that its canonical text has one spelling whoever writes it; `experiment.toml`'s `preregistration_sha256` is the SHA-256 of that text;
- a `file` key names a file in the repository, and the table's `<key>_sha256` is the SHA-256 of that file (with CRLF read as LF), so the file's content is frozen with the table, not only its path;
- the table's `seeds`, which every entry of the list must require as an `int-list`, names the manifest's seeds, so no seed can be added after an outcome is seen;
- every baseline the list names is pinned at each key the list names for it, its status is pinned and not `blocked-*`, and the table's `baseline_<name>_sha256` is the digest of every file in the baseline's directory: the canonical text of the table mapping each file's path within the directory to its SHA-256 (with CRLF read as LF), over the files git tracks or would track there, so every setting of the baseline, not only the keys that must be pinned, and the implementation beside its configuration are frozen with the table;
- `experiment.toml`'s `preregistration_rules_sha256` is the SHA-256 of the canonical text of the experiment's entry in `preregistration.toml`, so its types and baselines are frozen with the table;
- every run record (`run-*.json`, whose manifest `run_experiment.py` records) and aggregate (`run.json`) the experiment has committed, in any directory the registry has given the experiment and whatever its `results_dir` is now, and those in its results directory, name those two digests and the commit they ran at (`git_sha`), and that commit is on HEAD's history and holds the same `[preregistration]` table, manifest digests, list entry, files and baselines; a run record also names the SHA-256 of `experiment.toml` at that commit (`manifest_sha256`) and has held the same content in every commit on every side of every merge since it was committed, an aggregate, which an aggregator may write again, holds to this in every version it was committed in, none once committed is deleted, renamed or moved, and each is a regular file reached through no symlink. A preregistration rewritten after its runs thus fails even when the manifest and the records are rewritten to match, short of rewriting history. An aggregator of a listed experiment writes `preregistration_sha256`, `preregistration_rules_sha256` and `git_sha` into `run.json`;
- the manifest and `config.toml` at every such commit are the ones now, but for the manifest's status, which has only moved forward since: from `prepared` to `running` to `completed` or `failed`, which are final. So neither the falsification criterion nor the results directory nor any setting outside the table changes after a run, and the results directory stays inside the experiment's directory;
- no file the table freezes and no baseline's directory lies in the results directory, or holds it: the runner holds the experiment's files to HEAD except its results directory, where runs write;
- every commit that held the experiment frozen as the runner launches it (listed, past `planned`, its preregistration complete and named by the manifest's digests, its files and baselines as the table freezes them) holds the same as a run's commit must, since a run could be made there and its record discarded before it was committed: the first committed freeze binds. A commit that held less, such as one still holding a placeholder, could not launch it and freezes nothing.

The list keeps every experiment it has named at a commit on HEAD's history that the registry holds now or has held at any commit, whether registered before or after it was listed; one taken out of it fails the gate and is refused by the runner, so neither its runs nor its preregistration leave the gate. A listed experiment that was committed frozen, or whose run records were committed, cannot go back to `planned`; it can only be superseded.

`scripts/run_experiment.py` refuses to run or prepare a listed experiment until all of this holds for it (`launch_errors`), so none of its outcomes is seen before its preregistration is frozen, and holds the files that decision reads (the list, the registry, the gate, the directories of the entry's baselines and the files its table names, `launch_inputs`) to HEAD before and while the run, as it holds the experiment's own files; a pilot run outside the runner is not confirmatory evidence. A file or baseline the list names, and every file of a baseline's directory, must be a regular file inside the repository, reached through no symlink: git holds a link as its target's path, so the commit a run names could not show the content frozen through it.

The canonical text (`canonical_text` in `scripts/experiment_records.py`) is `true` or `false`; an integer in decimal; a string between double quotes in which only `\` and `"` are escaped, as `\\` and `\"`, and `/` is not; a list as `[` and its elements joined by `,` and `]`; a table as `{` and its `"key":value` entries in ascending byte order of the keys joined by `,` and `}`; with no whitespace. On this domain it is the JSON Canonicalization Scheme of RFC 8785.

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
