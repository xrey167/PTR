.PHONY: check test fmt repo-check docs docs-check meta-check experiments-check evals-check msrv python-test manifest tree

check:
	cargo check --workspace --all-targets --locked

fmt:
	cargo fmt --all -- --check

test:
	cargo test --workspace --locked

repo-check:
	python3 scripts/check_repo.py
	python3 scripts/check_msrv_alignment.py

docs:
	python3 scripts/update_component_docs.py --write

docs-check:
	python3 scripts/update_component_docs.py --check

meta-check:
	python3 scripts/check_component_metadata.py --base HEAD^

experiments-check:
	python3 scripts/run_experiment.py validate

evals-check:
	python3 scripts/run_component_eval.py validate

msrv:
	cargo +1.85.0 check --workspace --all-targets --locked
	cargo +1.85.0 test --workspace --locked

python-test:
	python3 -m unittest discover -s training/tests
	python3 -m unittest discover -s scripts/tests

manifest:
	python3 scripts/hash_manifest.py

tree:
	python3 scripts/print_tree.py
