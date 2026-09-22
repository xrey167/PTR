.PHONY: check test fmt a0 repo-check docs docs-check meta-check experiments-check evals-check msrv python-test manifest tree

# `model/burn-a0` is its own workspace, excluded from the root one, so every
# `--workspace` target below reaches none of it. That is why `fmt` names it
# explicitly and why `a0` exists: a contributor running the cheap checks should not
# be able to get a clean result on half the code they changed and a red CI on the
# other half. `make a0` is the `burn-a0` workflow's job, step for step.

check:
	cargo check --workspace --all-targets --locked

fmt:
	cargo fmt --all -- --check
	cargo fmt --manifest-path model/burn-a0/Cargo.toml -- --check

a0:
	cargo fmt --manifest-path model/burn-a0/Cargo.toml -- --check
	cargo test --manifest-path model/burn-a0/Cargo.toml --locked
	cargo check --manifest-path model/burn-a0/Cargo.toml --examples --locked
	cargo clippy --manifest-path model/burn-a0/Cargo.toml --all-targets --locked -- -D warnings

test:
	cargo test --workspace --locked

repo-check:
	python3 scripts/check_repo.py
	python3 scripts/check_msrv_alignment.py
	python3 scripts/check_contract_citations.py

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
