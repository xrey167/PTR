.PHONY: check test fmt a0 repo-check docs docs-check meta-check experiments-check evals-check msrv python-test manifest tree \
	ci-local ci-quality ci-rust-stable ci-rust-msrv ci-python-training ci-repository-invariants \
	ci-lifecycle-failpoints ci-ledger-raft-engine ci-ledger-raft-rs ci-state-turso \
	ci-network-iroh ci-cluster-wire ci-execution-wire ci-pod-wire

# `model/burn-a0` is its own workspace, excluded from the root one, so every
# `--workspace` target below reaches none of it. That is why `fmt` names it
# explicitly and why `a0` exists: a contributor running the cheap checks should not
# be able to get a clean result on half the code they changed and a red CI on the
# other half. `make a0` is the `burn-a0` workflow's job, step for step.
#
# `a0` names its toolchain because the root `rust-toolchain.toml` pins 1.85.0 for
# every bare `cargo` run from here, and `model/burn-a0` requires 1.95: a bare
# `make a0` could not build the crate it exists to check.

PYTHON ?= python3
A0_TOOLCHAIN ?= 1.95.0
# The metadata gate compares against where this branch left main, as CI's
# pull-request run compares against the pull request's base.
BASE ?= $(shell git merge-base HEAD origin/main)

check:
	cargo check --workspace --all-targets --locked

fmt:
	cargo fmt --all -- --check
	cargo fmt --manifest-path model/burn-a0/Cargo.toml -- --check

a0:
	cargo +$(A0_TOOLCHAIN) fmt --manifest-path model/burn-a0/Cargo.toml -- --check
	cargo +$(A0_TOOLCHAIN) test --manifest-path model/burn-a0/Cargo.toml --locked
	cargo +$(A0_TOOLCHAIN) check --manifest-path model/burn-a0/Cargo.toml --examples --locked
	cargo +$(A0_TOOLCHAIN) clippy --manifest-path model/burn-a0/Cargo.toml --all-targets --locked -- -D warnings

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

# `ci-local` is `.github/workflows/ci.yml`, one target per job and step for step,
# so a contributor can run all of CI or only the job that failed. The documented
# check lists were a subset of CI (no vendor, notices, codebook or research-gate
# checks, no feature backends), so passing every one of them still left CI red.
# scripts/tests/test_ci_local.py fails when a CI step has no counterpart here.
# Not reproduced: the Windows leg of rust-stable and the Python 3.11 leg of
# python-training (matrix legs of the same steps), `pip install -e training`
# (PYTHONPATH is set instead, so nothing is installed into your environment),
# and security.yml, which needs cargo-audit and cargo-deny. `make a0` is the
# burn-a0 workflow, which runs only when A0 or what it depends on changes.
ci-local: ci-quality ci-rust-stable ci-rust-msrv ci-python-training ci-repository-invariants \
	ci-lifecycle-failpoints ci-ledger-raft-engine ci-ledger-raft-rs ci-state-turso \
	ci-network-iroh ci-cluster-wire ci-execution-wire ci-pod-wire

ci-quality:
	cargo +stable fmt --all -- --check
	cargo +stable clippy --workspace --all-targets --locked -- -D warnings
	RUSTDOCFLAGS="-D warnings" cargo +stable doc --workspace --no-deps --locked
	cargo +stable test --manifest-path templates/rust-crate/Cargo.toml --locked
	cargo +stable clippy --manifest-path templates/rust-crate/Cargo.toml --all-targets --locked -- -D warnings
	cargo +stable test -p ptr-observe --features tracing-adapter --locked
	cargo +stable clippy -p ptr-observe --features tracing-adapter --all-targets --locked -- -D warnings

ci-rust-stable:
	cargo +stable check --workspace --all-targets --locked
	cargo +stable test --workspace --locked

ci-rust-msrv:
	rustup toolchain install 1.85.0 --profile minimal
	cargo +1.85.0 check --workspace --all-targets --locked
	cargo +1.85.0 test --workspace --locked
	cargo +1.85.0 check -p ptr-observe --features tracing-adapter --locked
	cargo +1.85.0 test --manifest-path templates/rust-crate/Cargo.toml --locked

ci-python-training:
	PYTHONPATH=training/src $(PYTHON) -m unittest discover -s training/tests
	PYTHONPATH=training/src $(PYTHON) training/src/ptr_training/validate_dataset.py datasets/samples/epistemic_calibration.jsonl
	PYTHONPATH=training/src $(PYTHON) training/src/ptr_training/validate_dataset.py datasets/samples/ood_pod.jsonl
	PYTHONPATH=training/src $(PYTHON) training/src/ptr_training/validate_dataset.py datasets/samples/operator_route.jsonl

ci-repository-invariants:
	$(PYTHON) scripts/check_component_metadata.py --base $(BASE)
	$(PYTHON) scripts/check_vendor_integrity.py
	$(PYTHON) scripts/check_vendor_retirement.py --require-observations
	$(PYTHON) scripts/check_notices.py
	$(PYTHON) scripts/check_codebook.py
	$(PYTHON) scripts/update_component_docs.py --check
	$(PYTHON) scripts/check_repo.py
	$(PYTHON) scripts/check_msrv_alignment.py
	$(PYTHON) scripts/check_contract_citations.py
	$(PYTHON) scripts/check_architecture_catalog.py
	$(PYTHON) scripts/check_rust_conventions.py
	$(PYTHON) scripts/run_experiment.py validate
	$(PYTHON) scripts/run_component_eval.py validate
	$(PYTHON) scripts/check_research_gates.py
	$(PYTHON) -m unittest discover -s scripts/tests
	$(PYTHON) -m unittest discover -s research/baselines/rag_reference/tests
	$(PYTHON) -m unittest discover -s research/baselines/strong_rag/tests
	$(PYTHON) -m unittest discover -s docs-site/tests

ci-lifecycle-failpoints:
	cargo +stable test -p ptr-ledger --features failpoints --test failpoints --locked
	cargo +stable clippy -p ptr-ledger --features failpoints --all-targets --locked -- -D warnings

ci-ledger-raft-engine:
	cargo +stable test -p ptr-ledger --features raft-engine-backend --test raft_engine --locked
	cargo +stable clippy -p ptr-ledger --features raft-engine-backend --all-targets --locked -- -D warnings

ci-ledger-raft-rs:
	cargo +stable test -p ptr-ledger --features raft-rs-backend --test raft_rs --locked
	cargo +stable test -p ptr-ledger --features raft-rs-backend --test raft_cluster --locked
	cargo +stable test -p ptr-ledger --features raft-rs-backend --test raft_fence --locked
	cargo +stable clippy -p ptr-ledger --features raft-rs-backend --all-targets --locked -- -D warnings

ci-state-turso:
	cargo +stable test -p ptr-state --features turso-backend --test turso --locked
	cargo +stable clippy -p ptr-state --features turso-backend --all-targets --locked -- -D warnings

ci-network-iroh:
	cargo +stable test -p ptr-net --features iroh-backend --locked
	cargo +stable clippy -p ptr-net --features iroh-backend --all-targets --locked -- -D warnings
	rustup toolchain install 1.91.0 --profile minimal
	cargo +1.91.0 test -p ptr-net --features iroh-backend --locked

ci-cluster-wire:
	cargo +stable test -p ptr-cluster --features cluster-backend --locked
	cargo +stable clippy -p ptr-cluster --features cluster-backend --all-targets --locked -- -D warnings
	rustup toolchain install 1.91.0 --profile minimal
	cargo +1.91.0 test -p ptr-cluster --features cluster-backend --locked

ci-execution-wire:
	cargo +stable test -p ptr-execwire --features execwire-backend --locked
	cargo +stable clippy -p ptr-execwire --features execwire-backend --all-targets --locked -- -D warnings
	rustup toolchain install 1.91.0 --profile minimal
	cargo +1.91.0 test -p ptr-execwire --features execwire-backend --locked

ci-pod-wire:
	cargo +stable test -p ptr-podwire --features podwire-backend --locked
	cargo +stable clippy -p ptr-podwire --features podwire-backend --all-targets --locked -- -D warnings
	rustup toolchain install 1.91.0 --profile minimal
	cargo +1.91.0 test -p ptr-podwire --features podwire-backend --locked
