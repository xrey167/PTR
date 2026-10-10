.PHONY: check test fmt bench-check py-bindings a0 a0-stable a0-msrv repo-check docs docs-check meta-check experiments-check evals-check msrv python-test manifest tree \
	desktop-setup gpu-doctor smoke-gpu olmoe-sft-smoke release-package \
	ci-local ci-quality ci-rust-stable ci-rust-msrv ci-python-training ci-repository-invariants \
	ci-lifecycle-failpoints ci-ledger-raft-engine ci-ledger-raft-rs ci-state-turso ci-state-postgres \
	ci-network-iroh ci-cluster-wire ci-execution-wire ci-pod-wire ci-storage-tiers ci-feature-gates

# `model/burn-a0` is its own workspace, excluded from the root one, so every
# `--workspace` target below reaches none of it. That is why `fmt` names it
# explicitly and why `a0` exists: a contributor running the cheap checks should not
# be able to get a clean result on half the code they changed and a red CI on the
# other half. `make a0` is the `burn-a0` workflow's job, step for step.
#
# `a0` is the workflow's two jobs, `a0-stable` and `a0-msrv`, and every step names
# its toolchain: the root `rust-toolchain.toml` pins 1.99.0 for every bare `cargo`
# run from here, and `model/burn-a0` requires 1.95, so a bare `cargo` could not
# build the crate at all. `a0-stable` lints on your `stable`, as CI lints on the
# current one: run `rustup update stable` first, or a lint newer than your
# toolchain passes here and fails there.

PYTHON ?= python3
A0_STABLE ?= stable
A0_MSRV ?= 1.95.0
# The metadata gate compares against where this branch left main, as CI's
# pull-request run compares against the pull request's base.
BASE ?= $(shell git merge-base HEAD origin/main 2>/dev/null)

check:
	cargo check --workspace --all-targets --locked

fmt:
	cargo fmt --all -- --check
	cargo fmt --manifest-path model/burn-a0/Cargo.toml -- --check
	cargo fmt --manifest-path evaluations/bench/kit/Cargo.toml -- --check

# `bindings/ptr-py` is a PyO3 extension in its own excluded workspace, built with
# maturin into a venv kept inside it (gitignored). Not a CI job: nothing in
# ci.yml builds it, so no ci-* counterpart exists.
MATURIN ?= 1.15.0
PY_BINDINGS := bindings/ptr-py

py-bindings:
	$(PYTHON) -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "py-bindings needs Python >= 3.11 (set PYTHON=...)"; exit 1; }
	$(PYTHON) -m venv $(PY_BINDINGS)/.venv
	$(PY_BINDINGS)/.venv/bin/pip install -q maturin==$(MATURIN)
	cargo fmt --manifest-path $(PY_BINDINGS)/Cargo.toml -- --check
	cargo clippy --manifest-path $(PY_BINDINGS)/Cargo.toml --all-targets --locked -- -D warnings
	cd $(PY_BINDINGS) && VIRTUAL_ENV="$(CURDIR)/$(PY_BINDINGS)/.venv" PATH="$(CURDIR)/$(PY_BINDINGS)/.venv/bin:$$PATH" maturin develop --locked
	$(PY_BINDINGS)/.venv/bin/python -m unittest discover -s $(PY_BINDINGS)/tests -v

# Release package without GitHub: the same files .github/workflows/release.yml uploads
# (binary tarball, CycloneDX SBOM, cargo metadata, SHA-256 files), written to dist/.
# The GitLab `release` job runs this on tags. RELEASE_TOOLCHAIN defaults to stable like
# release.yml; pass RELEASE_TOOLCHAIN=1.99.0 where the local stable is older than the MSRV.
RELEASE_TOOLCHAIN ?= stable
RELEASE_BINS := ptrd ptrctl ptr-worker ptr-bench
RELEASE_TARGET_DIR ?= $(or $(CARGO_TARGET_DIR),target)

release-package:
	cargo +$(RELEASE_TOOLCHAIN) build --release --locked
	rm -rf dist && mkdir -p dist/bin
	cp $(addprefix $(RELEASE_TARGET_DIR)/release/,$(RELEASE_BINS)) dist/bin/
	tar -C dist/bin -czf dist/ptr-linux-x86_64.tar.gz $(RELEASE_BINS)
	cd dist && sha256sum ptr-linux-x86_64.tar.gz > ptr-linux-x86_64.tar.gz.sha256
	cargo +$(RELEASE_TOOLCHAIN) metadata --locked --format-version 1 > dist/cargo-metadata.json
	$(PYTHON) scripts/generate_sbom.py dist/cargo-metadata.json --out dist/ptr-cyclonedx.json
	cd dist && sha256sum ptr-cyclonedx.json > ptr-cyclonedx.json.sha256
	rm -rf dist/bin

# GPU desktop (Ubuntu 24.04, one RTX 4090): docs/DESKTOP_SETUP.md. Not CI jobs, so no
# ci-* counterpart. The two smoke targets run the training venv's python (the CUDA torch
# wheel and ptr_py live there); `--execute` refuses a dirty tracked worktree. Both configs
# are pipeline evidence only, never research results.
TRAIN_PY ?= training/.venv/bin/python
SMOKE_GPU_CONFIG ?= training/configs/stage-moe-smoke-gpu.toml
OLMOE_SFT_CONFIG ?= training/configs/stage-moe-olmoe-sft-smoke.toml

desktop-setup:
	scripts/setup_desktop.sh

gpu-doctor:
	python3 scripts/gpu_doctor.py

smoke-gpu:
	PYTHONPATH=training/src $(TRAIN_PY) -m ptr_training.run --config $(SMOKE_GPU_CONFIG) --execute

olmoe-sft-smoke:
	PYTHONPATH=training/src $(TRAIN_PY) -m ptr_training.run --config $(OLMOE_SFT_CONFIG) --execute

a0: a0-stable a0-msrv

a0-stable:
	cargo +$(A0_STABLE) fmt --manifest-path model/burn-a0/Cargo.toml -- --check
	cargo +$(A0_STABLE) test --manifest-path model/burn-a0/Cargo.toml --locked
	cargo +$(A0_STABLE) check --manifest-path model/burn-a0/Cargo.toml --examples --locked
	cargo +$(A0_STABLE) clippy --manifest-path model/burn-a0/Cargo.toml --all-targets --locked -- -D warnings

a0-msrv:
	rustup toolchain install $(A0_MSRV) --profile minimal
	cargo +$(A0_MSRV) test --manifest-path model/burn-a0/Cargo.toml --locked
	cargo +$(A0_MSRV) check --manifest-path model/burn-a0/Cargo.toml --examples --locked

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
	python3 scripts/check_component_protocols.py

# The component-evaluation bench workspaces (evaluations/bench/*) are excluded from the
# root workspace, so `make check`/`make test` reach none of them. Same commands as the
# bench steps of CI's `quality` job.
bench-check:
	cargo fmt --manifest-path evaluations/bench/kit/Cargo.toml -- --check
	cargo check --manifest-path evaluations/bench/kit/Cargo.toml --all-targets --locked
	cargo clippy --manifest-path evaluations/bench/kit/Cargo.toml --all-targets --locked -- -D warnings
	cargo test --manifest-path evaluations/bench/kit/Cargo.toml --locked

msrv:
	cargo +1.99.0 check --workspace --all-targets --locked
	cargo +1.99.0 test --workspace --locked

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
# Not reproduced: the Windows-only `cargo check -p ptr-net --features wintun-backend` step (wintun does not compile on Linux), the Windows leg of rust-stable and the Python 3.11 leg of
# python-training (matrix legs of the same steps), `pip install -e training`
# (PYTHONPATH is set instead, so nothing is installed into your environment),
# and security.yml, which needs cargo-audit and cargo-deny. `make a0` is the
# burn-a0 workflow, which runs only when A0 or what it depends on changes.
ci-local: ci-quality ci-rust-stable ci-rust-msrv ci-python-training ci-repository-invariants \
	ci-lifecycle-failpoints ci-ledger-raft-engine ci-ledger-raft-rs ci-state-turso ci-state-postgres \
	ci-network-iroh ci-cluster-wire ci-execution-wire ci-pod-wire ci-storage-tiers ci-feature-gates

ci-quality:
	cargo +stable fmt --all -- --check
	cargo +stable clippy --workspace --all-targets --locked -- -D warnings
	RUSTDOCFLAGS="-D warnings" cargo +stable doc --workspace --no-deps --locked
	cargo +stable test --manifest-path templates/rust-crate/Cargo.toml --locked
	cargo +stable clippy --manifest-path templates/rust-crate/Cargo.toml --all-targets --locked -- -D warnings
	cargo +stable test -p ptr-observe --features tracing-adapter --locked
	cargo +stable clippy -p ptr-observe --features tracing-adapter --all-targets --locked -- -D warnings
	cargo +stable fmt --manifest-path evaluations/bench/kit/Cargo.toml -- --check
	cargo +stable check --manifest-path evaluations/bench/kit/Cargo.toml --all-targets --locked
	cargo +stable clippy --manifest-path evaluations/bench/kit/Cargo.toml --all-targets --locked -- -D warnings
	cargo +stable test --manifest-path evaluations/bench/kit/Cargo.toml --locked

ci-rust-stable:
	cargo +stable check --workspace --all-targets --locked
	cargo +stable test --workspace --locked

ci-rust-msrv:
	rustup toolchain install 1.99.0 --profile minimal
	cargo +1.99.0 check --workspace --all-targets --locked
	cargo +1.99.0 test --workspace --locked
	cargo +1.99.0 check -p ptr-observe --features tracing-adapter --locked
	cargo +1.99.0 test --manifest-path templates/rust-crate/Cargo.toml --locked

ci-python-training:
	PYTHONPATH=training/src $(PYTHON) -m unittest discover -s training/tests
	PYTHONPATH=training/src $(PYTHON) training/src/ptr_training/validate_dataset.py datasets/samples/epistemic_calibration.jsonl
	PYTHONPATH=training/src $(PYTHON) training/src/ptr_training/validate_dataset.py datasets/samples/ood_pod.jsonl
	PYTHONPATH=training/src $(PYTHON) training/src/ptr_training/validate_dataset.py datasets/samples/operator_route.jsonl

ci-repository-invariants:
	@test -n "$(BASE)" || { echo "BASE is empty: fetch origin/main, or pass BASE=<commit>" >&2; exit 2; }
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
	$(PYTHON) scripts/check_component_protocols.py
	$(PYTHON) scripts/check_research_gates.py
	$(PYTHON) -m unittest discover -s scripts/tests
	$(PYTHON) -m unittest discover -s research/baselines/rag_reference/tests
	$(PYTHON) -m unittest discover -s research/baselines/strong_rag/tests
	$(PYTHON) -m unittest discover -s research/baselines/strong_retrieval/tests
	$(PYTHON) -m unittest discover -s research/embeddings/tests
	$(PYTHON) research/embeddings/verify_manifest.py --regenerate
	$(PYTHON) -m unittest discover -s research/consciousness/tests
	$(PYTHON) -m unittest discover -s docs-site/tests
	$(PYTHON) benchmarks/operator-routing/generator.py
	$(PYTHON) -m unittest discover -s benchmarks/operator-routing/tests
	$(PYTHON) benchmarks/operator-routing-v2/generator.py --check
	$(PYTHON) -m unittest discover -s benchmarks/operator-routing-v2/tests

ci-lifecycle-failpoints:
	cargo +stable test -p ptr-ledger --features failpoints --test failpoints --locked
	cargo +stable clippy -p ptr-ledger --features failpoints --all-targets --locked -- -D warnings
	cargo +stable test -p ptr-ledger --features raft-engine-failpoints --test engine_failpoints --locked
	cargo +stable clippy -p ptr-ledger --features raft-engine-failpoints --all-targets --locked -- -D warnings
	cargo +stable test -p ptr-bench --features ledger-experiments --locked l001
	cargo +stable clippy -p ptr-bench --features ledger-experiments --all-targets --locked -- -D warnings

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
	rustup toolchain install 1.99.0 --profile minimal
	cargo +1.99.0 test -p ptr-net --features iroh-backend --locked

ci-cluster-wire:
	cargo +stable test -p ptr-cluster --features cluster-backend --locked
	cargo +stable clippy -p ptr-cluster --features cluster-backend --all-targets --locked -- -D warnings
	rustup toolchain install 1.99.0 --profile minimal
	cargo +1.99.0 test -p ptr-cluster --features cluster-backend --locked

ci-execution-wire:
	cargo +stable test -p ptr-execwire --features execwire-backend --locked
	cargo +stable clippy -p ptr-execwire --features execwire-backend --all-targets --locked -- -D warnings
	rustup toolchain install 1.99.0 --profile minimal
	cargo +1.99.0 test -p ptr-execwire --features execwire-backend --locked

ci-pod-wire:
	cargo +stable test -p ptr-podwire --features podwire-backend --locked
	cargo +stable clippy -p ptr-podwire --features podwire-backend --all-targets --locked -- -D warnings
	rustup toolchain install 1.99.0 --profile minimal
	cargo +1.99.0 test -p ptr-podwire --features podwire-backend --locked

ci-storage-tiers:
	cargo +stable test -p ptr-storage --features tier-opendal-fs --test opendal_tier --locked
	cargo +stable check -p ptr-storage --features tier-opendal-fs,tier-opendal-s3 --locked
	cargo +stable clippy -p ptr-storage --all-targets --features tier-opendal-fs,tier-opendal-s3 --locked -- -D warnings

# Feature gates no other job compiles. `candle-cuda` (ptr-pods / ptr-runtime) is
# deliberately excluded: it needs the CUDA toolkit, so nothing here compiles it.
ci-feature-gates:
	cargo +stable clippy -p ptr-runtime --features oidc-http --all-targets --locked -- -D warnings
	cargo +stable test -p ptr-runtime --features oidc-http --lib --locked
	cargo +stable clippy -p ptr-net --features wireguard-uapi-backend --all-targets --locked -- -D warnings
	cargo +stable test -p ptr-net --features wireguard-uapi-backend --lib --locked

# Start a local PostgreSQL 16+ / pgvector service first. These defaults match CI's
# disposable container; override them for your local disposable test database.
ci-state-postgres: export PTR_PG_TEST_DSN ?= host=127.0.0.1 port=5432 user=postgres password=postgres dbname=postgres
ci-state-postgres: export PTR_PG_EXPERIMENT_DSN ?= host=127.0.0.1 port=5432 user=postgres password=postgres dbname=postgres
ci-state-postgres:
	cargo +stable test -p ptr-pg --features postgres-backend --locked
	cargo +stable clippy -p ptr-pg --features postgres-backend --all-targets --locked -- -D warnings
	rustup toolchain install 1.99.0 --profile minimal
	cargo +1.99.0 test -p ptr-pg --features postgres-backend --locked
	cargo +stable clippy -p ptr-bench --features turso-oracle --all-targets --locked -- -D warnings
	cargo +stable run -p ptr-bench --features turso-oracle --locked -- projection-equivalence 2 17
	cargo +stable run -p ptr-bench --features postgres-experiments --locked -- fastmem-revocation 2 17
