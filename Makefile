.PHONY: check test fmt repo-check docs docs-check manifest tree

check:
	cargo check --workspace --all-targets

fmt:
	cargo fmt --all -- --check

test:
	cargo test --workspace

repo-check:
	python3 scripts/check_repo.py

docs:
	python3 scripts/update_component_docs.py --write

docs-check:
	python3 scripts/update_component_docs.py --check

manifest:
	python3 scripts/hash_manifest.py

tree:
	python3 scripts/print_tree.py
