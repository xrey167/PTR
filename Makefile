.PHONY: check test fmt repo-check docs docs-check meta-check manifest tree

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

meta-check:
	python3 scripts/check_component_metadata.py --base HEAD^

manifest:
	python3 scripts/hash_manifest.py

tree:
	python3 scripts/print_tree.py
