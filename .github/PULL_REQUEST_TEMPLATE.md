## Summary

## Layer
- [ ] domain/runtime
- [ ] model/training
- [ ] experiment/evaluation
- [ ] dataset
- [ ] docs/infrastructure

## Evidence
- [ ] tests added/updated
- [ ] component.toml updated for changed crate implementation
- [ ] generated component docs refreshed
- [ ] experiment/evaluation manifest updated where applicable
- [ ] no hard invariant weakened

## Checks
```
make docs-check
make repo-check
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
