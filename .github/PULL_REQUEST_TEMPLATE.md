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
make ci-local   # everything .github/workflows/ci.yml runs
make a0         # when model/burn-a0, crates/ptr-types or the codebook changed
```
