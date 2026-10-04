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
make ci-local   # the Linux jobs of .github/workflows/ci.yml; CONTRIBUTING.md lists what it does not reproduce
make a0         # when model/burn-a0, anything under crates/ (the A0 bridge depends on ptr-runtime), root Cargo.toml, the codebook or vendor/ changed
```
