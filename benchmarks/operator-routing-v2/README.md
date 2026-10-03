# operator-routing v2

This directory is the self-contained v2 dataset path. It does not change the v1
generator, model integration, dataset registry, status files, or experiment
manifests.

The confirmatory folds are `evidence-temporal`, `evidence-tabular`, and
`claim-interventional`. `evidence-interventional` is emitted for diagnostic
continuity only; `claim-temporal-development` is reserved for the non-claimable
pilot. In every fold the target role x regime cell is rejected from
`train` and `val`; `test_ood` fixes the target regime and requires a live target
fact with `confidence_bucket > 0`. `test_iid` follows the same target-cell
exclusion as train/validation so non-inferiority is measured on the learned
distribution rather than on the held-out composition.

Commands:

```sh
python benchmarks/operator-routing-v2/generator.py --check
python benchmarks/operator-routing-v2/generator.py --update-lock
python benchmarks/operator-routing-v2/generator.py --write-sample
python -m unittest discover -s benchmarks/operator-routing-v2/tests
```

`--update-lock` writes generated data beneath the git-ignored
`datasets/generated/operator_routing_v2/` tree and updates only this directory's
lock. `--check` regenerates in memory, verifies the lock, and checks generated
files if they are present.
