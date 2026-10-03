# M002-v6 supersession record

M002-v6 is `superseded`, not a scientific result. Its original frozen commit
`be6181bc36a2e01b82007a80031081fbf99c7d03` ran Seed 17 once as a preflight.
The Rust binary first compiled successfully, then exited with code `2` before
loading data, training, checkpointing, or emitting a metric row:

```
a0_ablation: arm factorized-v2 belongs to M002-v5, not M002-v6
```

The exact runner record is
`results/run-20261002T081141.415000Z-seed-17.json`, SHA-256
`5a2edc9463bc8b74f5180ef17bd10e5c5861192920000e2b1f0d10ea568fa173`.
It binds the unmodified v6 manifest, commit, command, FNV fold values, host,
toolchain, and exit status. It contains no completed evidence and is not an
input to any aggregate.

This is a runner-identity failure, not a model outcome. The prepared v6 study
may not be altered after an execution attempt, so M002-v7 supersedes it with
new versioned arm names registered in the Rust binary itself. No v6 or v5
artifact authorizes a claim or a Learned Backend/M009 unlock.
