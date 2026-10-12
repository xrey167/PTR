# R013 pilots: pilots, not evidence

Pilots, not evidence. The files here are the stdout JSON (`pilot-100N.json`) and stderr (`pilot-100N.err`) of the release harness
(`cargo run --release --locked -p ptr-bench -- self-modification-authority <seed>`, the pinned entrypoint) at the pilot seeds
1001, 1002 and 1003, run on the working tree of fix round 2 before the freeze. No declared seed (17, 29, 43, 71, 101) was run.
They are not records of the experiment: the runner writes those under `results/` at the freeze.

Their use is one: the freeze procedure (`PREREGISTRATION.md`, section 10, step 7) reruns pilot 1001 on the frozen commit and
compares its outcome (everything but `timing_ns` and `environment`) with `pilot-1001.json`; a difference stops the procedure.
