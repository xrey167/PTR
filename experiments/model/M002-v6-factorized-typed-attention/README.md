# M002-v6 — superseded before scientific evidence

M002-v6 correctly repaired the v5 `fold=FNV64` serialization, but its first
commit-bound preflight exposed a second runner contract error: the Rust arm
catalog only registered `factorized-v2` and `factorized-v2-off` for `M002-v5`.
The runner therefore rejected the v6 command before a dataset was loaded or a
metric, checkpoint, or completed record could be produced.

The exact failed runner record is committed in `results/` and its immutable
summary is in `SUPERSEDED.md`. The v6 configuration, gate and record are not
rewritten. M002-v7 is the only successor and adds distinct runner-bound arm
names; it is a new preregistered study, not a repair of v6 evidence.

M009 and Learned Backend qualification remain locked. Neither v5's pilot nor
v6's failed preflight is positive evidence.
