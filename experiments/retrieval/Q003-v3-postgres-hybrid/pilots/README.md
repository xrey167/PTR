# Pilots, not evidence

Files here are **pilot** runs of the Q003-v3 harness on a seed that is not declared (seed >= 1001). They are not evidence for
the hypothesis, steer no rule, threshold, margin or seed, and no aggregate reads them. Their one purpose is the reference for the
freeze script's pilot rerun: the frozen tree is run once more on the same seed with the same command and its outcome fields must equal the
reference's, so that a difference between what was piloted and what is frozen shows before a declared seed is spent.

* `pilot-1002.json` (standard output, one JSON line) and `pilot-1002.err` (standard error: the dry-run notice, progress lines and the
  600 s heartbeat): the pinned scale with the full strong arm, made with the release binary built from the tree that carries
  this file, by `ptr-bench postgres-hybrid 1002 --strong-threads 1` (the pinned command's arguments; the runner cannot launch an
  undeclared seed, so the binary was run directly with the runner's environment names: loopback DSN, strong interpreter, models
  root). Exit 0, wall time 2,426 s of which 2,198 s the strong arm, host load average 7 to 9 on four logical CPUs, other work
  running. 16 coverage and 11 falsifier rules held; fused recall@10 978 per mille, reference 921, exact dense search 981
  (descriptive only).
* A second pinned-scale run of the same seed with the same arguments, made before the last edit of the harness's stop
  handling (not committed; 2,618 s), agreed with this one on every field except the ones listed below: the strong-arm recall
  fields were bit-equal across the two runs (same host, one thread; this is an observation, not a guarantee, since fp32
  summation order is not bit-reproducible across hosts).

## Fields excluded from the freeze comparison

Every other field of the JSON line, the strong-arm recall and discordance fields included, must be equal. The excluded fields
are the wall-clock ones, and they are the only fields that differed between the two runs:

* `timing_ns.elapsed_ns`, `timing_ns.register_space_ns`, `timing_ns.pg_ingest_ns`, `timing_ns.strong_arm_ns` (all of `timing_ns.*`)
* `latency_ns.pg_search_p50`, `latency_ns.pg_search_p99`, `latency_ns.reference_search_p50`, `latency_ns.reference_search_p99`
  (all of `latency_ns.*`)
* `index_build_ns`
* `measurement_notes.*` (static prose that describes the timing fields; it never differed, it is excluded because it is
  commentary and not a measurement)

No other wall-clock field exists in the record (the heartbeat is on standard error only).
