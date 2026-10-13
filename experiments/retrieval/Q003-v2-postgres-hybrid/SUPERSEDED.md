# Q003-v2 supersession record

Q003-v2 is `superseded` by [Q003-v3](../Q003-v3-postgres-hybrid/README.md) by owner decision of 2026-10-11. Its outcome is
**INVALID (incomplete)**: one of its five declared seeds completed, one was started and never ended, three never ran, no
`run.json`, `metrics.json` or `mutations.json` exists, and this status is not evidence about `ptr-pg`'s retrieval: it neither
confirms nor falsifies the hypothesis.

What happened:

* The preregistration was frozen at `f07386f9` (tag `q003v2-prereg`) and prepared at `28ea8d48`.
* Seed 17 ran to exit 0 (`results/run-20261010T112951.978407Z-seed-17.json`, committed at `aae1ebd2`): 30,693 s (8.5 hours)
  in all, of which 30,350 s were the strong arm (200 cross-encoder queries; about 1,600 s on an unloaded host by the
  preregistered planning figure of 8 CPU-seconds a query). The record's progress lines show the cost arriving in bursts (about 1,200 s
  in the first 50 queries, about 100 s in the next 50, then about 8,300 s and 14,600 s in two later windows of 50 queries, then none).
* Seed 29 was started at 2026-10-10 20:03 UTC (`results/run-20261010T200355.785810Z-seed-29.json`, status `started`, committed at
  `0fc2b319`) and had produced no outcome after about 13 hours when the owner stopped it on 2026-10-11. The seed is spent: a
  listed experiment runs each seed once, and the reservation stays as the record that it ran.
* Seeds 43, 71 and 101 never ran.

Cause: see `experiments/retrieval/Q003-v3-postgres-hybrid/q003v2-hang-diagnosis.md`, which holds the process snapshot taken at
the stop and the re-analysis. In short: the snapshot does not show the deadlock it was first read as. The harness's main thread
waited for a reply (a futex wait), the BM25 and reference drivers were idle on their input, as they are whenever the strong arm
is being asked, and the strong driver was running (on the CPU), so the run was waiting on a very slow cross-encoder on a shared host,
as seed 17 had at times. The v2 harness could not say so: it had no heartbeat, a one-hour request limit that a slow-but-answering
driver never meets, no bound on a request line that a driver does not read, and no bound on the whole run. Nothing in the
record could tell slow from stuck.

What stays as it was: the Q003-v2 manifest (but for its status), configuration, preregistration text, criteria, mutation plan,
harness history and all three records (the prepare record and the two seed records) are not amended. The numbers seed 17
produced steered no rule, threshold, margin or seed of Q003-v3.

Successor: **Q003-v3** (`experiments/retrieval/Q003-v3-postgres-hybrid`). Hypothesis, metrics, baselines, margins, thresholds
and `criteria.toml` are unchanged value for value (no loosening); the harness gains typed stops, a per-request timeout, a
whole-run watchdog, a heartbeat and driver-failure propagation, and the declared seeds are 43, 71, 101 (never run in Q003-v2) and
131 and 173 (new), so that no declared seed was run, reserved or seen before.

This status is not evidence.
