# L002 pilots: pilots, not evidence

These are the harness output of pilot runs on the final pre-freeze code, kept so that the freeze can check that the frozen
commit still produces the same record. They are **pilots, not evidence**: seeds 1001, 1002 and 1003 are pilot seeds (the
declared seeds are 17, 29, 43, 71 and 101 and have not been run by any process), they were used to look at the harness
and tune nothing after the last code edit, and nothing here enters the verdict. `aggregate.py` reads `results/`, never
this directory.

Files: `pilot-<seed>.json` is the one JSON object the harness prints (stdout) and `pilot-<seed>.err` its log (stderr,
including the `explain` lines), from `cargo run --release --locked -p ptr-bench --features ledger-experiments -- raft-recovery
<seed>` (the pinned scale).

## Excluded fields (the freeze comparison)

The same seed on the same code gives the same JSON apart from two fields, which the freeze comparison ignores:

- `timing_ns` (an object), whose only entry is `timing_ns.elapsed_ns` (wall-clock nanoseconds of the whole run); it is the
  only timing field in the record.
- `environment` (an object: `filesystem` and `power_loss_model`), which describes the machine the run was made on; the
  filesystem of a scratch directory can differ between hosts without any change of behavior.

Every other key is compared and a difference stops the freeze. (`measurement_notes` is fixed text; recovery time is a count
of heartbeat rounds, `rounds_to_converge_*`, which is logical and is compared.)
