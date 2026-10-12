# E007 pilots (seeds >= 1001): not evidence

Release pilots 1001-1003 on the final tree of fix round 3 (pinned scale), written by the harness's stdout
(`.json`) and stderr (`.err`). They are dry runs for the preregistered checks and the reference for the
freeze-time rerun: the freeze reruns seed 1001 on the frozen commit and compares every field except
`elapsed_ms`, `timing_ms` and `late_window_replay_delays_ms` with `pilot-1001.json`. Confirmatory evidence
comes only from the declared seeds in `config.toml`.
