# F002 pilot (not evidence)

`pilot-summary.json` holds the key numbers of 27 full-scale runs of `ptr-bench weak-supervision <seed>`
on seeds that are never confirmatory: 1001, 1002, 1003 (the `pilot_seeds` of `config.toml`) and 2001 to
2024 (a larger power analysis of the same kind). They were run with the final harness (the files were regenerated after the review fixes: calibrated vote shares, census
accuracy gate), before any confirmatory seed other than the single dry run of seed 17. The order of events, the history
of the world parameters and the decisions taken on these numbers are in PREREGISTRATION.md section 7. They
exist to decide the form of the gain gates (PREREGISTRATION.md section 7) and to check that the coverage
minima are reachable and the accuracy rule does not falsify by noise. Every pilot run exited 0 (no
coverage unmet, no falsifier tripped); that says only that the harness runs and the product is
consistent with its oracle on those worlds.

Reproduce one run: `cargo run --release --locked -p ptr-bench -- weak-supervision 1001`. The output
is deterministic apart from `timing_ms`.
