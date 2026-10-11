# F007 pilots (seeds 1007, 1008, 1009): not evidence

Three runs of `ptr-bench vcw-loop <seed>` at the pinned scale on pilot seeds (>= 1001, never confirmatory) from the final
code, with the release profile of the entrypoint (`cargo build --release`, no override). `pilot-<seed>.json` is the harness's one JSON object, `pilot-<seed>.err` its stderr,
`runs.status` the exit codes and wall times (about 2.5 minutes per seed), `summary.md` the output of
`python3 experiments/feedback/F007-vcw-loop/pilot_summary.py` (the aggregator's own statistics; three seeds, so df = 2).

They show the pipeline works end to end, give the variances for the power analysis, and exposed defects before the
freeze: the gate was tuned at the edge of its grid in round 1, and a fixed edge tolerance of 0.001 fired on a noise-level drift in round 2 (then raised post hoc to 0.005); round 3 replaced the tolerance by a paired-noise rule (3 SE on the same stream B, z = 3 a judgement) and widened the grids to nine points; round 4 made the first draft shared by every arm (the round-3 pilots had shown it differing for 15% of the L3 population), re-derived the thresholds from the planning model, added a sensitivity check to the edge rule and re-ran these pilots on the final code (source digest `0f2a3ff2757ace1ca4a2013e4fa3dccea0c8e8b9b10e7fee72fc786604877433`, the same in all three results; it can never equal the frozen one, see section 11 of the preregistration). The reviewer saw the round-1 pilots; the owner has seen all rounds. Outcome digests of the pilots are
in the JSON (`worlds.<w>.arms.<a>.outcome_digest`): the freeze procedure re-runs seed 1007 on the committed tree and
requires them to match.
