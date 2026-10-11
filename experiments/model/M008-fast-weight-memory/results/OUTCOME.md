# M008 named outcome: FALSIFIED on the four valid seeds, seed 17 interrupted

Owner decision 2026-10-10. Seed 17 was killed by a container restart before it wrote any output (its record stays `started`, committed alone as `b9455b9b`). The research gate publishes no aggregate (`run.json`, `metrics.json`) while a preregistered seed has no finished record, so this experiment stays `prepared` and its outcome is recorded here instead, as for an interrupted listed experiment. Nothing was rerun: each seed runs once, and seed 17 is spent.

The outcome below is the unchanged output of the preregistered aggregator (`aggregate.py`) over the committed records, run at `ee170c6ad4dba977e687b3fe8f3b333be467b99b`.

- aggregate.py sha256: `4b366f1acfdbcca983119508ee021d5760927554c96b835666e11aaaf500ef82`
- mutations.json sha256: `c928d3cebb98806cdc08886190b424eaf472da087805987269f2863b7eb7018e` (killed 15 of 15 at 17717fd054d3)

| seed | record | sha256 | status | exit |
|---|---|---|---|---|
| None | `run-20261009T203450.917952Z.json` | `0b8e15e8175f14c3…` | prepared | None |
| 17 | `run-20261009T203511.881703Z-seed-17.json` | `ff12edaac0e83411…` | started | None |
| 29 | `run-20261009T205434.061982Z-seed-29.json` | `e3a24a93dce7df94…` | failed | 1 |
| 43 | `run-20261009T210222.118158Z-seed-43.json` | `03c0f2aed530335a…` | failed | 1 |
| 71 | `run-20261009T210955.210555Z-seed-71.json` | `000a58d6b5ec06b4…` | failed | 1 |
| 101 | `run-20261009T211718.353637Z-seed-101.json` | `46103b2be9da2455…` | failed | 1 |

## Decision

```json
{
  "verdict": "FALSIFIED",
  "vacuous_seeds": [],
  "falsified_seeds": [
    29,
    43,
    71,
    101
  ],
  "invalid_seeds": [
    17
  ],
  "static_evidence_failed": [],
  "mutations_ok": true,
  "problems": [],
  "recommended_status": "failed"
}
```

## Report (aggregator output, verbatim)

# M008 fast-weight memory: FALSIFIED

Seeds: 17, 29, 43, 71, 101. Every seed must hold every coverage and falsifier rule.

- seed 17: INVALID; 0 coverage rules unmet, 0 falsifier rules tripped
    - invalid: no harness result
    - invalid: run-20261009T203511.881703Z-seed-17.json: it has no result line
- seed 29: valid; 0 coverage rules unmet, 2 falsifier rules tripped
    - falsifiers_failed: recall_gain_vs_recency_lo_permille >= 1
    - falsifiers_failed: recall_gain_vs_hybrid_live_lo_permille >= 1
- seed 43: valid; 0 coverage rules unmet, 2 falsifier rules tripped
    - falsifiers_failed: recall_gain_vs_recency_lo_permille >= 1
    - falsifiers_failed: recall_gain_vs_hybrid_live_lo_permille >= 1
- seed 71: valid; 0 coverage rules unmet, 2 falsifier rules tripped
    - falsifiers_failed: recall_gain_vs_recency_lo_permille >= 1
    - falsifiers_failed: recall_gain_vs_hybrid_live_lo_permille >= 1
- seed 101: valid; 0 coverage rules unmet, 3 falsifier rules tripped
    - falsifiers_failed: false_recall_fast_permille <= false_recall_recency_permille
    - falsifiers_failed: recall_gain_vs_recency_lo_permille >= 1
    - falsifiers_failed: recall_gain_vs_hybrid_live_lo_permille >= 1

- mutation check: 15 of 15 killed
- mutations_ok: True

Limitations: the claim is about a synthetic templated English pool (5 attributes' worth of one-sentence facts and short questions) embedded once by a small general sentence embedder (all-MiniLM-L6-v2, CPU, vectors pinned as half floats); nothing is claimed about real corpora, long contexts, LLM answer quality or other embedders, and sibling cues of one entity are similar by construction, which is the interference the memory is tested against; the memory is keyed by the embedding of a cue text (paraphrase 0 of the cue), which stands for a cue extractor in front of the memory; the recency and hybrid baselines index the fact texts only and have no cue extractor, so the comparison favours the memory on this point; a result in the memory's favour is a statement about the mechanism given such an extractor, not about an end-to-end system; hybrid_live is told which document an update supersedes (the harness deletes the old version), the same kind of cue knowledge the memory gets from its key; the projection and the codebook are single seeded draws used for all five seeds, so variance across draws is not sampled and the bootstrap is over queries only; the claim is scoped to 400 cues written per session, 8 heads x key_dim 32 x value_dim 64 (pilot shapes up to key_dim 384 and 16 heads did not change the picture), untrained seeded random projections of frozen MiniLM vectors; the capacity table (16 to 400 cues) and the unthresholded recall in every run are descriptive and show where the memory holds its cues and where it does not; the matched-operating-point comparison is one point (the memory abstains on most queries there because crosstalk makes unanswerable readouts as strong as correct ones), and the false-recall rule is weak because an arm that abstains almost always passes it; the memory is a derived, revocable projection with random (untrained) orthogonal key projections and sign identifier codes as values; it is bounded by the delta rule's capacity (heads x key_dim), and a result against it is a statement about this configuration and pool, not about learned projections, which are not implemented; the recency buffer is the last 39 writes answered by cosine on the same frozen vectors and cannot answer for facts outside the window; its window is the budget over the longest fact, so it holds the budget in the worst case; the strata (inside and outside the window) are reported and the old stratum is where the buffer cannot win; the hybrid arms are the digest-pinned rag_reference implementation run through a Python driver; its rank score is ordinal, so their abstention is a cosine between the query and the top hit chosen by the harness, which is not part of the pinned baseline; hybrid_append is reported and not gated; the abstention thresholds and the memory shape were set on pilot seeds 1001..1005 by a rule fixed beforehand and are pinned; the confirmatory seeds apply them unchanged, so the realised false answer rate on the unanswerable queries differs from the 5 percent target by sampling (it is reported, and a transfer failure above 150 permille makes the seed vacuous); the pilot may have been optimistic for one arm and pessimistic for another; the truth is the harness's table (the last write of each cue), read three ways that must agree (a forward fold, a backward scan and the generator's label); the oracle shares no code with the arms, but all three share the generator's session; the controls use no code under test and are scored by the same function as the arms; the memory's contract checks (decode agreement, readback, stale residual, projection binding) are read from the readout and the codes the product publishes; the product's decode_readout is compared with a re-implementation of the same rule in the harness, so a defect both share would not be seen; the checks run on the pinned shape only; one process, one machine profile, no concurrency; wall-clock times (timing_ns, latency_ns_per_query) are recorded and never judged; vectors were produced on CPU with torch 2.7.1 and are pinned, so no claim of bit-reproducibility across machines is made.

## Per-seed gate metrics

| seed | gain vs recency (LB, permille) | gain vs hybrid_live (LB, permille) | false recall memory | false recall recency |
|---|---|---|---|---|
| 29 | -66 | -572 | 37 | 42 |
| 43 | -79 | -610 | 50 | 70 |
| 71 | -75 | -575 | 57 | 72 |
| 101 | -69 | -625 | 52 | 35 |
