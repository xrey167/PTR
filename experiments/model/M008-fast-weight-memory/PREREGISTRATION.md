# M008 preregistration

Status: complete text, not yet frozen. At the freeze the head digests this text and `criteria.toml` into the two
`*_sha256 = "must-be-pinned-before-prepared"` keys of `config.toml`'s `[preregistration]` (every other key of that table
is decided, the `rag_reference` directory digest included), sets the entrypoint and the manifest digests, and runs the
full mutation check on the committed tree. Nothing below was tuned on a confirmatory seed (17, 29, 43, 71, 101). The data
looked at are the pilot seeds 1001..1005, seeds 2001 (mutation runs) and the full-scale runs of pilot seeds named in
section 6, and the release dry runs of seed 17 disclosed in section 6.

## 1. Question and scope

Does `ptr-fastmem` (a gated delta-rule associative memory, a derived and revocable projection) recall the *current*
version of an updated fact better than (a) a recency buffer and (b) plain hybrid retrieval over the live facts, at an
equal token budget, without a higher false-recall rate than the recency buffer?

Scope: a mechanism comparison on a synthetic session, with real pretrained sentence vectors as the frozen key
embedding. No claim about LLM answer quality, long contexts, real corpora, learned projections or other embedders. A
negative result is a first-class outcome and the pilot (section 6) already points to one.

## 2. Frozen inputs

- Text pool `research/embeddings/pool-m008.jsonl` (14,000 rows: 1,900 cues x 3 fact versions, 4 paraphrases per cue, 700
  off-topic questions), sha256 `0192b66e...6ef7`; vectors `research/embeddings/vectors-m008.f16` (half floats, 384 per
  row, `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, pinned),
  sha256 `378d013f...62ba`. The harness embeds nothing, verifies both digests, recounts every row's tokens with
  `[A-Za-z0-9]+|[^\sA-Za-z0-9]`, and exits 2 on any mismatch.
- Token budget 512. The longest fact of the pool is 13 tokens (computed at start-up, asserted), so the recency window is
  floor(512 / 13) = 39 (asserted; a mismatch is exit 2).

## 3. Session and arms

A seed draws 400 cues to write (version 1), 200 of them updated later at a random time (version 2, a near-duplicate
fact), 40 further cues nobody writes, and 400 queries, each a paraphrase (1..3) of a cue, never the cue text itself
(paraphrase 0), which keys the writes: 160 about updated cues, 160 about never-updated cues, 40 about unwritten cues, 40
off-topic. 600 writes in all.

Every arm answers with a ranked list of at most five facts; the empty list is Unknown.

1. **Fast memory** (`ptr-fastmem`): `FastMemory::with_projection`; key = seeded orthogonal projection of the frozen
   embedding of the cue text; value = identifier code of capsule `m008:c:<cue>` at generation `<version>`; beta 1, scalar
   decay 0.999; shape 8 heads x key_dim 32 x value_dim 64 (the plan's); readout decoded by `decode_readout` against every
   fact the journal holds (stale versions included) with `min_score` and `min_margin` from section 4.
2. **Recency buffer**: the last 39 writes (old versions included), ranked by cosine between the query and the buffered
   fact on the frozen vectors, the newer first on a tie.
3. **hybrid_live** (gated baseline): the digest-pinned `research/baselines/rag_reference` stack (BM25 k1=1.5 b=0.75 +
   cosine, RRF k=60) over the live facts only (an update deletes the old version), run by `q003_driver.py` through
   `rag_bridge.rs`.
4. **hybrid_append** (reported, never gated): the same over an append-only index.

The oracle is the harness's table of the last write of each cue, read three ways that must agree (forward fold, backward
scan, generator's label); a disagreement is `oracle_label_mismatches` (a harness defect: INVALID, not falsified).
Negative controls (no code under test, scored by the same function): *oracle-cheat* must reach recall 1000 per mille,
*random* at most 100, *stale-first* (answers the oldest version) a false-recall rate of at least 300; any of them not
separated is exit 2.

## 4. Operating points (pilot seeds only)

Each arm abstains below a threshold. They are set on the pilot seeds 1001..1005 (pooled, 2,000 queries, 400
unanswerable) by a rule written in `m008/calibrate.rs` before the pilot was run (the tree was not committed then, so
the order is documented here and not timestamped): the smallest score threshold, in steps of 5 per mille, at which the
arm answers at most 5 percent of the unanswerable queries. Recency and the hybrid arms abstain on the cosine between
the query and their top hit (the driver's RRF score is ordinal and cannot separate a good hit from a bad one); fast
memory abstains on the decode score with a lead `min_margin` over the runner-up from {25, 50, 100, 200}, the margin
giving the highest pooled recall@1 (ties: smaller; every margin gave the same recall at the matched score threshold, so
the choice is immaterial and the margin gives the memory a four-point choice the others do not have). The memory shape
is the plan's unless another shape of the grid reached at least 100 per mille more pooled recall@1 at its matched point;
none did, so the plan's shape was kept. The honest reading of that rule is "no shape of this grid (8 to 32 heads, key_dim
32 to 384, value_dim 64 only) helps at 400 cues", not "the plan's shape is right"; the grid cannot change the answer
because key_dim cannot exceed the embedding width 384. Pinned: `fast_min_score_milli = 510`, `fast_min_margin_milli =
25`, `recency_min_cosine_milli = 580`, `hybrid_live_min_cosine_milli = 670`, `hybrid_append_min_cosine_milli = 670`. The
confirmatory seeds apply them unchanged; the realised rate on the unanswerable queries is reported
(`unanswerable_answered_permille`; on the dry run the memory realised 62 and the others 25, so the memory is, if
anything, looser than matched, which favours it) and a seed on which a gated arm answers more than 150 per mille of them
is vacuous.

Why the memory abstains on about 94 percent of queries at its matched point: crosstalk between the 600 stored
associations makes the readout of an unanswerable query as strong as that of a correct one (the best correct readouts
score about 0.5), so a threshold that keeps the unanswerable answers under 5 percent discards nearly every answer. This
is one operating point; the threshold-free numbers below do not depend on it.

## 5. Decision (criteria.toml)

Every seed must hold every coverage rule and every falsifier rule; COMPLETED additionally needs `results/mutations.json`
with every planned mutation killed. A verdict published without that evidence is FALSIFIED with `mutations_ok = false`,
never COMPLETED. Falsifiers: false recall of the memory over all queries not above the recency buffer's (a weak rule,
since an arm that abstains almost always passes it); positive 95 percent one-sided paired-bootstrap lower bound (2,000
resamples, pinned seed, integer permille) of the recall@1 gain over the recency buffer and over hybrid_live (two rules,
two counters); no arm over the token budget; no product panic or error (constructors and projection calls included);
and the memory's contracts read from the readout (decode equals the harness's own decode and never returns an empty
`Hits`, every write reads back, after an update the stale version scores under 0.4 in the readout of a paraphrase query,
the projection binding refuses foreign queries). Harness and baseline integrity, judged INVALID and not falsified:
oracle label mismatches, a stale version returned by the live hybrid index, a recency window over the budget.
`hybrid_append`, recall@k, Unknown rate, tokens, strata, latency and the descriptive evidence of section 6 are never judged.
The positive control (the oracle-cheat answers standing in the memory's place through the same scoring, bootstrap and
rule path) must pass every outcome rule, else the harness exits 2; this, a unit test with the real scorer, and the
mutation `decode-answers-every-readout` (false recall, at 32 cues) are what show the outcome falsifiers can pass and fire;
the two gain counters have no discriminating mutation, because no run made had a clean gain counter (the memory is below
live hybrid retrieval at every size and seed tried and its gain bounds over the recency buffer were negative on every run).

## 6. Evidence and what it supports

Everything below is descriptive and ships with every confirmatory run (`descriptive.rs`), so it is frozen with it.

**Scope of the conclusion.** At 400 cues written per session, the plan's shape (8 heads x key_dim 32 x value_dim 64;
pilot shapes up to 16 heads and key_dim 384), untrained seeded random projections of the frozen MiniLM vectors, keys
taken from cue texts, and this synthetic pool, the delta-rule memory holds a small fraction of its cues. Nothing is
claimed for fewer cues, learned projections, other embedders or other pools.

**Matched point (pilot, pooled).** Recall@1 at the matched point, per mille: fast 18 at the plan's shape, 19 without
decay, 23 / 25 / 30 / 26 / 35 / 36 / 35 at 8x64, 16x64, 8x128, 32x64, 8x256, 8x384, 16x384 heads x key_dim; recency 60;
hybrid_live 536; hybrid_append 379 (`tests/pilot_calibration.json`).

**Threshold-free (pilot seed 1001, 400 cues, no abstention, top candidate).** Recall@1 per mille: fast 56, recency 65,
hybrid_live 828, hybrid_append 528; fast memory asked with the cue text itself 50. The memory is at the level of the
recency buffer without any threshold, so the loss at 400 cues does not come from the abstention rule; and asking with
the exact cue text does not help, so it does not come from paraphrase generalisation either.

**Capacity table (pilot seed 1001, unthresholded recall@1 per mille, same seed, sessions of N cues written).**

| cues written | fast (paraphrase query) | fast (cue text) | recency | hybrid_live |
|---|---|---|---|---|
| 16 | 750 | 666 | 750 | 833 |
| 32 | 541 | 583 | 750 | 1000 |
| 64 | 360 | 320 | 360 | 1000 |
| 128 | 147 | 186 | 225 | 941 |
| 400 | 56 | 50 | 65 | 828 |

The memory holds its cues while their number stays near the key dimension per head (32) and decays smoothly beyond it.
Pilot seed 1002 gives 916 / 625 / 300 / 127 / 56 for the memory against 583 / 625 / 420 / 225 / 65 for the recency buffer:
at 16 cues (about a dozen answerable queries) the order flips between seeds, from 64 cues up the memory is at or below the
recency buffer on both seeds (whose window of 39 writes covers the whole session up to 26 cues), and the old stratum is
at 0 at 400 cues. Reading: the memory works inside its capacity and behaves like a
bounded recency store outside it. The limit at larger shapes is `key_dim <= 384` (the projection cannot be wider than
the embedding) and the effective rank of the templated MiniLM keys, not the 64 MiB state bound. The threshold sweep (recall
and unanswerable-answered per mille at score thresholds 0..900) is in every run's JSON.

**Dry runs of seed 17 (a confirmatory seed; disclosed).** Seed 17 was run on the release build twice for harness
validation and its output was looked at: first with the harness as of the first review (exit 1, 0 coverage unmet, 2
falsifiers tripped: recall@1 fast 21, recency 46, hybrid_live 559, hybrid_append 365 per mille; gain lower bounds -44 and
-585; false recall fast 25, recency 42), and again after the review fixes (exit 1, 0 coverage unmet, the same 2 falsifiers tripped: recall@1 fast 21, recency 46, hybrid_live 559, hybrid_append 365 per mille at the matched point; unthresholded fast 65, recency 53, hybrid_live 809 (the memory is a little ahead of the recency buffer without any threshold on this seed and far behind live retrieval); the memory's cue-text recall 56; gain lower bounds -44 and -585; false recall fast 25, recency 42, so the weak false-recall rule passes; realised unanswerable-answered rate fast 62 and the others 25; positive control: lower bounds 931 and 393, headroom 42). The first dry run preceded the
empty-`Hits` check, the contract probes' margin counter, the descriptive evidence and the containment of constructor
errors; none of those edits changed an operating point, a shape or a rule threshold, and no parameter was changed after
either dry run. Seeds 29, 43, 71 and 101 have not been run; the full harness was run at the pinned scale on pilot seeds
(1001 and 1002: 0 coverage unmet, the same 2 falsifiers tripped, all contract counters 0) to check the coverage floors.

**Not tried** (untested alternatives, not rejected ones): mean-centring or whitening of the embeddings (an independent
numpy re-implementation in the review found it improves recall by a factor of 1.3 to 2 at 400 cues without rescuing
them), learned projections, per-channel decay, value codes other than random signs, update-aware keys.

## 7. Mutation plan

Fifteen defects (`tests/mutations.toml`): decode margin and score checks removed, decode that answers every readout (32
cues), delta-rule sign flipped, erase term dropped, write strength halved, projection-binding check removed, decode that
panics, decode that refuses a valid policy, a projection constructor that fails, a projection that panics, recency
window plus one, live index that keeps the old version, presented lists longer than the budget, oracle fold that keeps
the first write. Mutation runs use seed 2001 at 400 cues (32 for the false-recall mutation). The full `mutation_check.py
M008` on the committed tree is mandatory at the freeze, and must be rerun after any edit of `fast.rs`, `decode.rs`,
`state.rs`, `memory.rs` or `projection.rs` because the anchors are exactly-once.

## 8. Limits

See `aggregate.py` (`LIMITATIONS`): synthetic templated pool; the memory is keyed by a cue text that the baselines do not
get (favours the memory); hybrid_live is told which document an update supersedes (favours the baseline); random
untrained projections and single seeded draws of the projection and the codebook for all five seeds; hybrid abstention
is a harness choice; thresholds set on 5 pilot seeds; the decode check re-implements the product's rule; single machine
profile; the matched-operating-point comparison is one point.

## 9. Pins for experiments/preregistration.toml and config.toml

`[experiment.M008.required]` lists `seeds`, `harness`, `token_budget`, `recency_window`, `recency_buffer_baseline`,
`hybrid_retrieval_baseline`, `embedding_model`, `embedding_revision`, `embedding_vectors_sha256`, `pool_sha256`,
`preregistration_text` (file) and `criteria` (file), and a `[[experiment.M008.baseline]]` for `rag_reference`
(`baseline_rag_reference_sha256` is the digest of its directory, computed with `check_research_gates.directory_digest`;
any edit of a file under `research/baselines/rag_reference` changes it, and so does a stray `__pycache__`, which the
bridge prevents by running Python with `PYTHONDONTWRITEBYTECODE`). Every value is in `config.toml`'s `[preregistration]`
except the two text digests, which the head pins.

