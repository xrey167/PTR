# Q003 PostgreSQL generation-joined retrieval: preregistration

Pre-freeze text of the preregistration of Q003: the `preregistration_text` file of `config.toml`'s `[preregistration]` table, whose
SHA-256 is frozen with the table at the freeze.
Q003 is `planned`: nothing here is a result and no run is evidence. The decisions are made by `criteria.toml` and applied by
`aggregate.py`; this file gives their reasons, the definitions, the modelling choices, what the harness cannot produce and the
limits of the claim. Every criterion is the text of `experiment.toml` (hypothesis, falsification, metrics) and the pins of
`config.toml`'s `[pinned_thresholds]`; the harness is `bins/ptr-bench/src/experiments/q003/` (subcommand
`postgres-hybrid <seed>`, feature `postgres-experiments`); the product under test is `crates/ptr-pg` (`src/adapters/search.rs`:
`register_space`, `upsert_document`, `search`; the projector in `src/adapters/projection.rs`, whose deletes and `is_admissible` the
harness also holds to the live table) and, through the fusion, `crates/ptr-search/src/fusion.rs`.

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench --features postgres-experiments -- postgres-hybrid <seed> --strong-threads 1`
(section 12: the strong arm's driver runs on ONE torch thread) with `PTR_PG_EXPERIMENT_DSN` naming a loopback PostgreSQL 16+ server with pgvector 0.8+, `PTR_Q003_STRONG_PYTHON` an interpreter
with the `research/embeddings` environment (torch, transformers) and `PTR_Q003_MODELS_ROOT` the directory holding the pinned
`bge-reranker-v2-m3/` (`research/embeddings/fetch_models.py`). None of the three is stored in a result. Optional:
`PTR_Q003_STRONG_RESUME_DIR` (or `--strong-resume-dir`), where the strong arm keeps each query's answer so that a failed run
resumes (section 3). **Q003-v3 supersedes the thread advice of this paragraph as written for Q003 and Q003-v2** (section 12): the
confirmatory strong runs are to be made one seed after another with `--strong-threads 1`, which the runner's fixed command passes
(the reranker's depth, precision and length are not to change; the thread count changes only the fp32 summation order); the
measured cost is about 8 CPU-seconds a query, about 15 s wall at one thread on a host at load 11 to 14 (section 12).

Scope (owner decisions of 2026-10-09, binding; head decisions on cost and on the reading of the claim): the baselines are
`research/baselines/rag_reference` (BM25 with OR semantics + cosine + RRF) and the new `research/baselines/strong_retrieval`
(BM25 + bge-m3 dense + RRF + bge-reranker-v2-m3, retrieval only, pinned, CPU); `strong_rag` stays blocked (E002/E005) and no
answer generator exists here. `rag_reference` is the gating baseline; the strong baseline is **reported, never gated**. The
embedding is a small pretrained CPU model: `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, vectors frozen in `research/embeddings/vectors-q003.f16` (the product's and the
reference's), and `BAAI/bge-m3` vectors in `vectors-q003-bge-m3.f16` (the strong arm's).

**The claim, exactly.** On seeded synthetic live-lifecycle corpora, `ptr_pg`'s generation-joined retrieval (a) never returns a
generation the harness's own live table does not call live (not from the lexical list, the vector list or the fused one, not
even when stale rows are planted behind its back), (b) leaves no superseded or revoked document behind, (c) refuses every late
write of a non-live generation, (d) answers a search the same way twice and returns the live document of at least 45 of the 50
live-body probes through its lexical mode, and (e) has fused recall@10 at most 30 per mille below both `rag_reference`'s (a
sanity floor) and the harness's own exact dense search over the live set (the sensitivity control). **On natural-language
queries ptr-pg's lexical list is empty** (`plainto_tsquery` joins every term with AND and no fact sentence contains the question's
words), so its fused list is effectively its halfvec HNSW dense list: leg (e) compares that list with a BM25-OR + cosine + RRF
hybrid and with the best dense list the corpus allows. It does not test the quality of a working lexical half, and a COMPLETED verdict must not be read as "the PostgreSQL
hybrid matches a hybrid". This is a finding about `ptr-pg` (its `search` documents the lexical list as exact but not that
`plainto_tsquery` ANDs; a note in `search.rs`, the ptr-pg README and its component record is a follow-up for the ptr-pg owners, not
part of Q003), recorded here as a limitation and not tested as a falsifier; a successor experiment with an OR or `websearch_to_tsquery` lexical mode would test the hybrid. The
harness measures the finding (`main_query_lexical_hits`) and separates two effects descriptively: the harness's exact cosine
search over the live set (`recall10_exact_cosine_permille`) against `recall10_pg_vector_only_permille` is the HNSW
approximation, and against `recall10_reference_permille` it is the fusion effect. On this templated corpus every sentence shares
its template words, OR-BM25 is mostly noise and RRF with it lowers the reference below plain dense retrieval, so a product recall
above the reference's is mostly a fusion artefact, not evidence of the quality of a ptr-pg hybrid.

**Sensitivity of the gate.** The rule against `rag_reference` is a sanity floor: it trips only when the product's recall falls more
than 30 per mille below the reference's (912 for a reference at 942), far below the product, because on this corpus the reference
is worse than plain dense retrieval. It sees gross failures only (the mutations that keep one hit or sort the fused list
worst-first). The **sensitivity control** is the rule against the harness's exact dense search over the live set
(`recall10_pg_deficit_vs_exact_permille <= 30`): the exact search shares no code with the product, and the pilot seeds 1001, 1002
and 1003 and the dry runs on 17 and 1001 showed the product's fused list at or within 2 per mille of it (deficits 0, 0 and 2 on the pilots), so 30 per mille is clear of
noise. It catches a mid-severity regression of the dense list (a smaller candidate list, a lower `ef_search`, a truncated
embedding) that the floor passes; the mutation `vector-list-keeps-four-hits` shows it. What it cannot see is a lexical half that
is dead on natural-language queries (it is dead already); only the live-body probes test the lexical mode
(`live_probe_lexical_self_hits >= 45` of 50).

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> N" violation `<= N`). COMPLETED iff every seed meets every coverage rule and trips no falsifier **and the mutation
evidence is complete**: `results/mutations.json` is present for this checkout's code and plan and `killed == total ==` the number
of mutations `tests/mutations.toml` lists (19). FALSIFIED if a coverage rule (the run is vacuous) or a falsifier fails on any
seed; a falsification is never downgraded by the mutation evidence (`mutations_ok` is reported beside it). INVALID if a record is
missing, does not describe the frozen run (this is the "unjudgeable" class: the verdict order is below), a counter is missing or mistyped, the exit status disagrees with the rules, the
no-live-join control returned no planted row, the seed ran at another scale than the pinned one or without the full strong arm
(`procedure.strong_arm` must say `full`, `ran` and `reranked`), a metric is declared unsupported, a harness gap is reported, a
harness-integrity guard tripped (`workload_digest_mismatches > 0`: judged INVALID although the harness also judges it as a
coverage rule, so that the mutation checker has a named counter and an exit status), or the mutation evidence is missing or
incomplete (INVALID says nothing about the hypothesis). No rule changes after a run.

**Verdict order.** (1) A seed whose record is unjudgeable (the reasons above) is INVALID and is not judged further. (2) A seed
whose reference recall is below its floor (`recall10_reference_permille < 900`; a weak reference is no product defect) is judged on
every other rule: any tripped falsifier or unmet coverage rule other than the floor and the deficit against the reference
falsifies it, and the sensitivity control stays in force; if none does, the recall rule against the reference is not evaluable and
the seed is INVALID. (3) The experiment is FALSIFIED if any judged seed falsifies, INVALID if not and some seed is INVALID, not
evaluable, or the mutation evidence is incomplete, COMPLETED otherwise. A stale hit on a seed whose reference was weak is
therefore reported as a falsification. At the pilot's reference recall (921 to 948) and a floor of 900 a weak seed is possible,
and would end the experiment as INVALID (a successor then).

The recall rules are the part of the decision a reader will weigh most. `recall10_pg_deficit_vs_reference_permille` and
`recall10_pg_deficit_vs_exact_permille` are `max(0, baseline recall@10 - fused recall@10)` over the answerable queries, in integer
permille, each bounded by a pinned margin (`recall_margin_permille = 30`, `recall_exact_margin_permille = 30`). A seed on which
PostgreSQL falls short of either by more than its margin falsifies Q003 on that seed. That is a valid outcome and is reported as it
is; nothing in `ptr-pg` is tuned to this experiment. The strong baseline (`strong_retrieval`) is **not gated**: its cross-encoder
costs 16 to 46 s a query on the shared CPU, so it answers a pinned, seed-derived subsample of the answerable queries, and its
recall@10, nDCG@10, the product's recall@10 and nDCG@10 on the same queries, the deficit against it, the discordant queries and an
upper bound on their share are reported with the subsample size. The only rule that touches it is `strong_queries >= 200`, which
says that the arm ran.

## 2. The workload

One seed fixes everything below from the pool `research/embeddings/pool-q003.jsonl` (12,000 rows: 2,000 cues, each with three
fact versions and three paraphrased queries; the pool, the MiniLM vectors and the bge-m3 vectors are hashed at start-up and the
run exits 2 on a mismatch with the pins of `config.toml`, and every pool row's token count is recounted).

* **Capsules.** One capsule per cue (`cap-0000` to `cap-1999`). Generation `g` of a capsule is the fact version `g` of its cue,
  so generations differ in their value and in a seeded template choice and an older generation is a plausible near-duplicate of
  the current one.
* **Lifecycle.** A seeded shuffle assigns each capsule one fate, in permille of the capsules: 350 kept at generation 1 (700),
  250 superseded once (500, live generation 2), 200 superseded twice (400, live generation 3), 75 revoked at generation 1 (150),
  62 revoked at generation 2 (124) and the rest revoked at generation 3 (126). The events are applied through `ptr-pg`'s
  projector (`apply_committed` with the ledger's own anchor chain) in four phases: every `CapsuleCommitted`, every first
  `CapsuleSuperseded` (1 to 2), every second (2 to 3), then every `Revoked`. After each commit or supersession the document of
  the new generation (text and vector from the pool) is written with `upsert_document`; the projector deletes the superseded or
  revoked generation's document in its own transaction. A projector record that is refused, not applied or panics is a product
  counter (`product_errors`, `product_panics`), not exit 2; the chain stops there and every later write is refused and counted.
* **Checkpoints.** The first after the supersessions and before any revocation (every capsule answerable), the second after the
  revocations. 1000 queries per seed (the pilot found 400 too few, section 8), each for a distinct capsule, the paraphrase drawn by
  the seed: 300 at the first checkpoint (100 each for live generations 1, 2, 3, of capsules that stay), and 700 at the second: 100
  for revoked capsules (34, 33 and 33 over the revoked generations; unanswerable) and 600 for live capsules (200 each for
  generations 1, 2, 3). 900 queries are answerable, and 600 of them ask for a capsule with a superseded generation.
* **Plants.** Before the queries of each checkpoint stale rows are inserted by raw SQL into the derived table, bypassing
  `upsert_document` (the state that an index lagging the ledger, or a failed delete, leaves): the predecessor generation of every
  superseded target queried, the revoked generation of every revoked target queried (whose `live_generation` row still names it,
  so that only the tombstone clause stops it), and 100 more of other capsules at each checkpoint (older generations and, after
  the revocations, revoked ones). Each is the pool text and vector of that generation, so it matches the same queries as the live
  document. 900 rows per seed. A plant is `INSERT ... ON CONFLICT DO UPDATE`: it is the intended state of the row, whatever the
  projector left there. Before the plants of each checkpoint the harness lists the derived table and counts every row that is
  neither live, nor planted, nor an earlier late write (`projector_residual_rows`, a falsifier: a projector that fails to delete
  a superseded or revoked document leaves rows that the search join hides).
* **Late writes** (second checkpoint, before its plants): 60 attempts of `upsert_document` for a generation that is not live: 20
  superseded, 20 revoked, 10 ahead of the live generation and 10 for a capsule never committed (`ghost-NNNN`). The pairs are
  disjoint from the plants; a pair whose row already exists (a residual row, counted separately) is skipped and counted
  (`late_write_precondition_rows`), not an exit 2.
* **Probes** (second checkpoint, never part of recall): 50 queries whose text is a planted row's own body and 50 whose text is a
  live document's body. They exist because the main queries are natural-language questions (for instance "Which language does
  Marta Jessop speak at home?"), which `plainto_tsquery('simple', ...)` joins with AND term by term and which no fact sentence
  satisfies, so the lexical mode returns nothing for them; a probe is a text the lexical mode can match, so the lexical lifecycle
  join and the mode itself are exercised and audited.
* **Repeats.** The second-checkpoint ones of 20 evenly spaced queries are asked twice on the unchanged snapshot; a differing answer is the product's (`nondeterministic_searches`, a falsifier), not the harness's.

The SHA-256 of the canonical rendering of the workload (fates, queries, plants, late writes, probes, repeats, the strong subset, seed, shape) is the
workload digest. It is pinned per seed at the freeze (`workload_digest_seed_N`), compared at the pinned shape
(`workload_digest_mismatches`) and measured without a server by `postgres-hybrid --print-workload-digest <seed>`
(`tests/pin_workload.py` rewrites both pins).

## 3. The arms

* **PG** (code under test): `HybridQuery` with the project filter, `limit = 40` per mode, weights 1:1, `rank_constant = 60`;
  the fused list is the arm's ranking, its first 10 entries the top 10. The relevant document of a query is the live generation
  of its target capsule.
* **Reference** (`rag_reference`, through `q003_driver.py`): `HybridRAG` (BM25 top `max(4k, 10)`, cosine top `max(4k, 10)`, RRF
  with k = 60) over the MiniLM vectors, limit 10, rebuilt from the oracle's live set at each checkpoint (`reset`, `upsert_many`,
  `size` checked against the live count; a disagreement is exit 2).
* **Strong** (`strong_retrieval`, through its `driver.py --reranker bge`): BM25 top 100 and bge-m3 dense top 100, RRF (k = 60),
  the fused top 50 reranked by `bge-reranker-v2-m3` (fp32, CPU), limit 10, rebuilt the same way. Every reply must say
  `reranked: true`; the model directory is digest-checked by the driver. It answers a **subsample of 200 of the 900 answerable
  queries** (`strong_queries`), fixed by the seed and the rule in `workload.rs`/`pick_strong`, which reads nothing but the strata
  and the seed-shuffled query order and so is independent of every outcome: the answerable queries are grouped in strata by
  (checkpoint, live generation); each stratum contributes `floor(size * 200 / answerable)` of its queries and the remainder
  goes to the strata with the largest fractional parts (ties by stratum order); within a stratum the first queries in the
  seed-shuffled order are taken. Revoked targets are never asked of it (they have no relevant document). The subsample is part
  of the workload digest. Reported on the subsample: `recall10_strong_permille`, the product's `recall10_pg_on_strong_subset_permille`
  and `ndcg10_pg_on_strong_subset_permille` beside `ndcg10_strong_permille`, `recall10_pg_deficit_vs_strong_permille`,
  `strong_discordant_queries` (one arm hits, the other misses) and `strong_discordance_bound_permille` (a 95 percent upper bound
  on their share: 3/n when there is none, `p + 1.96 sqrt(p (1 - p) / n)` otherwise; a standard error would read 0 when both arms
  hit every query). Its wall time is recorded per seed (`timing_ns.strong_arm_ns`).
  **Resume.** With `--strong-resume-dir` each query's answer is appended to `q003-strong-seed<seed>-<digest>.jsonl` as soon as it
  exists; a run that starts again with the same file reuses the answers (`procedure.strong_arm.resumed_queries`), loads the
  strong index only if some query still needs the driver, and refuses a file whose header (seed, workload digest, vectors digest,
  dense and reranker revisions the driver reports, mode, k, the fingerprints of the `strong_retrieval` and `rag_reference`
  directories, the harness commit and whether the harness, crates or baselines are modified) differs, which has a line for a query
  that is not one of the arm's, a malformed or over-long line, more than k ids a line or more than 1 MiB, or an id that is not a
  live document of the checkpoint; a last line cut short by a kill is cut back before the next append. A fresh resume directory is
  to be used for the confirmatory run unless `resumed_queries` is explained in its report. A resumed answer is exactly what an earlier run of
  the same driver returned, and fp32 scores are not bit-reproducible across machines or thread counts, so a resumed run mixes
  answers computed under possibly different conditions; the record says how many. A custom-scale run (the mutation checker) and
  `--no-strong` skip the arm; the aggregator accepts only the full arm at the pinned scale.
* **Descriptive lists:** the PG lexical list and the PG vector list alone (first 10 of each), BM25 alone (`bm25_driver.py`,
  `BM25` imported from `rag_reference/runner.py`) and the harness's own exact cosine search over the live documents
  (`recall10_exact_cosine_permille`; ties by document id as the reference breaks them).
* The queries and documents of the reference and PG arms are the same texts and the same half-precision vectors (the reference
  receives the half values widened to single precision as decimal text), so the arms differ in their retrieval, not their input.

## 4. Oracle, controls and cross-checks

* **Oracle.** `oracle.rs` keeps the live table (capsule to live generation and revoked flag) from the workload's events; it
  shares no code with `ptr-pg`, `ptr-search` or `ptr-runtime` and never asks the product. Every hit of every list of every product
  answer (lexical, vector, fused; main queries and probes) not live in it is a stale hit (`stale_hits`, by list in
  `stale_hits_by_list`); a hit equal to a planted pair is also `planted_stale_returned`.
* **Projection cross-check.** At both checkpoints the product's `is_admissible` is asked for every capsule and generation 1 to 3
  and compared with the table (`lifecycle_projection_mismatches`, a falsifier: the projector is part of what makes the join true).
* **Late writes.** An attempt is refused correctly iff it returns `PgError::NotLive` and the harness's own SQL finds no row for
  the pair afterwards. `Ok`, or a stored row, is `late_write_accepted`; any other error is `late_write_wrong_refusal`.
* **Negative control.** For each stale probe the harness runs the same two queries without the lifecycle join (its own SQL). The
  exact lexical mode must return the planted row for every stale probe, else the run exits 2; `control_planted_returned >= 50`
  and `control_vector_returned >= 50` count the two modes (the vector side is a coverage rule). It shows that the planted rows are reachable by the queries the product answers.
* **Panics.** Every call into the product runs under `catch_unwind`; a panic is `product_panics`, the substrate is reconnected
  and the run goes on. A product error on a live write, a search, a projector record or an admissibility question is `product_errors`.
* **Reproducibility.** The repeated searches must answer identically (else `nondeterministic_searches`, a falsifier).
* **Lexical probes.** The live-body probes (50) must each get their own live document from the lexical mode: at least 45 of 50
  (`live_probe_lexical_self_hits`, coverage) and, relatively, 90 percent of the probes asked (`lexical_probe_violation`, the
  count the mutation checker reads).

## 5. Metrics (JSON integers; grouped metrics are objects of integers)

The 49 metrics of `experiment.toml`. Recall@10 is the fraction (permille, floored) of answerable queries whose relevant document
is in the first 10 of the arm's list; nDCG@10 is `1/log2(rank + 1)` for the relevant document at rank 1 to 10, averaged over the
answerable queries and floored (one relevant document, so the ideal DCG is 1). `recall_paired_se_vs_reference_permille` is the
standard error of the mean paired difference of the two arms' hit indicators, in permille, rounded up. `main_query_lexical_hits`
is the number of lexical hits summed over the main queries. `latency_ns` holds p50 and
p99 of the product's search and of the reference's search (nearest rank); `index_build_ns` is the product's `register_space`
rebuilding the HNSW index over the final corpus after every query (the index is dropped by raw SQL first). Timings are never read
by a rule.

## 6. Exit 2 (the harness cannot run), and nothing else

Bad arguments; an input whose SHA-256 differs from its pin, a pool row whose token count differs from the recount, a vectors
file of the wrong size; a missing `PTR_PG_EXPERIMENT_DSN`; `register_space` or the harness's own SQL (planting, listing, the
control) failing; a driver that does not start, answers badly, refuses a request, answers `reranked` other than the mode needs or
does not answer within an hour; a stale or malformed resume file, or one whose answers name a document that is not live; a baseline index that does not hold the live set; the
no-live-join control missing a planted row in the exact lexical mode; bad arguments to the harness. Everything the product does
wrong, a panic and a failing projector included, is a counter and a falsifier. One exception, stated: a product that cannot
bootstrap its schema (`migrate`), create the index (`register_space`) or reconnect after a panic, or that rejects the harness's
raw plants by a constraint, makes the experiment unrunnable and is INVALID (exit 2 or a harness panic), not FALSIFIED.

## 7. Weak metrics and what is not measured

* The lexical mode is not measured on the main queries (it answers none of them); the lexical join is tested on the probes only.
  See the claim above: the recall comparison is a comparison of an approximate dense list with a hybrid, not a test of a hybrid.
* Latency and index build time are one session on one local server and are not gated.
* The rank constant, weights and limit are the product's customary defaults and the reference's equivalent depth; no sweep.
* Concurrency (a writer racing the projector) is the subject of the `ptr-pg` integration tests, not of this experiment.
* The strong arm's reranker is fp32 on shared CPU; its scores are not bit-reproducible across machines or thread counts.

## 8. Margin and power

The margin is 30 per mille. The plan derived it as three paired standard errors of the difference at 400 queries and recall about
0.9 (a standard error of about 10 per mille), so that the margin cannot be reached by noise alone on one seed, and required the
pilot to confirm that standard error and, if it did not, to raise the query count and never to loosen the margin.

The pilot (seeds 1001, 1002 and 1003, outside the confirmatory five, `postgres-hybrid <seed> --no-strong`) measured the paired
standard error of the recall difference between the product's fused list and `rag_reference` at **14, 14 and 16 per mille at 400
queries (300 answerable)**: not confirmed. The query count was therefore raised to 1000 (900 answerable; the share of revoked
targets lowered from 25 to 10 percent so that the increase buys answerable queries), and the pilot repeated: **8, 10 and 9 per
mille, at most 10**. The margin stays 30. The pilot seeds and the confirmatory seeds draw from the same 2000-cue pool, so the
five seeds are not five independent corpora. `recall_paired_se_vs_reference_permille` is reported by every run and is
descriptive: no rule reads it, because a noisy estimate of the standard error must not be able to make a seed vacuous.

Each recall rule is a point estimate with a margin of about three standard errors. Per rule and seed: at zero true deficit it
falsifies with probability about 0.1 percent (about 0.7 percent over the five seeds if they were independent; they share one
pool, so they are correlated and the figure is an upper bound), and a true deficit of 20 per mille would falsify 10 to 16 percent
of seeds (standard errors of 8 to 10), about 40 to 58 percent of five-seed experiments if independent. The rule against the
reference does not see a drop of the product's recall of less than about 66 to 79 per mille from the level the dry runs showed
(the gate sits at the reference's recall minus 30, and the reference is below the product); the rule against the exact search
sees a drop of more than 30. No power beyond that is claimed.

Mutation severity, measured on the pilot seed 1001 at the pinned 1000 queries (not a confirmatory seed): the mutation
`vector-list-keeps-four-hits` leaves the product's recall@10 at 948 per mille against the harness's exact-search 984 (deficit 36,
above the margin of 30, so `recall_violation_vs_exact` fires) and equal to `rag_reference`'s 948 (deficit 0, so
`recall_violation_vs_reference` stays at 0). It is therefore caught only by the sensitivity control, which is its purpose. A
five-hit version of the same defect (estimated 962) survived and was replaced by this one.

What the pilot also showed, stated here so that it is not read as a result: on those three seeds the product's fused recall@10
(978 to 981 per mille) was above `rag_reference`'s (921 to 948) and equal to its own vector list's, the product's lexical list
having returned nothing for any main query and BM25 alone reaching 865 to 888. The one-sided falsifier (a deficit above the
margin) was therefore not close to tripping against the reference.

**Seed 17 is not blind.** It was run end to end (a dry run, 991 against 942 per mille, outcome seen) and it was the seed of every
mutation run before the freeze, with earlier revisions of the harness; the dry run on the final harness was the pilot seed 1001; the other four confirmatory seeds were only generated (their workload digests
measured, their plants and late writes checked disjoint by a test), never run. Seeds 29, 43, 71 and 101 have therefore not been
run end to end before the confirmatory run, so an exit-2 path may first appear there.

## 9. Mutation plan

`tests/mutations.toml` plants 19 defects one at a time; each is killed iff the harness exits 1 and one of the counters its plan
names fired (`failed-elsewhere` does not count). Sixteen defects are in `ptr-pg` (the lexical join, the vector join, the tombstone
clause and the generation test of the lifecycle join; the generation check and the tombstone check of `upsert_document`; the
error of a refused late write; an inverted live check; a panic in the hit conversion; a vector scan that keeps one hit and one that keeps five (the mid-severity regression for the sensitivity control); an empty
lexical list; a vector list that flips on alternate searches; a projector that leaves superseded documents behind and one that
refuses supersessions; an `is_admissible` that ignores tombstones), one in `ptr-search` (the fused
order) and two in the harness (an oracle that forgets supersessions; a generator that draws another workload). Every one of the
thirteen hard counters is expected by at least one mutation (a test checks this). The mutation checker invokes
`postgres-hybrid 400 17` (the generator mutation `postgres-hybrid 1000 17`, the pinned shape, because the workload digest is
compared only there; `vector-list-keeps-four-hits` runs the pilot seed 1001 at the pinned 1000 queries): custom scale or a pilot seed,
without the strong arm.

## 10. Open points for review

* The strong baseline is reported on a 200-query subsample of the answerable queries and never gated (head decision, cost); the
  owner decision named it the baseline, so a successor may gate it on a larger subsample.
* The AND semantics of `plainto_tsquery` is a finding about `ptr-pg` (the lexical half of its hybrid is inert on natural-language
  queries); documenting it in `search.rs`/the README and a successor with an OR lexical mode are follow-ups, not part of Q003.
* The pins `workload_digest_seed_N` are measured by `tests/pin_workload.py` and re-checked on the frozen commit.

## 11. Successor of Q003 (Q003-v2)

Q003-v2 is the successor of Q003, which was frozen at `07c44206` (tag `q003-prereg`) and is `superseded` (see Q003's
`SUPERSEDED.md`). Sections 1 to 10 above are Q003's text unchanged; this section is the only addition, and where it differs from
them (the status "planned", the flags and the resume directory of sections 1 and 3) this section governs.

Why: Q003's seed 17 ended with exit status 2 before any product call. `scripts/run_experiment.py` passes a listed
experiment's command only an allowlisted environment (`PATH`, `HOME`, `TMPDIR`, ...), so `PTR_PG_EXPERIMENT_DSN`,
`PTR_Q003_STRONG_PYTHON` and `PTR_Q003_MODELS_ROOT` never reached the harness, which stopped at its first check ("the strong
arm needs an interpreter with torch and transformers"). That seed is spent: each seed of a listed experiment runs once.

What changed, and only this:

* the runner may pass variables through by declared name: the manifest declares
  `command_environment = ["PTR_PG_EXPERIMENT_DSN", "PTR_Q003_STRONG_PYTHON", "PTR_Q003_MODELS_ROOT"]` (names matching
  `PTR_[A-Z0-9_]+`, at most eight). A run record names each declared variable, whether it was set and whether the runner redacts
  it in the recorded output, never its value nor a hash of it (a DSN carries a password). The declaration is part of the frozen
  manifest, like its other fields. The three values are therefore not recorded by content: the strong interpreter (torch,
  transformers) and the models directory lie outside the commit, as the toolchain's libraries do; the strong driver's hello in
  the result names the pinned model revisions it loaded, and the strong arm is descriptive and never gated;
* the runner refuses, before it reserves a seed, a run whose declared variable its own environment does not set, sets empty or
  sets with whitespace at either end;
* the manifest declares a `preflight` command (`postgres-hybrid --preflight`, new in the harness), which the runner runs in the
  command's own environment before it reserves a seed; a non-zero exit refuses the run and nothing is reserved or recorded. It
  checks that the server answers and has PostgreSQL 16+ and pgvector 0.8+, that the strong interpreter is an absolute file that
  imports `torch` and `transformers`, that the models directory is absolute and holds the pinned reranker, and it answers one
  strong-arm query end to end through the driver (reranker included) and one query each through the reference and BM25
  `python3` drivers (started from the runner's `PATH` with the arguments a seed uses), and one `ptr-pg` hybrid query on a temporary
  schema that it drops. It measures nothing. If the preflight fails because the product arm itself exits, `q003_0_*` schemas may
  stay in the database; they may be dropped by hand and cannot affect a seed (a seed's schema prefix carries its own seed number);
* the identifiers: the experiment id, the paths and the name of this directory, the `experiment` field of `criteria.toml`, and
  the id the harness and `aggregate.py` use for the result; the harness otherwise differs only in the `--preflight` mode.

Under the runner the command is fixed (`... postgres-hybrid <seed> --strong-threads 1`): the one flag it passes is
`--strong-threads 1` (section 12; Q003 and Q003-v2 passed none, so the driver's default of 4 threads applied), and
`--strong-resume-dir` and `PTR_Q003_STRONG_RESUME_DIR` (not declared) are not used by the frozen
command, and a resume directory would be moot, since a seed of a listed experiment runs once.

Unchanged in every value: the harness's behaviour in a run (`postgres-hybrid <seed>`, `bins/ptr-bench/src/experiments/q003/`),
`criteria.toml` (but for its `experiment` id), the thresholds and pins of `config.toml`, the workload and its five digests, the
baselines, the mutation plan and `aggregate.py` (but for its experiment id). A test compares the parsed criteria, thresholds,
manifest values and plan with Q003's.

Disclosure. Q003's seed 17 confirmatory run produced no product observation: its record holds the build output and the harness's
refusal, no metric and no PostgreSQL call. Seed 17 nevertheless stays exactly as non-blind as section 2 above says (the dry run
of 991 against 942 per mille, the seed of every pre-freeze mutation run); seeds 29, 43, 71 and 101 were only generated, never run
end to end; all five seeds are kept for Q003-v2 because nothing was observed in the confirmatory run that could steer a choice.
The facts of the pilot (sections 3 to 9: workload, pilot seeds, measured costs, the AND semantics of `plainto_tsquery`) carry over;
the pilot seeds are not confirmatory seeds. After Q003 was superseded the successor was developed and checked with these runs, none
of them a confirmatory seed: the unit tests; pilot seed 1001 at 200 queries without the strong arm (a development run: coverage
floors unmet by design, no product error, no stale hit, fused recall@10 966 against the reference's 905 on 180 answerable queries);
the start of seed 1001 at the pinned scale, stopped after 480 s at the first query of the strong arm; and the `--preflight` runs
reported with the freeze. No metric of them steers any rule, threshold or seed of this experiment.

Accepted risk. The declared values are in the environment of the whole `cargo run`, so every build script and proc macro of the
dependency graph (vendored sources included) can read the DSN. It names a loopback test database; use a role with no password
(trust on 127.0.0.1) or a password worth nothing elsewhere. Redaction in the recorded output is best-effort defence in depth (the
value as printed, JSON-escaped or percent-encoded, for a value of 8 or more characters); the rule is that the harness never prints
a declared value.

## 12. Successor of Q003-v2 (Q003-v3)

Q003-v3 is the successor of Q003-v2, which was frozen at `f07386f9` (tag `q003v2-prereg`) and is `superseded` (see Q003-v2's
`SUPERSEDED.md`; its outcome is INVALID, incomplete). Sections 1 to 11 above are Q003-v2's text word for word
except for exactly three paragraphs, all of which state the thread flag of the command and nothing else: (a) the paragraph of the
opening block that gives the command (`postgres-hybrid <seed>` reads `postgres-hybrid <seed> --strong-threads 1`, with a pointer to
this section); (b) the sentences of the same block that advise a thread count (`--strong-threads N` set to the physical cores in
v2; here `--strong-threads 1`, with the measured cost restated at one thread); (c) the paragraph of section 11 that says the runner's
command is fixed (v2: no flag, the driver's default of four threads; here the one flag `--strong-threads 1`). A test
(`scripts/tests/test_q003_v3_successor.py`) fails unless these are the only three changed blocks. This section is the only
addition, and where it differs from sections 1 to 11 in what they say about the run (the status "planned", the seeds, the one-hour
reply limit of section 6, the exit-2 list of section 6) this section governs; those sections are not edited, so their words stand
as Q003-v2's, superseded on those points here.

**Chronology (disclosure).** Q003 was frozen, ran seed 17 to exit 2 before any product call and was superseded (section 11).
Q003-v2 was frozen at `f07386f9`, prepared at `28ea8d48`, and its seed 17 ran from 2026-10-10 11:29 UTC to 20:03 UTC (exit 0, 30,693
s, of which 30,350 s the strong arm; record committed at `aae1ebd2`). Its seed 29 was started at 20:03 UTC and had produced no
outcome after about 13 hours when the owner stopped it on 2026-10-11 (record `started`, committed at `0fc2b319`; the seed is
spent). Seeds 43, 71 and 101 never ran. By owner decision of 2026-10-11 Q003-v2 is stopped and reported INVALID (incomplete) and
this successor carries the harness fix. The record of seed 17 is committed and was readable by the people who run this programme;
**no number in it steered anything here**: the hypothesis, the margins and every threshold were fixed before seed 17 ran and are
unchanged, and seed 17 is dropped from the declared seeds for that reason. **What did inform the harness limits:** Q003-v2's timings,
seed 17 included (8.5 hours, stretches of about 1,500 s a request) and the pilots', were used to size the harness limits below
(the request limit, the run limit and the heartbeat), not for any outcome number or threshold. The limits are not thresholds: no
rule, metric or verdict reads them, and they only decide when a run that cannot finish is stopped as a harness stop.

**Chronology of the thread-count change.** Registered with the harness fix, before the freeze and before any declared seed ran:
the first draft of this section advised `--strong-threads 1` or 2 for a shared host but left the runner's command without a flag
(four threads). Pilot 1001 (four threads, still running after more than four hours) and pilot 1002 (one thread, 56 minutes) showed
that a confirmatory run through the fixed command would be 15 to 60 times slower on this host, so the head decided on 2026-10-11
that the command pins one thread. This changes the command only (a manifest field), by a successor that is still `planned`; no
declared seed saw either setting.

**Why the harness changes.** The head's first reading of the stopped run was a pipe deadlock between `ptr-bench` and the BM25
driver. The snapshot does not show one (`q003v2-hang-diagnosis.md`, in this directory): the main thread was waiting for a reply, the
reference and BM25 drivers were idle on their input, as they are whenever the strong arm is asked, and the strong driver was on the CPU.
Seed 17 had already shown the strong arm costing 100 to 180 times its price in stretches on this shared host. What the harness
lacked is a way to tell slow from stuck and a bound that fires: no heartbeat, a one-hour reply limit that starts after the request
line is written (so a driver that does not read a large request blocks the writer for as long as the driver lives) and that a slow
but answering driver never meets, no bound on the whole run, and no notice of a driver that ends badly. Nothing of this touches a
metric or a rule.

**What changes, and only this (the harness, `bins/ptr-bench/src/experiments/q003/`):**

* the request line is written, flushed, and answered inside one bounded exchange: a thread of its own writes it, and one limit,
  `request_timeout_seconds = 7200`, covers write, flush and reply; when it passes, the driver is killed (so a late reply cannot
  be read as the answer to the next request) and the run ends with the typed reason `request-timeout`;
* a request must be one line (no line break inside it): `bad-request` otherwise; a reply that is not one JSON object is `bad-reply`;
  a refusal (`{"ok": false}`) is `driver-refused`;
* a driver that exits, or closes its output, is `driver-exited` and the message names its exit status; a driver that answers
  the `quit` and then exits with a status other than 0 is `driver-exited` as well: its failure must not pass as a finished
  run; a driver that does not end 120 s after `quit` is killed with a note;
* a watchdog thread ends a run that passes `run_deadline_seconds = 86400` with the typed reason `run-deadline` and exit status 2
  (killing the drivers it started);
* every `heartbeat_seconds = 600` the run prints on standard error its elapsed time, phase, request in flight and slowest request;
* the harness removes `PYTHONUNBUFFERED` from the drivers' environment, so that a driver that does not flush its replies fails the
  same way wherever the harness runs (under the runner no Python is unbuffered); the three drivers flush;
* `bm25_driver.py` keeps its code (a test compares it) and only its documentation changes; the reference and strong drivers
  and both baseline directories are untouched (their digests, `baseline_*_sha256`, are the ones of Q003-v2).

Every **typed** stop (the nine reasons of the bridge and the watchdog: `spawn-failed`, `write-failed`, `request-timeout`,
`driver-exited`, `driver-refused`, `bad-reply`, `bad-request`, `reader-stopped`, and `run-deadline`) is exit status 2 and the line
`postgres-hybrid: harness stop [<reason>]: <detail>` on standard error. **Untyped harness errors also exit 2, without the tag**
(the line is then `postgres-hybrid: <message>`): a malformed command line or flag; a workload that cannot be generated; an input
file (pool, vectors) that cannot be read or whose digest is not the pinned one; a missing strong interpreter or models root; the
reference driver pinning another embedding model; a driver whose reranker state is not the mode's; a driver index (reference,
BM25, strong) whose `size` disagrees with the live set; a strong answer whose reranked flag is not the mode's; a resume file that
is inconsistent with the run; the negative control (`no_live_join`) returning no planted row or missing the planted row of the stale lexical probes; a failure
of the harness's own PostgreSQL session while planting or counting (`die`, exit 2). Only the typed stops are named by a mutation.
Section 6's list of exit-2 causes is extended by these (and "does not answer within an hour" reads "within two hours"); everything else in it stands.
The limits are in `config.toml`'s `[harness_limits]` (a drift test compares them with the compiled constants) and are not thresholds:
no rule reads them and no metric depends on them. Their a-priori basis is the preregistered cost model, not a pilot: the cross-
encoder costs about 8 CPU-seconds a query (section 3), so 200 strong queries cost under an hour on an unloaded host; Q003-v2's seed 17
took 8.5 hours on a loaded one, with stretches of about 1,500 s a request. A request limit of two hours and a run limit of a day
therefore separate stuck from slow without stopping a run that is merely loaded. A seed that a limit stops is spent (a listed
experiment runs each seed once), so the limits are generous on purpose and the stop's evidence is the heartbeat in the record.

**Stated limits of the stops.**
* *The run deadline cannot tell a product hang from a harness one.* The watchdog ends any run that passes 86,400 s as
  `run-deadline`, exit 2 ("harness only"). A call into `ptr-pg` or PostgreSQL that never returns (the product has no per-call bound)
  would be reported the same way, as a harness stop and not as a product defect (exit 1); the heartbeat on standard error names the phase and the
  request in flight, so a reader of the log can tell, but the exit status cannot. A seed so stopped is spent and the
  verdict is INVALID, never FALSIFIED: a product hang is not shown as a falsification here.
* *A strong driver that exits nonzero after `quit` spends a finished seed.* The decision, kept as it is: a nonzero exit of a driver
  after it answered `quit` is a harness stop (`driver-exited`, exit 2), also when the seed's measurements are complete. Reason: a
  driver whose interpreter ends badly may have corrupted what it answered, and the harness cannot tell a harmless teardown from a
  real fault; the cost is a 45- to 56-minute seed discarded. Pilot 1002 (below) and the earlier pilots exited cleanly, so this is a
  risk, not an observed event. The harness's output of a seed that ended this way is not published as a result.
* *The recorded thread count is the one the command asked for.* `procedure.strong_arm.strong_threads` is the value the harness passed
  to the driver as `--threads`; it does not read back what torch used. The aggregator therefore checks the record against 1 (any
  other value, or none, makes the seed INVALID), and the pin itself rests on the committed command.

**Unchanged, value for value, with no loosening:** the hypothesis, the falsification text and the metric list of `experiment.toml`;
both baselines and their directory digests; the margins (30 per mille against `rag_reference`, a sanity floor, and 30 per
mille against the exact dense search, the sensitivity control); every coverage and falsifier threshold of `[pinned_thresholds]`;
`criteria.toml` (but for its `experiment` id); the workload generator, the corpus, the queries, the plants, the late writes and the
probes (the digests of seeds 43, 71 and 101 are those of Q003-v2); the power analysis of section 8 (it concerns the paired standard
error at 1,000 queries, which is a property of the pool and not of the seeds); `aggregate.py` (but for its experiment id, one limitation
sentence about the seeds and one new validity rule: a declared seed whose record has `procedure.strong_arm.strong_threads` other than 1 is INVALID); the 19 mutations of Q003-v2, which run in the same way (the nine new ones are the mutation plan below). A test (`scripts/tests/test_q003_v3_successor.py`)
compares the parsed criteria, thresholds, manifest values, plan and aggregator with Q003-v2's.

**Seeds.** The declared seeds are 43, 71, 101, 131 and 173. Seeds 17 and 29, which Q003-v2 ran or reserved, are dropped. Seeds
43, 71 and 101 were generated in Q003-v2 (their workload digests) and never run against PostgreSQL, so they were never seen; 131 and 173 are new
(no experiment of the repository uses them) and their digests were measured here by `tests/pin_workload.py`. No declared seed has
been run against PostgreSQL: the development runs, the mutation runs and the pilots used seed 17 or a pilot seed from 1001 on.
Seed 17 stays the **development seed** (the mutation checker's and the development runs'; never declared, never evidence): its
workload digest keeps Q003-v2's pin (`[development_seed]` of `config.toml`, `DEVELOPMENT_DIGESTS` in `params.rs`, a drift test), because
the first run of the whole mutation plan on this harness found `generator-draws-another-workload` surviving: the harness compares
a workload digest only for a seed that has a pin, and with 17 dropped from the declared seeds the mutated generator at seed 17 was
compared with nothing. The pin of the development seed restores the check without running a declared seed.

**Mutation plan.** Q003-v2's 19 mutations, unchanged and in the same order, and nine new ones. Eight plant the failure of a driver or of
the clock and expect a typed stop instead of a counter (`expect_stop`; `scripts/mutation_check.py` kills one only when the harness exits 2
with `harness stop [<reason>]`): a BM25 driver that does not flush its replies (`request-timeout`), one that stalls on a search
(`request-timeout`), one that crashes on a search (`driver-exited`), one that exits with status 3 after `quit` (`driver-exited`), a run
that outlasts a shortened budget (`run-deadline`), a driver that stops reading after a reset while the loader sends the whole corpus
as one request over 64 KiB, which blocks the write that Q003-v2 made on the calling thread (`request-timeout`), one that refuses an upsert
(`driver-refused`) and one that prints a line that is not a reply (`bad-reply`). The ninth (`strong-arm-records-the-default-thread-count`)
is killed by a validity rule, not a counter or a stop (`expect_invalid`): the checker feeds the mutated harness's result to
`aggregate.judge_seed` and the mutation is killed when the aggregator's reason "the strong arm recorded strong_threads 4" appears (a
run records the thread count only when the strong arm ran, which a mutation-scale run does not, so the mutation records it
unconditionally; this shows the rule fires on a recorded 4, not that the real arm would record one). `bad-request` and the
`write-failed`, `spawn-failed` and `reader-stopped` stops have unit tests (tests of the bridge, `q003/tests.rs`) and no mutation. The request limit
and the run limit are shortened in `params.rs` by the mutations that need it so that the stop shows in seconds; the limits as pinned are
compared with `config.toml` by a unit test. `aggregate.py` accepts COMPLETED only with 28 of 28 killed.

**Development and pilot runs after the supersession.** None is a declared seed, none steers a rule, threshold, margin or seed.
* **Pilot 1002, the pinned scale, the full strong arm** (2026-10-11, the release binary built from this tree, `--strong-threads 1`; a
  dry run, since 1002 is not a declared seed; committed as `pilots/pilot-1002.json` and `pilots/pilot-1002.err`, with
  `pilots/README.md`: "pilots, not evidence"): exit 0 in **2,426 s** (40 minutes) on a host with a load average of 7 to 9 on four logical
  CPUs, of which **2,198 s were the strong arm** (200 queries, about 11 s each, slowest request 16 s). All 16 coverage and 11 falsifier rules held (0 stale hits,
  0 planted rows returned, 0 late writes accepted, 0 residual rows, 0 projection mismatches, 50 of 50 lexical probes); fused
  recall@10 was 978 per mille against the reference's 921 and the exact dense search's 981 (deficits 0 and 3). Two earlier runs of
  the same seed and arguments, on earlier revisions of the harness's stop handling, took 3,365 s (load 11 to 14) and 2,618 s (load 8 to 11) and gave the same recall; across the
  last two runs the only differing record fields were the wall-clock ones (`timing_ns.*`, `latency_ns.*`, `index_build_ns`). The numbers are
  descriptive and steered nothing: the margins, thresholds and rules are the ones of Q003-v2.
* **The same pinned scale at the default four threads** (seed 1001, started 12:21 UTC, still running after four hours when this was
  written, 15,000 s of heartbeats): the strong search took 57 to 877 s a request and its driver had used more than 11,000 CPU-seconds
  for about 30 strong queries (the planning model: 8 CPU-seconds each); the driver's involuntary context switches ran to about 17 million in 3 hours.
  That is the cross-encoder's four torch threads spinning against each other on an oversubscribed host (a CPU-time inflation, not a wait),
  the 100- to 180-fold slowdown that seed 17 of Q003-v2 showed, and the likeliest reason that seed 29 was still computing after 13 hours.
  It is the reason for the advice in the next paragraph. This run would end at the whole-run budget (`run-deadline`, exit 2): it
  is a dry run and its end is evidence of nothing but the watchdog.
* Seed 1001 at 200 queries without the strong arm (`scale: custom`): 232 s; it fails the pinned-scale coverage floors by design.

**Cost, for the person who runs it.** The strong arm costs about 8 CPU-seconds a query and a seed asks it 200 queries: run the five
seeds one after another on a quiet host (nothing else heavy on its four cores), with the heartbeat watched. The confirmatory command passes `--strong-threads 1` (see the next paragraph).
A seed that a limit stops is spent.

**Thread count of the strong arm (pre-freeze procedure change, round 2 of the successor; not a threshold change).** Confirmatory
runs pin the strong arm's driver to ONE torch thread: the runner's fixed command, and the preflight, carry `--strong-threads 1`
(`experiment.toml`), and the run record states it (`procedure.strong_arm.strong_threads`, 1). Reason: oversubscription measured on the
shared four-core host (Q003-v2's reading as given by the pilots above): at the driver's default of four threads one strong query took
57 to 877 s; at one thread under the same load about 15 s (slowest request 19 s). Four spinning torch threads on four cores that other
work also uses cost 15 to 60 times more than one. The thread count changes only the **fp32 summation order** of the cross-encoder
(scores are not bit-reproducible across machines or thread counts anyway, as sections 3 and 8 already say); it changes neither the
reranker's depth, its precision (fp32), its input length nor any rule, margin or threshold, and the strong arm is never gated.
**Wall-time estimate per seed at one thread:** about 3,400 s (56 minutes) at the pinned scale with the full strong arm, an upper figure from the first pilot 1002
(3,365 s, of which 3,066 s the strong arm, on a host at load 11 to 14; the committed `pilots/pilot-1002.json` took 2,426 s, 2,198 s the strong arm, at load 7 to 9 and records `strong_threads: 1`; an unloaded host should be faster). Five seeds one after another
take about five hours. This is an estimate, not a bound: the limits (7,200 s a request, 86,400 s a run) stay as pinned.

### Freeze procedure

The head freezes; the author does not. In this order, all from a clean tree at the commit to be frozen:

1. `python3 freeze_sign.py --id Q003-v3 --dir experiments/retrieval/Q003-v3-postgres-hybrid --dry-run` first (it checks the
   placeholders on parsed TOML values, never on comments, and writes nothing); stop on failure. There is nothing to sign by hand:
   the two `must-be-pinned-before-prepared` digests of `config.toml` (this text and `criteria.toml`) are pinned by the tool from the
   files as committed, then the manifest's `preregistration_sha256` and `preregistration_rules_sha256` and the statuses
   (`prepared` in `experiment.toml` and in `experiments/registry.toml`).
2. Before signing: `scripts/check_research_gates.py` clean on a clean tree; `tests/pin_workload.py --check` current; the unit
   tests (`cargo test -p ptr-bench --features postgres-experiments --locked q003`, with `-- --include-ignored` against a server),
   `python3 -m unittest scripts.tests.test_q003_v3_successor scripts.tests.test_q003_v2_successor scripts.tests.test_q003_aggregate
   scripts.tests.test_mutation_check scripts.tests.test_research_gates`; `scripts/mutation_check.py Q003-v3 --check`.
3. Sign (`freeze_sign.py` without `--dry-run`), refresh the generated component documents, run `run_experiment.py validate` and the
   gate again, commit as the freeze commit and tag it `q003v3-prereg`. The mutation evidence (`scripts/mutation_check.py Q003-v3`,
   28 of 28) is made at the committed tree and committed before the first seed; `run_experiment.py preflight Q003-v3` must pass
   (PostgreSQL 16+ with pgvector 0.8+, the strong interpreter and the models); then `prepare`, and the seeds one after another,
   each record committed before the next, stopping at the first exit 2.
4. After the seeds, `aggregate.py` publishes the verdict; COMPLETED, FALSIFIED, NOT_SHOWN and INVALID are exhaustive and INVALID takes
   precedence.
