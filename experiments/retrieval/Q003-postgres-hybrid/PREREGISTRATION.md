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
`cargo run --release --locked -p ptr-bench --features postgres-experiments -- postgres-hybrid <seed>`
with `PTR_PG_EXPERIMENT_DSN` naming a loopback PostgreSQL 16+ server with pgvector 0.8+, `PTR_Q003_STRONG_PYTHON` an interpreter
with the `research/embeddings` environment (torch, transformers) and `PTR_Q003_MODELS_ROOT` the directory holding the pinned
`bge-reranker-v2-m3/` (`research/embeddings/fetch_models.py`). None of the three is stored in a result. Optional:
`PTR_Q003_STRONG_RESUME_DIR` (or `--strong-resume-dir`), where the strong arm keeps each query's answer so that a failed run
resumes (section 3), and `--strong-threads N`. The confirmatory strong runs are to be made one seed after another on a quiet
machine with `--strong-threads` set to the physical cores (the reranker's depth, precision and length are not to change); the
measured cost is about 8 CPU-seconds a query, 16 to 18 s wall on a moderately loaded box and 46 s at a load average of 17 to 20.

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
