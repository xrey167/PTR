# Q003-v2-postgres-hybrid

Successor of [Q003](../Q003-postgres-hybrid/README.md), which is `superseded` ([SUPERSEDED.md](../Q003-postgres-hybrid/SUPERSEDED.md)): its seed 17 ended before any product call because the runner passed the harness none of its environment. Q003-v2 has the same harness, criteria, thresholds, workloads, mutation plan and aggregator; the manifest additionally declares `command_environment`, and the runner refuses before a seed is reserved when a declared variable is unset. See section 11 of [PREREGISTRATION.md](PREREGISTRATION.md).

## Hypothesis
Does ptr-pg's generation-joined retrieval keep its lifecycle guarantees and stay within a pinned recall margin of a conventional hybrid reference? `ptr_pg`'s search joins a full-text list and a `halfvec` HNSW list to the lifecycle catalog of their snapshot before each limit and fuses them by weighted reciprocal rank fusion keyed by capsule and generation; `upsert_document` refuses a generation that is not live. It is driven on seeded live-lifecycle corpora of 2000 capsules and 1000 queries per seed: against the harness's own live table no list holds a non-live generation, no planted stale row is returned, the projector leaves no superseded or revoked document behind, no late write of a non-live generation is accepted, a search answers the same way twice, the lexical mode returns the live document for the live-body probes, and the recall@10 of the fused list is at most 30 per mille below both `rag_reference`'s (a sanity floor) and the harness's own exact dense search's (the sensitivity control). **On natural-language queries ptr-pg's lexical list is empty** (`plainto_tsquery` ANDs every term), so its fused list is effectively its dense list; the recall leg compares that list with the BM25-OR + cosine + RRF reference and does not test a working lexical half (a finding about ptr-pg, recorded as a limitation, not a falsifier). The full text is in `experiment.toml`.

```mermaid
flowchart LR
    W[workload generator<br/>seed, pool-q003.jsonl] --> E[events: commit, supersede, revoke]
    E --> P[ptr-pg projector + upsert_document<br/>live documents, frozen MiniLM vectors]
    E --> O[live table oracle<br/>harness code only]
    X[raw SQL plants<br/>stale generations] --> P
    P --> S[ptr-pg search<br/>lexical + vector + fused]
    S --> A[audit every hit<br/>against the oracle]
    O --> A
    O --> R[rag_reference<br/>BM25 + cosine + RRF]
    O --> G[strong_retrieval<br/>BM25 + bge-m3 + RRF + reranker]
    S --> M[recall@10, nDCG@10]
    R --> M
    G --> M
```

## Primary metrics
The 49 metrics of `experiment.toml`: recall@10 and nDCG@10 of the fused list, of the reference and of the strong baseline (on its subsample, beside the product's on the same queries); the lexical, vector, exact-cosine and BM25 lists alone and the lexical hits of the main queries (descriptive); stale-generation hits per list, planted stale rows returned, late writes accepted and wrongly refused, product errors and panics, the projection cross-check; coverage counters; latency (p50, p99) and the index build time (descriptive, never judged).

## Baseline
`research/baselines/rag_reference` (BM25 + MiniLM cosine + RRF) is the gating baseline. `research/baselines/strong_retrieval` (BM25 + bge-m3 dense + RRF + bge-reranker-v2-m3, retrieval only, pinned, CPU) answers a seeded subset of 200 answerable queries and is **reported, never gated**; BM25 alone and the harness's exact dense search are reported. `strong_rag` stays blocked: it needs an answer generator that CPU cannot honestly provide, and it is not what this experiment compares with.

## Falsification
See `experiment.toml` and [criteria.toml](criteria.toml): eleven falsifier rules and sixteen coverage rules, every seed must pass, and COMPLETED needs complete mutation evidence. Any stale-generation hit is a hard failure; recall@10 more than the pinned margin (30 per mille) below `rag_reference` (a sanity floor: it trips only below 912 for a reference at 942) or below the harness's exact dense search (the sensitivity control, which sees a smaller regression) rejects the mechanism as the default.

## Running
```sh
export PTR_PG_EXPERIMENT_DSN="host=127.0.0.1 port=5432 user=postgres"      # PostgreSQL 16+, pgvector 0.8+
export PTR_Q003_STRONG_PYTHON=<python with torch and transformers, research/embeddings>
export PTR_Q003_MODELS_ROOT=<directory holding bge-reranker-v2-m3/, research/embeddings/fetch_models.py>
cargo run --release --locked -p ptr-bench --features postgres-experiments -- postgres-hybrid <seed>
```
The strong arm answers a seeded, stratified subsample of 200 of the 900 answerable queries per seed and is reported, never gated; its cross-encoder costs about 8 CPU-seconds a query (16 to 46 s wall on a loaded box), so run the confirmatory seeds one after another on a quiet machine with `--strong-threads` set to the physical cores, and keep `--strong-resume-dir DIR` (or `PTR_Q003_STRONG_RESUME_DIR`) so that a failure does not lose the hours already spent; `--no-strong` skips the arm and `postgres-hybrid <queries> <seed>` (the mutation checker's form) runs without the strong arm. Both are development forms: `aggregate.py` accepts only the full arm at the pinned scale. `--print-workload-digest` prints the workload digest without a server. The Python drivers (`rag_reference/q003_driver.py`, `strong_retrieval/driver.py`, this directory's `bm25_driver.py`) speak one JSON-lines protocol; the harness's bridge is `bins/ptr-bench/src/experiments/q003/bridge.rs`.

Through the runner, `python3 scripts/run_experiment.py preflight Q003-v2` checks the machine first (server, interpreter, models, the strong, reference and BM25 drivers, one `ptr-pg` query). A failed preflight can leave `q003_0_*` schemas in the database when the product arm itself exits; drop them by hand, they cannot affect a seed.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md). The preregistration is [PREREGISTRATION.md](PREREGISTRATION.md).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Status of this directory
Q003-v2 is `planned` (until its freeze): the harness (`bins/ptr-bench/src/experiments/q003/`, subcommand `postgres-hybrid <seed>`, built with the `postgres-experiments` feature), the preregistration, `criteria.toml`, `aggregate.py` and the mutation plan ([tests/mutations.toml](tests/mutations.toml)) are in place and `config.toml`'s `[preregistration]` table is filled but for the two digests of the frozen text and criteria (`must-be-pinned-before-prepared`, set at the freeze with the entrypoint); no seed has been run for the record (seed 17 was run end to end in a dry run before the freeze, its outcome seen; the other four were only generated). `[pinned_thresholds]` holds the choices the harness compiles in (a drift test compares them).

Stated limits: a synthetic corpus of templated facts and frozen sentence vectors; the main queries are natural-language questions that `plainto_tsquery` ANDs term by term, so the product's lexical mode answers none of them and its fused list is its vector list (the lexical mode and its lifecycle join are exercised by probe queries) and the recall comparison is of that list with a hybrid, where on templated text the reference's fusion lowers it, so a product above the reference is not evidence of hybrid quality; stale rows are planted by raw SQL; the baselines are rebuilt from the live table and are not under test; the cross-encoder is fp32 on shared CPU and not bit-reproducible across machines; the pilot and confirmatory seeds share one 2000-cue pool; latency and index build time are descriptive.
