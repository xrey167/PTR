# Prior-Art Matrix

Maintain one row per relevant system/paper and map it against PTR mechanisms: typed latent state, uncertainty representation, recurrent latent reasoning, operator routing, editable lifecycle state, exact revocation, incremental semantics, external cognitive modules and verifier feedback.

Do not infer novelty from naming differences.

## Agentic substrate (doc 35)

Rows for the mechanisms of [35 — Agentic substrate](../../docs/architecture/35-agentic-substrate.md). "PTR difference" names what is claimed to differ; every such claim is provisional until its experiment runs.

| System / paper | Mechanism | What PTR takes | PTR difference (claim) | Experiment |
|---|---|---|---|---|
| Kung and Robinson, optimistic concurrency control (ACM TODS 1981); Cahill, Röhm and Fekete, serializable snapshot isolation (SIGMOD 2008); Ports and Grittner, SSI in PostgreSQL (VLDB 2012) | Read-set validation, rw-antidependency detection | Certification of declared reads before commit | Certification over canonical value digests shared with neural-state admission, range digests for phantoms and lifecycle generations; merge only through a verified delta | S003 |
| Database branching (Dolt, Neon) | Copy-on-write branches of a whole database | Branch-per-agent workflow | A branch holds no state, only declared dependencies and typed operations; it can never be merged by a storage-level merge | S003 |
| Angelopoulos, Bates, Fisch, Lei and Schuster, conformal risk control (arXiv:2208.02814); Angelopoulos, Bates, Candès, Jordan and Lei, Learn then Test (arXiv:2110.01052); Jung, Brahman and Choi, Trust or Escalate (arXiv:2407.18370) | Finite-sample risk-controlling thresholds | Threshold selection on a fixed grid | Eligibility is decided by verification first; calibration uses a uniform audit slice because outcomes of auto-proposals are otherwise censored | F003 |
| Dudík, Langford and Li, doubly robust policy evaluation (arXiv:1103.4601); Swaminathan and Joachims, self-normalised estimator (NeurIPS 2015) | Off-policy evaluation from logged propensities | IPS, SNIPS and doubly robust estimates | Positivity is checked; a threshold below anything logged is refused rather than extrapolated | F003 |
| Schlag, Irie and Schmidhuber, linear transformers as fast weight programmers (arXiv:2102.11174); Yang et al., DeltaNet (arXiv:2406.06484); Yang, Kautz and Hatamizadeh, Gated DeltaNet (arXiv:2412.06464); Kimi Team, Kimi Linear / KDA (arXiv:2510.26692); Behrouz, Zhong and Mirrokni, Titans (arXiv:2501.00663) | Delta-rule and gated fast-weight memories | The gated delta rule in its KDA form as a working-memory update | Used outside the model as a derived memory with write attribution, admission at read and bit-exact revocation by refold | M008, L003 |
| Bourtoule et al., SISA machine unlearning (arXiv:1912.03817) | Exact unlearning by retraining affected shards | Exactness as the target, checkpointed retraining | Exact revocation of a deterministic fold from the last checkpoint before the revoked write; stated as revocation, not erasure | L003 |
| Packer et al., MemGPT/Letta (arXiv:2310.08560); Chhikara et al., Mem0 (arXiv:2504.19413); Rasmussen et al., Zep/Graphiti (arXiv:2501.13956) | Agent memory layers over LLM-extracted facts | Benchmark targets and baselines | Memory is a lifecycle-governed projection: revoked or superseded facts cannot be recalled, and Unknown is a valid answer | E005 |
| Wu et al., LongMemEval (arXiv:2410.10813); Maharana et al., LoCoMo (arXiv:2402.17753) | Long-horizon conversational memory benchmarks | Evaluation suites | Re-scored per category with equal token budgets; third-party audits report answer-key errors, so results are also reported on an audited subset | E005 |
| Wang et al., O-LoRA (Findings of EMNLP 2023); Liang and Li, InfLoRA (CVPR 2024) | Orthogonality constraints between task adapters | Subspace view of interference | Interference is measured per layer as principal-angle overlap against its chance level rather than imposed as a null-space constraint; the naive "null space of the sum" is not used | R004 |
| Ilharco et al., task arithmetic (arXiv:2212.04089); Yadav et al., TIES-Merging (arXiv:2306.01708) | Merging fine-tuned deltas | TIES merge of full updates | Consolidation is a gated lineage event with a data manifest, so erasure propagation still names every source | R004 |
| FSRS-4.5 (open-spaced-repetition); Efraimidis and Spirakis, weighted random sampling (IPL 2006) | Power-law forgetting; weighted sampling without replacement | Forgetting model and Gumbel-top-k draws | Forgetting is fitted on the training clock from probe losses; strata are sampled without replacement and held-out samples are excluded by type | R004 |
| Dawid and Skene (JRSS C 1979); Ratner et al., Snorkel (arXiv:1711.10160) | Label models over noisy labeling functions | EM confusion-matrix model | Verifier-backed functions only veto and are never outvoted; Unknown and Disputed are outcomes; only uniform gold estimates accuracy | F002 |
| Cormack, Clarke and Büttcher, reciprocal rank fusion (SIGIR 2009); Bruch, Gai and Ingber, fusion functions for hybrid retrieval (arXiv:2210.11934) | Rank and score fusion | Weighted RRF and convex fusion | Fusion keyed by capsule and generation, joined to the lifecycle catalog of the same snapshot | Q003 |
| pgvector; VectorChord; ParadeDB pg_search; Tiger Data pg_textsearch | In-database vector and BM25 search | halfvec HNSW, iterative scans, BM25 candidates | Derived rows exist only for live generations, enforced by row-lock ordering with the projector | Q003 |

## KV-cache reuse (doc 36)

Rows for the mechanisms of [36 — KV reuse](../../docs/architecture/36-kv-reuse.md). "PTR difference" names what is claimed to differ; every such claim is provisional until its experiment runs. "Suffix Cache Reuse" is not a published method name and is not used as one.

| System / paper | Mechanism | What PTR takes | PTR difference (claim) | Experiment |
|---|---|---|---|---|
| Kwon et al., PagedAttention / vLLM (arXiv:2309.06180) | Paged KV blocks with copy-on-write sharing | 16-token pages | Pages are bound to principal, manifest and generation; revocation is refused at admission | R005 |
| Zheng et al., SGLang RadixAttention (arXiv:2312.07104); vLLM automatic prefix caching (documentation: parent-chained block hash, `cache_salt`) | Token-keyed radix prefix cache; chained block hashes with an optional salt | Token-keyed radix prefix catalog | The salt comes from the currently admitted binding, and revocation invalidates the affected subtree | R005, E006 |
| Greenewald et al., Activated LoRA (arXiv:2504.12397) | Adapter weights apply only from the invocation tokens on, so all states before them equal the base model (Prop. 1) | Base-aligned prefix sharing | Per-layer chain key and adapter generations | R009 |
| PIE (arXiv:2407.03157); Leyline (arXiv:2606.01065) | Positional re-rotation of cached keys; FORGET with re-prefill | Re-rotation and re-prefill | Pre-RoPE storage; an Approximate tag above one layer; span eviction without hidden carry-over | R006, R007 |
| CacheBlend (arXiv:2405.16444); EPIC (arXiv:2410.15332; its 8x is measured against CacheBlend); MEPIC (arXiv:2512.16822) | Position-independent chunk reuse with selective recompute | Chunk reuse as an approximate path | Approximate results never reach effects | M009 |
| ShadowKV (arXiv:2410.21465); TailorKV (arXiv:2505.19586); CacheGen (arXiv:2310.07240) | Lossy KV tiers (low-rank, hybrid, compressed streaming) | Lossy tiers | A derived lossy tier object that is always Approximate | R008 |
| SnapKV (arXiv:2404.14469); Ada-KV (arXiv:2407.11550); K-VEC (arXiv:2606.29563) | Query-dependent KV eviction | Rejected: query-dependent eviction breaks prefix sharing | None | none |
| KV-streams (arXiv:2609.35750); MEMENTO (arXiv:2604.09852); STILL (arXiv:2606.07878); DroidSpeak (arXiv:2411.02820) | Hidden-state or KV carry-over across spans, turns or models | Rejected as authoritative: hidden state carries deleted or revoked content. MEMENTO is used through its text channel only, as a ptr-memory object | None for the rejected paths | R007 |
| MegaScale-Infer (arXiv:2504.02263); Step-3 AFD (arXiv:2507.19427) | Attention-FFN disaggregation: KV lives with attention, expert nodes are KV-stateless | KV stays with attention | Expert pods hold no KV; activation frames carry the upstream chain digest | E007 |
| Thinking Machines, "Defeating Nondeterminism in LLM Inference" (Sep 2025); Soft MoE (arXiv:2308.00951, §2.2) | Batch-invariant kernels; capacity routing is batch-dependent | Batch-invariance as the condition for exactness | Exact requires dropless routing and batch-invariant kernels | R010 |

## Pod-MoE (doc 37)

Rows for the mechanisms of [37 — Pod-MoE](../../docs/architecture/37-pod-moe.md). "PTR difference" names what is claimed to differ; every such claim is provisional until its experiment runs. GSPO's statement that no routing replay is needed is contested by R3 (arXiv:2510.11370).

| System / paper | Mechanism | What PTR takes | PTR difference (claim) | Experiment |
|---|---|---|---|---|
| GRPO (arXiv:2402.03300); GSPO (arXiv:2507.18071); SAPO (arXiv:2511.20347); REINFORCE++ (arXiv:2501.03262); R3 routing replay (arXiv:2510.11370) | RLVR policy-gradient algorithms; routing replay for MoE | RLVR algorithms | Reward comes from the ptr-verifier `VerificationReport`; routing churn is measured per layer | M010 |
| Yue et al. (arXiv:2504.13837); ProRL (arXiv:2505.24864) | RLVR versus distillation compared by pass@k | The comparison protocol | Measured on our own MoE | M015 |
| HSD (arXiv:2606.15576) | Path-conditioned self-distillation | A later arm; no public code | Not yet implemented | M010 (later) |
| PERFT-R (arXiv:2508.02587; PERFT arXiv:2411.08212); MixLoRA (arXiv:2404.15159); Mixture of Routers (arXiv:2503.23362) | Routed parameter-efficient experts | Routed LoRA experts | Each LoRA expert is a versioned, revocable pod with lineage | M011 |
| Aux-loss-free balancing (arXiv:2408.15664); its theory (arXiv:2512.03915, Dec 2025); φ-Balancing (arXiv:2605.15403) | Bias-based load balancing without an auxiliary loss | Bias-based balancing | Applied to the LoRA-expert router | M012 |
| RA-MoE (arXiv:2605.28306, multilingual) | Routing alignment loss | Alignment loss | Adapted to tasks, as an ablation | M011 |
| Li et al., MoE in continual learning theory (arXiv:2406.16437); DynMoE (arXiv:2405.14297); Lifelong-MoE (arXiv:2305.12281); MC-SMoE (arXiv:2310.01334); sparse upcycling (arXiv:2212.05055); expert upcycling (arXiv:2604.19835); BTX (arXiv:2403.07816); EMO (arXiv:2605.13247) | Individual lifecycle operations on experts (add, merge, upcycle, branch-train-mix) | The individual operations | Full gated lifecycle Birth, Merge, Split, Freeze, Reactivate and Prune with a ledger commit; we know of no prior work covering the full lifecycle, and Reactivate is unpublished | M013 |
| AIMER (arXiv:2603.18492) | Weights-only expert ranking | Ranking as a candidate signal | Gate-checked prune with recovery | M014 |
| Self-MoE (arXiv:2406.12034); Mixture-of-Agents (arXiv:2406.04692); Router-R1 (arXiv:2506.09033); RouteLLM (arXiv:2406.18665); BTM (arXiv:2208.03306) | Model-as-expert routing | Request-level experts | Experts are pods; the router is evaluated off-policy | M013, R011 |
| Darwin Gödel Machine (arXiv:2505.22954); Gödel Agent (arXiv:2410.04444); AlphaEvolve (arXiv:2506.13131) | Self-modifying agents (code and scaffolding) | Self-modification as an idea | Self-modification of MoE structure only through proposals, gates and the ledger; the policy cannot alter gates or the verifier | R011 |
| Gloeckle et al., MTP (arXiv:2404.19737); DeepSeek-V3 (arXiv:2412.19437); MTP curriculum (arXiv:2505.22757) | Multi-token prediction | Deferred: MTP hurts small models | None | none |
| OLMoE (arXiv:2409.02060) | Fully open base MoE | Base model for pod-MoE | None | M010–M015 |
