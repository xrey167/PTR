# Candidate Novelty Claims — NOT YET PROVEN

1. A dual raw + probabilistically typed latent workspace can improve semantic fidelity and controllability without forcing all reasoning into text.
2. Explicit epistemic slot types plus typed attention may improve calibration and hard-constraint retention.
3. Reasoning-operator routing can learn when non-language computation should replace token-heavy reasoning.
4. An incremental semantic DB can reduce long-project context cost while preserving lifecycle-correct state.
5. A typed ModelEvent interface may support tighter interleaving of neural reasoning and external cognitive Pods than conventional text tool calls.
6. PTR-Diff may reduce sequential depth for structured reasoning/action generation.
7. A relational projection verified record by record against ledger anchors can host an agent platform's state, search and working memory without becoming a second authority (L004 passed on PostgreSQL 18; Q003 pending).
8. Agent branches certified against value, range and lifecycle digests and merged through verified deltas let agents work concurrently without lost updates (S003), and a verifier-bounded arbiter with a uniform calibration slice keeps the harm rate of auto-proposals within a finite-sample bound (F003).
9. An associative fast-weight memory can be made exactly revocable by attributing every write to an input generation and refolding deterministically (L003 passed on PostgreSQL 18) while still improving update-aware recall (M008).
10. Gated adapter lineage with subspace-interference measurement, forgetting-driven replay and consolidation forgets less than a naive adapter chain (R004); verifier-precedence weak supervision is better calibrated than majority vote (F002); and the combined stack improves long-horizon agent memory benchmarks at equal budget (E005).
11. Experts managed as pods with a ledger-committed, gate-checked lifecycle (Birth, Merge, Split, Freeze, Prune) keep forgetting at or below the ForgettingGate bounds while matching a static MoE at equal compute (M013, R011). Unproven.
12. Re-gating a retired expert when its domain returns recovers performance faster than training a new expert (M013). Unproven.
13. Exposing routing and interference observations to the model as context improves structural proposals over a heuristic baseline, without any self-promotion path. The improvement is tested by the M013 context-ablation arm (full-context policy versus the same policy with routing and interference context removed versus a fixed threshold heuristic; gate pass rate and net quality change per accepted proposal; falsified if the full-context policy does not beat both on gate pass rate by a pre-registered margin in at least 3 seeds), and the absence of a self-promotion path by R011. Unproven.
14. Token-keyed, salt-bound prefix reuse with per-layer chain keys never serves revoked adapter or expert generations while remaining bit-identical to full prefill (R005, R009, R010). Unproven.

Each claim remains provisional until benchmarked and compared with relevant prior work.
