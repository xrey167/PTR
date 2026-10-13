# pod-routing

PTR contract and candidate comparison for **pod-routing**. Add benchmarks under `evidence/` and update `candidates.toml`; keep the architecture-facing trait stable.

How a request is matched to one of the registry's candidate pods. The registry stays authoritative for eligibility (project, capability, input type); a router only ranks what the registry offers. A learned score is advisory (Invariant 11), abstention is a valid result (Invariant 12) and revoking one pod must not invalidate the others (Invariant 17).

Candidates: the constant default, description-vector cosine, the learned Dragonfly address, and the Symlink-LoRA idea (design only). The routing claim for the learned address is experiment R015 (arms `learned`, `cosine_only`, `constant_score`; metrics `routing_accuracy`, `abstain_rate`, `false_route_rate`).

Design: [40 — Pod linking](../../../docs/architecture/40-pod-linking.md); decision: [ADR-0022](../../../research/decisions/ADR-0022-learned-pod-addresses-advisory.md).
