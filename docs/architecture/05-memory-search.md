# Memory and Search

Memory is split into semantic, episodic, procedural and epistemic classes. `SemanticCapsule` is the durable validated semantic representation. Search backends are derived projections.

Planned retrieval roles: Tantivy for lexical/exact text, Zvec for local embedded semantic search, cuVS for GPU hot search and clustering, LanceDB for large multimodal corpora, Havenask for distributed search, and GritQL for structural code search. Each role is benchmarked under `evaluations/components/` before lock-in.

Evidence promotion is explicit: `SearchCandidate -> PossibleEvidence -> Observed -> VerifiedKnown`.
