# Dataset Workspace

Large data is intentionally separated from source code. The registry records provenance, license, schema, split strategy and hashes. Never commit private/customer data to the repository.

Required dataset families:

1. typed behavior / Rust-like state transitions;
2. PodWire native protocol;
3. raw ↔ typed semantic alignment;
4. semantic corruption and disagreement examples;
5. operator-routing tasks;
6. calibrated probabilistic tasks;
7. unseen Pod/capability composition;
8. long-horizon revision/generation lifecycle;
9. verifier/critique/repair trajectories;
10. retrieval evidence promotion;
11. strong RAG/GraphRAG baselines;
12. action-boundary and permission tasks.

Splits must prevent nominal-label leakage and include true compositional OOD cases.
