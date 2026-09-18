# Architecture Component Evaluation

Every infrastructure decision is provisional until evaluated. Each component folder contains candidate implementations, criteria and evidence paths. New alternatives can be added at any time without changing PTR domain contracts.

Rules:

1. Correctness and failure semantics are gates, not merely weighted preferences.
2. Benchmark at least one serious alternative before marking a decision `locked`.
3. Record hardware, dataset, version/commit and configuration.
4. Separate prototype convenience from production suitability.
5. Re-open decisions when workload assumptions materially change.
