# Decision Process

A component moves from `open` to `preferred` only after gate tests pass and evidence compares at least two credible candidates when available. `preferred` is still reversible. `locked` is reserved for contracts whose replacement would invalidate serialized state or wire compatibility, and should be rare.

When a new project appears:

1. map it to an existing PTR contract;
2. if no contract fits, justify why a new architectural responsibility is genuinely needed;
3. add it as a candidate, not as a new hard dependency;
4. define a benchmark that could show it is worse;
5. record exact version/commit, hardware and workload;
6. update the decision with evidence.
