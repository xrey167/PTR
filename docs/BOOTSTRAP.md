# Bootstrap Plan

The repository intentionally separates **architecture contracts** from external implementation choices. The first implementation wave should stay standalone and dependency-light:

1. compile/test all internal crates;
2. make `ptr-semdb` exact and benchmark incremental invalidation;
3. implement bounded isolate scheduler + replay;
4. implement local causal ledger and lifecycle tests;
5. wire one external inference backend;
6. establish baselines before PTR-Core training;
7. add one integration at a time only through the corresponding trait;
8. run component evaluation before declaring any external backend the default.

Cluster consensus, GPU indexes and large distributed systems come after local lifecycle correctness.
