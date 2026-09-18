# Incremental Semantic Database

The semantic DB is modeled after an incremental compiler: external observations are ground inputs; entities, constraints, relations, intent, epistemics and reasoning requirements are derived queries. Dependency tracking allows precise invalidation. Reasoning operates on immutable snapshots. If a new revision arrives, stale work can be cancelled.

Revisions identify the input world seen by a computation. Generations identify lifecycle versions of individual semantic objects. They are never interchangeable.
