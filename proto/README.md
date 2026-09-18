# PTR Wire Schemas

These schemas define versioned network representations. They are **not** the domain model.

- `podwire.proto` — Pod/control frames
- `raft_transport.proto` — consensus transport envelope
- `model_events.proto` — remote model-event stream
- `events.proto` — distributed event envelope

Generated Prost types are converted into validated domain values in `ptr-protocol`. Semantic compatibility has its own version policy in addition to wire compatibility.
