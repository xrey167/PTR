# External Integrations

Integrations document adapters and candidate technologies. They never define core PTR semantics.

For each integration document:
1. identify the PTR contract it implements;
2. list the exact features used;
3. record licensing/deployment constraints;
4. describe failure behavior;
5. link its component evaluation;
6. keep provider-native types inside the adapter boundary.


## Editor/developer integrations

Editor integrations such as [rust-mode, rustic and cargo-mode](editor/README.md) are contributor tooling only. They are evaluated for command parity and ergonomics and never define PTR runtime semantics.
