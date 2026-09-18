# PTR SDKs

SDKs are thin client contracts for PTR's public control plane. They must not duplicate runtime semantics such as capability, generation or verification logic on the client.

## TypeScript

[`sdk/typescript`](typescript/) contains the first dependency-free HTTP client contract. It is intentionally marked contract-only until `ptr-server` implements and freezes the matching `/v1` API.
