# PTR SDKs

SDKs are thin client contracts for PTR's public control plane. They must not duplicate runtime semantics such as capability, generation or verification logic on the client.

## TypeScript

[`sdk/typescript`](typescript/) contains the first dependency-free HTTP client contract. `ptr-server` now implements both routes the client calls (`/health` and `/v1/requests`), so it is no longer contract-only. The `/v1` API is not frozen: streaming, auth/session policy and a real configured model backend remain open.
