# PTR Iroh Current Adapter

This isolated package evaluates **Iroh 1.2.0** without forcing PTR's Rust 1.85 core workspace to adopt Iroh's Rust 1.91 MSRV.

The adapter currently targets:
- PTR ALPN usage;
- direct local QUIC request/response;
- peer identity derived from Iroh's authenticated endpoint key;
- a separate Cargo.lock and separate security/toolchain lifecycle.

It is intentionally outside the root Cargo workspace. If Iroh becomes the selected production transport, PTR can either raise the core MSRV later or keep the network adapter as an isolated package/process boundary.
