# ptr-fabric — carriers of the Pod data plane

> **Role:** Moves or maps the bytes behind a `TensorHandle` and checks them against the handle's commitment before use. Two carriers exist: `Bytes` (inline, every platform) and `HostShm` (a dedicated sealed `memfd` per export, Linux only, feature `shm-backend`).
> **Maturity:** prototype; the carriers are proven in `tests/bytes_carrier.rs` and `tests/host_shm.rs` without a worker process. Passing the descriptor between processes is not built.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-10-14  
**Code footprint:** 4 Rust source files · 357 nonblank source lines · 3 integration-test files · 23 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- Bytes carrier (all platforms): export_bytes turns a tensor into CarrierLocator::Bytes and refuses an empty tensor and one above MAX_INLINE_BYTES with a typed error; import_bytes checks the carrier and the locator variant, the length against the handle's byte_len and the SHA-256 against its content_digest before it returns the bytes
- HostShm carrier (feature shm-backend, Linux only), producer side: ShmExport::create writes the tensor once into a dedicated memfd (MFD_CLOEXEC | MFD_ALLOW_SEALING), checks that the file size equals byte_len, applies F_SEAL_WRITE, F_SEAL_SHRINK, F_SEAL_GROW and then F_SEAL_SEAL, refuses an empty tensor and one above MAX_HANDLE_BYTES, and exposes the locator (offset 0, caller-supplied fd_slot), the digest and footprint_bytes (the memfd rounded up to whole pages, for the runtime's handle admission)
- HostShm carrier, consumer side: ShmMapping::map checks offset 0, all four seals and the file size before mapping, maps exactly byte_len bytes with PROT_READ and MAP_SHARED, verifies the SHA-256 of the mapped bytes before as_bytes can be called, and unmaps on drop; every failed check is a typed FabricError and nothing stays mapped
- One exhaustive FabricError enum (empty tensor, too large, wrong carrier, wrong locator, length mismatch, digest mismatch, non-zero offset, missing seal, not sealable (a descriptor that is not a memfd, EINVAL from the seal query), length not addressable, OS error); the only unsafe code is mmap, munmap and the slice over the mapping, each with a SAFETY comment
- The shm-backend feature compiles to nothing off Linux and rustix is a Linux-only target dependency, so neither the default build nor a build with the feature pulls or compiles anything on Windows and macOS; import_bytes refuses an inline payload above MAX_INLINE_BYTES before hashing it

### Missing for the target architecture

- Passing the descriptor between processes (SCM_RIGHTS bound to the handle nonce) and every worker-process wiring: the carriers are exercised in one process with a dup of the descriptor (package C5)
- CudaIpc, DevicePeer and PinnedHost carriers (package C7); the NvmeFile carrier is not planned here
- HostShm on Windows or macOS: Windows and macOS have the Bytes carrier only
- A host with vm.memfd_noexec=2 refuses memfd_create without MFD_NOEXEC_SEAL; that surfaces as FabricError::Io and is not handled yet
- ShmMapping is not Send or Sync; a hand-over between threads needs a reviewed unsafe impl or a different owner type
- Nothing calls the carriers from ptr-runtime's handle admission yet; footprint_bytes is the value a caller passes to HandleAdmissionRequest::footprint_bytes

### Next milestones

- SCM_RIGHTS passing of the sealed descriptor and a worker that maps it (C5)
- Register the data-plane experiment (E009) that measures the carriers against the Bytes baseline

### Linked experiments

- None recorded.

### Technology evaluations

- None recorded.

### Decision records

- [ADR-0004-backend-independence.md](../../research/decisions/ADR-0004-backend-independence.md)

### Current automated checks

- tests/bytes_carrier.rs: roundtrip, the bound accepted and one byte above it refused, empty tensor, wrong digest, tampered inline payload, wrong locator variant, wrong carrier, length mismatch in both directions
- tests/host_shm.rs (shm-backend, Linux): roundtrip with exact size and offset 0, all four seals present on the export, write and grow and shrink refused with EPERM after export, a dup of the descriptor imports the same bytes, a hand-built memfd missing each seal in turn refused, wrong file size, non-zero offset, wrong digest with no mapping left, mapping length exactly byte_len, empty tensor on both sides, page-rounded footprint, 3000 map and drop cycles leave no mapping, a regular-file descriptor refused as NotSealable with nothing mapped

<!-- PTR:STATUS:END -->

## Diagram

Source: [`docs/diagrams/components/ptr-fabric.mmd`](../../docs/diagrams/components/ptr-fabric.mmd).

## Carriers

- `Bytes`: `export_bytes` refuses an empty tensor and one above `MAX_INLINE_BYTES`; `import_bytes` checks the carrier and locator variant, the length and the SHA-256 before it returns the bytes.
- `HostShm`: `ShmExport::create` writes the tensor once into a fresh memfd and applies `F_SEAL_WRITE`, `F_SEAL_SHRINK`, `F_SEAL_GROW` and `F_SEAL_SEAL`, in that order. `ShmMapping::map` checks the seals, offset 0 and the file size, maps exactly `byte_len` bytes read-only and verifies the digest before it hands out the slice (doc 42 rule 3a).

## Tests

```bash
cargo test -p ptr-fabric --locked
cargo test -p ptr-fabric --features shm-backend --locked
```
