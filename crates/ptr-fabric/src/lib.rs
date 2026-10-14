//! Carriers of the Pod data plane: how the bytes behind a `TensorHandle` travel.
//!
//! The handle (in `ptr-protocol`) says where a tensor is; a carrier here moves or maps the
//! bytes and checks them against the handle's commitment before they are used. Two carriers
//! exist: `Bytes` (inline in the handle, every platform) and `HostShm` (a sealed `memfd`,
//! Linux only, feature `shm-backend`).

mod bytes;
mod error;
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
mod shm;

pub use bytes::{export_bytes, import_bytes};
pub use error::{FabricError, Seal};
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
pub use shm::{host_page_size, ShmExpectation, ShmExport, ShmMapping};
