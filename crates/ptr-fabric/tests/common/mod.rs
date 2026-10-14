//! A well-formed handle around a locator, shared by the carrier test files.

use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
use ptr_protocol::{CarrierLocator, TensorHandle};
use ptr_types::TypeId;

fn element(name: &str) -> ChainElement {
    ChainElement {
        identity: name.to_owned(),
        generation: 1,
        digest: [1; 32],
        fence_epoch: 0,
    }
}

/// A handle that commits to `byte_len` bytes with `content_digest` and names the carrier of
/// `locator`.
pub fn handle(locator: CarrierLocator, byte_len: u64, content_digest: [u8; 32]) -> TensorHandle {
    TensorHandle {
        carrier: locator.carrier(),
        locator,
        dtype: BoundaryDtype::F32,
        shape: vec![byte_len / 4],
        byte_len,
        upstream_chain: UpstreamChain {
            stage: element("stage"),
            base_model: element("model"),
            adapters: vec![],
            experts: vec![],
        },
        placement_epoch: 1,
        signed_absolute_deadline_ms: 10_000,
        handle_nonce: 1,
        parent_operation_id: 1,
        consumer: "consumer".to_owned(),
        payload_type: TypeId::from("ptr.split.activation.v1"),
        content_digest,
        signature: [0; 64],
    }
}
