use ptr_types::NodeId;

pub const ALPN_RAFT: &[u8] = b"ptr-raft/1";
pub const ALPN_PODWIRE: &[u8] = b"ptr-podwire/1";
pub const ALPN_MODEL: &[u8] = b"ptr-model/1";
pub const ALPN_BLOB: &[u8] = b"ptr-blob/1";
pub const ALPN_EVENTS: &[u8] = b"ptr-events/1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeIdentity { pub id: NodeId, pub public_key: String }

pub trait Transport { fn send(&self, peer: &NodeIdentity, alpn: &[u8], payload: &[u8]) -> Result<(), String>; }
