//! Semantic PodWire domain frames. Network encodings (prost/rkyv/JSON) are adapters around these types.

use ptr_types::{CapabilityId, Generation, Revision, TypeId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WireVersion {
    pub major: u16,
    pub minor: u16,
}

pub const PODWIRE_V1: WireVersion = WireVersion { major: 1, minor: 0 };

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypedPayload {
    pub type_id: TypeId,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallFrame {
    pub session_id: String,
    pub call_id: String,
    pub capability: CapabilityId,
    pub generation: Option<Generation>,
    pub revision: Revision,
    pub payload: TypedPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Frame {
    Call(CallFrame),
    Return {
        call_id: String,
        payload: TypedPayload,
    },
    Error {
        call_id: String,
        code: String,
        message: String,
    },
    Event {
        topic: String,
        payload: TypedPayload,
    },
    Revoke {
        subject: String,
        generation: Generation,
    },
}

impl Frame {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Call(_) => "CALL",
            Self::Return { .. } => "RET",
            Self::Error { .. } => "ERR",
            Self::Event { .. } => "EVT",
            Self::Revoke { .. } => "REVOKE",
        }
    }
}
