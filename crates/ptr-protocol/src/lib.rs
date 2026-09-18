//! Semantic PodWire domain frames. Network encodings (prost/rkyv/JSON) are adapters around these types.

pub mod generated;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolError {
    MissingSessionId,
    MissingCallId,
    MissingCapability,
    MissingPayload,
    MissingTypeId,
}

impl TryFrom<generated::podwire::PodCall> for CallFrame {
    type Error = ProtocolError;

    fn try_from(value: generated::podwire::PodCall) -> Result<Self, Self::Error> {
        if value.session_id.trim().is_empty() {
            return Err(ProtocolError::MissingSessionId);
        }
        if value.call_id.trim().is_empty() {
            return Err(ProtocolError::MissingCallId);
        }
        if value.capability.trim().is_empty() {
            return Err(ProtocolError::MissingCapability);
        }
        let payload = value.payload.ok_or(ProtocolError::MissingPayload)?;
        if payload.type_id.trim().is_empty() {
            return Err(ProtocolError::MissingTypeId);
        }

        Ok(Self {
            session_id: value.session_id,
            call_id: value.call_id,
            capability: CapabilityId(value.capability),
            generation: value.generation.map(Generation),
            revision: Revision(value.revision),
            payload: TypedPayload {
                type_id: TypeId(payload.type_id),
                bytes: payload.payload,
            },
        })
    }
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
