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

/// A Pod call, once its required fields are known to be present.
///
/// It carries no session identifier. It used to, and `proto/podwire.proto` records
/// why the field number stays reserved: a request that crosses a trust boundary has
/// no field naming its caller, so there is nothing for a caller to forge. Which
/// caller is asking is the authenticated connection's answer — `ptr-podwire` puts
/// it that way on the wire and `ptr-execwire` does the same for actions.
///
/// The reservation is held by this test rather than by the comment in the `.proto`
/// file: a comment cannot notice when somebody reuses the number. If `session_id`
/// ever comes back, this stops failing to compile and the test fails.
///
/// ```compile_fail
/// use ptr_protocol::generated::podwire::PodCall;
/// let _ = PodCall {
///     session_id: "a caller-supplied identity".into(),
///     call_id: "c".into(),
///     capability: "predict".into(),
///     generation: None,
///     revision: 0,
///     payload: None,
/// };
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallFrame {
    pub call_id: String,
    pub capability: CapabilityId,
    pub generation: Option<Generation>,
    pub revision: Revision,
    pub payload: TypedPayload,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolError {
    MissingCallId,
    MissingCapability,
    MissingPayload,
    MissingTypeId,
}

impl TryFrom<generated::podwire::PodCall> for CallFrame {
    type Error = ProtocolError;

    fn try_from(value: generated::podwire::PodCall) -> Result<Self, Self::Error> {
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
