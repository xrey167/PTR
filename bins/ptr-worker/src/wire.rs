//! Payload layout of worker calls and replies, carried inside `ControlFrame::Call` /
//! `ControlFrame::Reply` as [`TypedPayload`] bytes.
//!
//! ```text
//! call  (type "worker.call.v1"):  u16 BE name length | pod name (UTF-8) | input bytes
//! reply (type "worker.reply.v1"): status u8 (0 ok, 1 error) | output bytes or UTF-8 error code
//! ```

use ptr_protocol::TypedPayload;
use ptr_types::TypeId;

pub const CALL_TYPE: &str = "worker.call.v1";
pub const REPLY_TYPE: &str = "worker.reply.v1";
/// Type of the input and output a built-in Pod accepts and produces.
pub const BYTES_TYPE: &str = "worker.bytes";
pub const MAX_POD_NAME_BYTES: usize = 64;
pub const MAX_INPUT_BYTES: usize = 64 * 1024;
/// Error text beyond this is cut so a reply stays small.
pub const MAX_ERROR_BYTES: usize = 256;

const STATUS_OK: u8 = 0;
const STATUS_ERROR: u8 = 1;

/// Why a payload is not a well-formed worker call or reply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WireError {
    WrongType,
    Truncated,
    EmptyName,
    NameTooLong { len: usize },
    NameNotUtf8,
    InputTooLarge { len: usize },
    UnknownStatus(u8),
}

impl WireError {
    /// Stable short code used in error replies.
    pub fn code(self) -> &'static str {
        match self {
            Self::WrongType => "wrong-payload-type",
            Self::Truncated => "truncated-call",
            Self::EmptyName => "empty-pod-name",
            Self::NameTooLong { .. } => "pod-name-too-long",
            Self::NameNotUtf8 => "pod-name-not-utf8",
            Self::InputTooLarge { .. } => "input-too-large",
            Self::UnknownStatus(_) => "unknown-status",
        }
    }
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for WireError {}

/// A call naming a Pod and its input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerCall {
    pub pod: String,
    pub input: Vec<u8>,
}

impl WorkerCall {
    pub fn to_payload(&self) -> Result<TypedPayload, WireError> {
        check_name(self.pod.as_bytes())?;
        check_input(self.input.len())?;
        let mut bytes = Vec::with_capacity(2 + self.pod.len() + self.input.len());
        bytes.extend_from_slice(&(self.pod.len() as u16).to_be_bytes());
        bytes.extend_from_slice(self.pod.as_bytes());
        bytes.extend_from_slice(&self.input);
        Ok(TypedPayload {
            type_id: TypeId::from(CALL_TYPE),
            bytes,
        })
    }

    pub fn from_payload(payload: &TypedPayload) -> Result<Self, WireError> {
        if payload.type_id.0 != CALL_TYPE {
            return Err(WireError::WrongType);
        }
        let bytes = &payload.bytes;
        let prefix = bytes.get(..2).ok_or(WireError::Truncated)?;
        let name_len = usize::from(u16::from_be_bytes([prefix[0], prefix[1]]));
        if name_len > MAX_POD_NAME_BYTES {
            return Err(WireError::NameTooLong { len: name_len });
        }
        let name = bytes.get(2..2 + name_len).ok_or(WireError::Truncated)?;
        check_name(name)?;
        let input = &bytes[2 + name_len..];
        check_input(input.len())?;
        Ok(Self {
            pod: String::from_utf8_lossy(name).into_owned(),
            input: input.to_vec(),
        })
    }
}

fn check_name(name: &[u8]) -> Result<(), WireError> {
    if name.is_empty() {
        return Err(WireError::EmptyName);
    }
    if name.len() > MAX_POD_NAME_BYTES {
        return Err(WireError::NameTooLong { len: name.len() });
    }
    std::str::from_utf8(name)
        .map(|_| ())
        .map_err(|_| WireError::NameNotUtf8)
}

fn check_input(len: usize) -> Result<(), WireError> {
    if len > MAX_INPUT_BYTES {
        Err(WireError::InputTooLarge { len })
    } else {
        Ok(())
    }
}

/// The worker's answer to a call: the Pod's output, or a short error code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkerReply {
    Ok(Vec<u8>),
    Err(String),
}

impl WorkerReply {
    pub fn to_payload(&self) -> TypedPayload {
        let mut bytes = Vec::new();
        match self {
            Self::Ok(output) => {
                bytes.push(STATUS_OK);
                bytes.extend_from_slice(output);
            }
            Self::Err(code) => {
                bytes.push(STATUS_ERROR);
                let mut end = code.len().min(MAX_ERROR_BYTES);
                while !code.is_char_boundary(end) {
                    end -= 1;
                }
                bytes.extend_from_slice(&code.as_bytes()[..end]);
            }
        }
        TypedPayload {
            type_id: TypeId::from(REPLY_TYPE),
            bytes,
        }
    }

    pub fn from_payload(payload: &TypedPayload) -> Result<Self, WireError> {
        if payload.type_id.0 != REPLY_TYPE {
            return Err(WireError::WrongType);
        }
        match payload.bytes.split_first() {
            None => Err(WireError::Truncated),
            Some((&STATUS_OK, rest)) => Ok(Self::Ok(rest.to_vec())),
            Some((&STATUS_ERROR, rest)) => {
                Ok(Self::Err(String::from_utf8_lossy(rest).into_owned()))
            }
            Some((&other, _)) => Err(WireError::UnknownStatus(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_round_trips_and_rejects_malformed_payloads() {
        let call = WorkerCall {
            pod: "echo".into(),
            input: b"hi".to_vec(),
        };
        let payload = call.to_payload().unwrap();
        assert_eq!(WorkerCall::from_payload(&payload).unwrap(), call);

        let mut wrong = payload.clone();
        wrong.type_id = TypeId::from("other");
        assert_eq!(WorkerCall::from_payload(&wrong), Err(WireError::WrongType));

        let mut short = payload.clone();
        short.bytes.truncate(1);
        assert_eq!(WorkerCall::from_payload(&short), Err(WireError::Truncated));
        short.bytes = vec![0, 9, b'a'];
        assert_eq!(WorkerCall::from_payload(&short), Err(WireError::Truncated));

        let mut empty = payload.clone();
        empty.bytes = vec![0, 0, b'x'];
        assert_eq!(WorkerCall::from_payload(&empty), Err(WireError::EmptyName));

        let mut bad = payload.clone();
        bad.bytes = vec![0, 1, 0xFF];
        assert_eq!(WorkerCall::from_payload(&bad), Err(WireError::NameNotUtf8));

        let mut long = payload;
        long.bytes = vec![0xFF, 0xFF];
        assert!(matches!(
            WorkerCall::from_payload(&long),
            Err(WireError::NameTooLong { .. })
        ));
    }

    #[test]
    fn oversize_input_is_refused_both_ways() {
        let call = WorkerCall {
            pod: "echo".into(),
            input: vec![0; MAX_INPUT_BYTES + 1],
        };
        assert!(matches!(
            call.to_payload(),
            Err(WireError::InputTooLarge { .. })
        ));
        let mut payload = WorkerCall {
            pod: "echo".into(),
            input: Vec::new(),
        }
        .to_payload()
        .unwrap();
        payload.bytes.extend(vec![0; MAX_INPUT_BYTES + 1]);
        assert!(matches!(
            WorkerCall::from_payload(&payload),
            Err(WireError::InputTooLarge { .. })
        ));
    }

    #[test]
    fn reply_round_trips_and_truncates_long_errors() {
        let ok = WorkerReply::Ok(vec![1, 2]);
        assert_eq!(WorkerReply::from_payload(&ok.to_payload()).unwrap(), ok);
        let long = WorkerReply::Err("é".repeat(MAX_ERROR_BYTES));
        let back = WorkerReply::from_payload(&long.to_payload()).unwrap();
        match back {
            WorkerReply::Err(text) => assert!(text.len() <= MAX_ERROR_BYTES && !text.is_empty()),
            WorkerReply::Ok(_) => panic!("expected an error reply"),
        }
        let mut bad = ok.to_payload();
        bad.bytes[0] = 9;
        assert_eq!(
            WorkerReply::from_payload(&bad),
            Err(WireError::UnknownStatus(9))
        );
        bad.bytes.clear();
        assert_eq!(WorkerReply::from_payload(&bad), Err(WireError::Truncated));
    }
}
