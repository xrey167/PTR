//! Built-in deterministic CPU reference Pods and the dispatch from a call to one of them.
//!
//! These are honest toys written against [`ptr_pods::DynPod`]: byte echo, CRC-32, a checked
//! integer sum and a per-process execution counter. None loads weights or touches the network.

use std::sync::atomic::{AtomicU64, Ordering};

use ptr_pods::{DynPod, PodManifest, PodRegistrationError, PodRegistry, TypedPayload};
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, TypeId};

use crate::wire::{WorkerCall, WorkerReply, BYTES_TYPE};

/// Project every built-in Pod belongs to.
pub const WORKER_PROJECT: &str = "worker";
/// Most integers the `sum` Pod accepts in one call.
pub const MAX_SUM_TERMS: usize = 4096;

/// Name and contract summary of a built-in Pod.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PodInfo {
    pub name: &'static str,
    pub summary: &'static str,
}

/// The built-in Pods, in the order `ptr-worker pods` lists them.
pub const BUILTIN: [PodInfo; 4] = [
    PodInfo {
        name: "echo",
        summary: "pure; returns the input bytes unchanged",
    },
    PodInfo {
        name: "crc32",
        summary: "pure; CRC-32 (IEEE, reflected) of the input bytes as 4 bytes big-endian",
    },
    PodInfo {
        name: "sum",
        summary: "pure; ASCII whitespace-separated i64 integers (at most 4096) in, \
                  checked decimal sum out",
    },
    PodInfo {
        name: "counter",
        summary: "mutation (process-local); returns, in decimal, how many times this process \
                  has executed it, this call included",
    },
];

fn manifest(name: &str, effect: Effect) -> PodManifest {
    PodManifest {
        project: ProjectId::from(WORKER_PROJECT),
        id: PodId::from(name),
        capabilities: vec![CapabilityId::from(name)],
        accepts: vec![TypeId::from(BYTES_TYPE)],
        produces: vec![TypeId::from(BYTES_TYPE)],
        effects: vec![effect],
        protocol_version: 1,
    }
}

fn bytes(output: Vec<u8>) -> TypedPayload {
    TypedPayload {
        type_id: TypeId::from(BYTES_TYPE),
        bytes: output,
    }
}

/// A stateless Pod: a manifest and a pure function.
type PureFn = fn(&[u8]) -> Result<Vec<u8>, String>;

struct FnPod {
    manifest: PodManifest,
    run: PureFn,
}

impl DynPod for FnPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        (self.run)(&input.bytes).map(bytes)
    }
}

struct CounterPod {
    manifest: PodManifest,
    count: AtomicU64,
}

impl DynPod for CounterPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, _input: TypedPayload) -> Result<TypedPayload, String> {
        let n = self.count.fetch_add(1, Ordering::SeqCst).saturating_add(1);
        Ok(bytes(n.to_string().into_bytes()))
    }
}

fn echo(input: &[u8]) -> Result<Vec<u8>, String> {
    Ok(input.to_vec())
}

/// CRC-32/ISO-HDLC (the zlib one); `crc32(b"123456789") == 0xCBF43926`.
pub fn crc32(input: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in input {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn crc32_pod(input: &[u8]) -> Result<Vec<u8>, String> {
    Ok(crc32(input).to_be_bytes().to_vec())
}

fn sum(input: &[u8]) -> Result<Vec<u8>, String> {
    let text = std::str::from_utf8(input).map_err(|_| "input-not-utf8".to_owned())?;
    let mut total: i64 = 0;
    for (index, term) in text.split_ascii_whitespace().enumerate() {
        if index >= MAX_SUM_TERMS {
            return Err("too-many-terms".to_owned());
        }
        let value: i64 = term.parse().map_err(|_| "bad-integer".to_owned())?;
        total = total
            .checked_add(value)
            .ok_or_else(|| "overflow".to_owned())?;
    }
    Ok(total.to_string().into_bytes())
}

/// A registry holding every built-in Pod, with fresh state (the counter starts at zero).
pub fn builtin_registry() -> Result<PodRegistry, PodRegistrationError> {
    let mut registry = PodRegistry::default();
    let stateless: [(&str, PureFn); 3] = [("echo", echo), ("crc32", crc32_pod), ("sum", sum)];
    for (name, run) in stateless {
        registry.register_new(std::sync::Arc::new(FnPod {
            manifest: manifest(name, Effect::Pure),
            run,
        }))?;
    }
    registry.register_new(std::sync::Arc::new(CounterPod {
        manifest: manifest("counter", Effect::Mutation),
        count: AtomicU64::new(0),
    }))?;
    Ok(registry)
}

/// Decodes a call payload, resolves the named Pod and runs it. Every failure is an error
/// reply; nothing here can fail the session.
pub fn dispatch(registry: &PodRegistry, payload: &TypedPayload) -> WorkerReply {
    let call = match WorkerCall::from_payload(payload) {
        Ok(call) => call,
        Err(error) => return WorkerReply::Err(format!("malformed-call:{}", error.code())),
    };
    let project = ProjectId::from(WORKER_PROJECT);
    let Some(pod) = registry.get(&project, &PodId::from(call.pod.as_str())) else {
        return WorkerReply::Err("unknown-pod".to_owned());
    };
    match pod.invoke(bytes(call.input)) {
        Ok(output) => WorkerReply::Ok(output.bytes),
        Err(message) => WorkerReply::Err(format!("pod-failed:{message}")),
    }
}

/// The text `ptr-worker pods` prints: one `name<TAB>summary` line per Pod.
pub fn listing() -> String {
    BUILTIN
        .iter()
        .map(|pod| format!("{}\t{}\n", pod.name, pod.summary))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(registry: &PodRegistry, pod: &str, input: &[u8]) -> WorkerReply {
        let payload = WorkerCall {
            pod: pod.to_owned(),
            input: input.to_vec(),
        }
        .to_payload()
        .unwrap();
        dispatch(registry, &payload)
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn builtin_pods_behave_as_documented() {
        let r = builtin_registry().unwrap();
        assert_eq!(call(&r, "echo", b"abc"), WorkerReply::Ok(b"abc".to_vec()));
        assert_eq!(
            call(&r, "crc32", b"123456789"),
            WorkerReply::Ok(vec![0xCB, 0xF4, 0x39, 0x26])
        );
        assert_eq!(
            call(&r, "sum", b"1 2\n3\t-4"),
            WorkerReply::Ok(b"2".to_vec())
        );
        assert_eq!(call(&r, "sum", b""), WorkerReply::Ok(b"0".to_vec()));
        assert_eq!(call(&r, "counter", b""), WorkerReply::Ok(b"1".to_vec()));
        assert_eq!(call(&r, "counter", b""), WorkerReply::Ok(b"2".to_vec()));
    }

    #[test]
    fn failures_are_error_replies() {
        let r = builtin_registry().unwrap();
        assert_eq!(
            call(&r, "nope", b""),
            WorkerReply::Err("unknown-pod".into())
        );
        let err = |input: &[u8]| call(&r, "sum", input);
        assert_eq!(
            err(b"1 x"),
            WorkerReply::Err("pod-failed:bad-integer".into())
        );
        assert_eq!(
            err(&[0xFF]),
            WorkerReply::Err("pod-failed:input-not-utf8".into())
        );
        assert_eq!(
            err(b"9223372036854775807 1"),
            WorkerReply::Err("pod-failed:overflow".into())
        );
        let many = "1 ".repeat(MAX_SUM_TERMS + 1);
        assert_eq!(
            err(many.as_bytes()),
            WorkerReply::Err("pod-failed:too-many-terms".into())
        );
        let bad = TypedPayload {
            type_id: TypeId::from("x"),
            bytes: Vec::new(),
        };
        assert_eq!(
            dispatch(&r, &bad),
            WorkerReply::Err("malformed-call:wrong-payload-type".into())
        );
    }

    #[test]
    fn listing_has_one_line_per_pod() {
        let text = listing();
        assert_eq!(text.lines().count(), BUILTIN.len());
        assert!(text.starts_with("echo\t"));
    }
}
