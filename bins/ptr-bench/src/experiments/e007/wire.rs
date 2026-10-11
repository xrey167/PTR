//! The wire the cases send over, and the capture of it.
//!
//! Every payload goes through the real PodWire request frame codec (`encode_request` then
//! `decode_request`, so `MAX_BODY_BYTES` is enforced as on the network) before it reaches the
//! assembler, each under a fresh PodWire request id. Every payload is recorded with its role,
//! so the wire-KV measurements read the capture and never the admission code.

use super::params;
use crate::experiments::layered_kv::corpus::hex;
use ptr_ledger::integrity::sha256;
use ptr_podwire::{
    decode_request, encode_request, FrameAssembler, FrameError, Offered, PodRequest, Rejection,
};
use ptr_protocol::activation::{chunk_payload, ActivationChunk, ALLOWLISTED_TYPES};
use ptr_protocol::TypedPayload;
use ptr_types::CapabilityId;

/// What a payload on the wire is for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    /// Legitimate traffic: counted in the admitted-traffic byte measurements.
    Valid,
    /// The activation-shaped KV payload consumed only as activation (a positive control).
    Control,
    /// An adversarial probe, named by its kind: excluded from the admitted-traffic bytes.
    Probe(&'static str),
}

impl Role {
    pub fn name(self) -> String {
        match self {
            Self::Valid => "valid".to_string(),
            Self::Control => "activation_control".to_string(),
            Self::Probe(kind) => format!("probe:{kind}"),
        }
    }
}

/// What came of one send.
#[derive(Debug, PartialEq)]
pub enum Sent {
    /// The PodWire codec refused the request before the assembler saw it.
    Codec(FrameError),
    /// The assembler answered.
    Answered(Result<Offered, Rejection>),
}

impl Sent {
    /// The assembler's rejection, if the payload reached it and was refused.
    pub fn rejection(&self) -> Option<&Rejection> {
        match self {
            Self::Answered(Err(rejection)) => Some(rejection),
            _ => None,
        }
    }

    pub fn admitted(&self) -> bool {
        matches!(self, Self::Answered(Ok(_)))
    }
}

/// One payload that crossed the codec.
#[derive(Clone, Debug)]
pub struct Record {
    pub request_id: u64,
    pub role: Role,
    pub type_id: String,
    pub bytes: usize,
    pub digest: [u8; 32],
    pub verdict: String,
    /// Whether the payload was admitted (the assembler took it).
    pub admitted: bool,
    /// Kept for admitted legitimate payloads only, for the KV scan.
    pub body: Option<Vec<u8>>,
}

#[derive(Default)]
pub struct Wire {
    next_request: u64,
    pub records: Vec<Record>,
}

impl Wire {
    /// Send `payload` from `principal` over `session` at `now_ms`.
    pub fn send(
        &mut self,
        assembler: &mut FrameAssembler,
        now_ms: u64,
        session: u64,
        role: Role,
        payload: &TypedPayload,
    ) -> Sent {
        self.next_request += 1;
        let request_id = self.next_request;
        let request = PodRequest {
            addressed_to: params::ADDRESSED_TO.to_string(),
            request_id,
            capability: CapabilityId::from("split-activation"),
            expect_protocol: 1,
            payload: payload.clone(),
        };
        let mut record = Record {
            request_id,
            role,
            type_id: payload.type_id.0.clone(),
            bytes: payload.bytes.len(),
            digest: sha256(&payload.bytes),
            verdict: String::new(),
            admitted: false,
            body: None,
        };
        let decoded = match encode_request(&request).and_then(|bytes| decode_request(&bytes)) {
            Ok(decoded) => decoded,
            Err(error) => {
                record.verdict = format!("codec:{}", error.code());
                self.records.push(record);
                return Sent::Codec(error);
            }
        };
        let answer = assembler.offer(
            now_ms,
            params::UPSTREAM_PRINCIPAL,
            session,
            &decoded.payload,
        );
        record.admitted = answer.is_ok();
        record.verdict = match &answer {
            Ok(Offered::Buffered { .. }) => "buffered".to_string(),
            Ok(Offered::Delivered(_)) => "delivered".to_string(),
            Ok(Offered::Policy(_)) => "policy".to_string(),
            Err(rejection) => rejection.kind().to_string(),
        };
        if record.admitted && role == Role::Valid {
            record.body = Some(decoded.payload.bytes);
        }
        self.records.push(record);
        Sent::Answered(answer)
    }

    pub fn send_chunk(
        &mut self,
        assembler: &mut FrameAssembler,
        now_ms: u64,
        session: u64,
        role: Role,
        chunk: &ActivationChunk,
    ) -> Sent {
        self.send(assembler, now_ms, session, role, &chunk_payload(chunk))
    }

    pub fn json(&self) -> super::Json {
        use super::Json;
        let mut by_role: std::collections::BTreeMap<String, (i64, i64)> = Default::default();
        for record in &self.records {
            let entry = by_role.entry(record.role.name()).or_default();
            entry.0 += 1;
            entry.1 += record.bytes as i64;
        }
        Json::obj([
            ("payloads", Json::Int(self.records.len() as i64)),
            (
                "by_role",
                Json::Obj(
                    by_role
                        .into_iter()
                        .map(|(role, (count, bytes))| {
                            (
                                role,
                                Json::obj([
                                    ("payloads", Json::Int(count)),
                                    ("bytes", Json::Int(bytes)),
                                ]),
                            )
                        })
                        .collect(),
                ),
            ),
            (
                "records",
                Json::Arr(
                    self.records
                        .iter()
                        .map(|record| {
                            Json::obj([
                                ("request_id", Json::Int(record.request_id as i64)),
                                ("role", Json::str(record.role.name())),
                                ("type_id", Json::str(record.type_id.clone())),
                                ("bytes", Json::Int(record.bytes as i64)),
                                ("sha256", Json::str(hex(&record.digest[..8]))),
                                ("verdict", Json::str(record.verdict.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

/// The KV pages held by both pods, as the independent scan sees them: the bytes of each page
/// (16 rows of one layer's keys or values), their digests and the identifiers of the caches.
pub struct KvInventory {
    pages: Vec<Vec<u8>>,
    digests: std::collections::BTreeSet<[u8; 32]>,
    handles: Vec<Vec<u8>>,
}

impl KvInventory {
    /// `layers`: per layer, the keys bytes and the values bytes, as `Stage::kv_bytes` gives
    /// (keys then values joined; `kv_width` floats per row).
    pub fn new(layer_bytes: &[Vec<Vec<u8>>], row_bytes: usize, handles: Vec<Vec<u8>>) -> Self {
        let page_bytes = row_bytes * params::PAGE_TOKENS as usize;
        let mut pages = Vec::new();
        for pod in layer_bytes {
            for layer in pod {
                for tensor in layer.chunks(layer.len() / 2) {
                    for page in tensor.chunks(page_bytes) {
                        if page.len() == page_bytes {
                            pages.push(page.to_vec());
                        }
                    }
                }
            }
        }
        let digests = pages.iter().map(|page| sha256(page)).collect();
        Self {
            pages,
            digests,
            handles,
        }
    }

    pub fn pages(&self) -> &[Vec<u8>] {
        &self.pages
    }

    /// Whether `body` equals a KV page or carries one (page-aligned or not) as a window.
    pub fn carries_page(&self, body: &[u8]) -> bool {
        if self.digests.contains(&sha256(body)) {
            return true;
        }
        let Some(first) = self.pages.first() else {
            return false;
        };
        let _ = first;
        self.pages.iter().any(|page| contains(body, page))
    }

    /// Whether `body` carries the identifier or the digest of a cache this inventory holds.
    pub fn carries_handle(&self, body: &[u8]) -> bool {
        self.handles
            .iter()
            .any(|handle| !handle.is_empty() && contains(body, handle))
    }
}

/// Whether `haystack` contains `needle` (naive scan with a first-byte filter).
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    let first = needle[0];
    let last_start = haystack.len() - needle.len();
    let mut at = 0;
    while at <= last_start {
        match haystack[at..=last_start]
            .iter()
            .position(|byte| *byte == first)
        {
            None => return false,
            Some(offset) => {
                let start = at + offset;
                if &haystack[start..start + needle.len()] == needle {
                    return true;
                }
                at = start + 1;
            }
        }
    }
    false
}

/// The measured bytes of the admitted legitimate traffic that is no allowlisted schema or
/// carries a KV page, and the admitted payloads that carry a KV handle.
pub fn scan(records: &[Record], inventory: &KvInventory) -> (i64, i64) {
    let mut bytes = 0;
    let mut handles = 0;
    for record in records {
        if !record.admitted || record.role != Role::Valid {
            continue;
        }
        let allowlisted = ALLOWLISTED_TYPES.contains(&record.type_id.as_str());
        let Some(body) = record.body.as_deref() else {
            continue;
        };
        if !allowlisted || inventory.carries_page(body) {
            bytes += record.bytes as i64;
        }
        if inventory.carries_handle(body) {
            handles += 1;
        }
    }
    (bytes, handles)
}

/// The same measure over whole reassembled frames: the bytes of the frames that carry a KV page
/// and the number of frames that carry a cache handle.
pub fn scan_frames(frames: &[Vec<u8>], inventory: &KvInventory) -> (i64, i64) {
    let mut bytes = 0;
    let mut handles = 0;
    for frame in frames {
        if inventory.carries_page(frame) {
            bytes += frame.len() as i64;
        }
        if inventory.carries_handle(frame) {
            handles += 1;
        }
    }
    (bytes, handles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scan_finds_a_window_anywhere_and_nothing_that_is_not_there() {
        let page: Vec<u8> = (0..64u8).collect();
        let inventory = KvInventory {
            pages: vec![page.clone()],
            digests: [sha256(&page)].into_iter().collect(),
            handles: vec![b"kv-state-b".to_vec()],
        };
        let mut body = vec![9u8; 300];
        assert!(!inventory.carries_page(&body));
        body[137..137 + 64].copy_from_slice(&page);
        assert!(inventory.carries_page(&body));
        assert!(inventory.carries_page(&page));
        assert!(!inventory.carries_page(&page[..63]));
        let mut with_handle = vec![0u8; 20];
        with_handle.extend_from_slice(b"kv-state-b");
        assert!(inventory.carries_handle(&with_handle));
        assert!(!inventory.carries_handle(&body));
    }
}
