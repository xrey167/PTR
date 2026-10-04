//! The wire format for one settlement and the receipt that answers it (C4 in
//! issue #23).
//!
//! A settlement is the second half of detached work requested over `ALPN_EXEC`:
//! the host committed the attempt, bound the requesting peer to settle it, and
//! answered [`WireOutcome::Pending`](crate::WireOutcome::Pending) naming the
//! attempt. The settler reports here, over `ALPN_SETTLE`, what the work did.
//!
//! Its own frames, its own magics and its own signing domain, for the reason it
//! has its own ALPN: a settlement and a request make opposite claims — one says
//! what an effect *did*, the other asks for one to be *caused* — and a frame that
//! could be read as either would be a request the host records as done, or a
//! report the host acts on. A settlement is refused as a request by its first eight
//! bytes, and the other way round.
//!
//! **A settlement carries no field naming its sender**, exactly as a request does
//! not: who is settling is the authenticated connection's answer, and whether that
//! peer may is the runtime's binding. It names whom it is *for* (`addressed_to`)
//! and *which attempt* it settles, and nothing else is a claim the host weighs.
//!
//! **What a settlement can report is that the work applied**, with what it
//! returned. There is no code for "it did not apply": that claim releases an
//! at-most-once key, and a wrong one lets the effect apply twice, so it stays the
//! host operator's to make through reconciliation (D6). A table with one entry is
//! still a table — the code is explicit and an unknown one is refused — so a later
//! build that adds a kind adds a code rather than a reinterpretation.
use crate::frame::{
    bound_frame, close, open_any, put_bytes, put_str, take_bytes, take_digest, take_fixed,
    take_str, take_u16, take_u64, take_u8, FrameError, ReceiptError, MAX_BODY_BYTES,
};
use ptr_net::{NodeSigner, SigningDomain, SIGNATURE_BYTES};
use ptr_types::NodeId;
use sha2::{Digest, Sha256};
use std::fmt;

/// Marks a settlement. Not a request's magic, so neither is read as the other.
const SETTLEMENT_MAGIC: &[u8; 8] = b"PTRSTREQ";

/// Marks a settlement receipt. Not an execution receipt's magic either: a
/// settlement receipt says a *report* was recorded, which an execution receipt
/// must never be mistaken for.
const SETTLEMENT_RECEIPT_MAGIC: &[u8; 8] = b"PTRSTRCP";

/// Domain separation for a settlement digest, distinct from the request digest's.
const SETTLEMENT_DIGEST_DOMAIN: &[u8] = b"PTRSTL01-SETTLE";

/// The only settlement layout this build writes and reads.
pub const SETTLEMENT_FORMAT_V1: u16 = 1;

/// The only settlement receipt layout this build writes and reads. Signed from
/// its first version: there is no unsigned layout of it to refuse.
pub const SETTLEMENT_RECEIPT_FORMAT_V1: u16 = 1;

/// The domain a settlement receipt signature is made in.
///
/// The signed message is this tag followed by **every byte of the receipt frame
/// before the signature**: magic, layout version, the length-prefixed
/// `responder`, `attempt`, `settlement_digest`, the outcome code and the outcome's
/// own field (the refusal code). The frame is the message, as for an execution
/// receipt, so no field can be left out of the signature. A different domain from
/// [`RECEIPT_DOMAIN`](crate::RECEIPT_DOMAIN), so a signature over one kind of
/// receipt verifies as nothing of the other kind.
pub const SETTLEMENT_RECEIPT_DOMAIN: SigningDomain = SigningDomain::new(b"ptr-settle-receipt/v1\0");

/// What the bound settler reports about a detached attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterReport {
    /// The work applied, and this is what it returned.
    Applied {
        /// What the work returned. Retained by the host up to its own bound and
        /// digested whatever its size.
        response: Vec<u8>,
    },
}

/// A settlement: the outcome of one detached attempt, reported to the host that
/// handed it out.
///
/// Decoding one of these settles nothing. The runtime accepts it only from the
/// peer bound to the attempt, once, while the attempt is outstanding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WireSettlement {
    /// The public key of the host the settler believes it is addressing, checked
    /// by the host against its own: a settlement meant for one host's attempt 7
    /// must not settle another host's attempt 7.
    pub addressed_to: String,
    /// The attempt the host named in its `Pending` receipt.
    pub attempt: u64,
    /// What the work did.
    pub report: AdapterReport,
}

/// What a host reports back about one settlement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SettlementReceipt {
    /// The public key of the host that produced this receipt.
    pub responder: String,
    /// The attempt the settlement named, or 0 when it did not decode far enough to
    /// name one.
    pub attempt: u64,
    /// A digest over the exact bytes that arrived, so a receipt is bound to one
    /// settlement and cannot be presented as the answer to another.
    pub settlement_digest: [u8; 32],
    /// What the host did with it.
    pub outcome: SettlementOutcome,
}

/// What a host did with a settlement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettlementOutcome {
    /// The report is in the host's history: the attempt is settled, its window is
    /// closed, and no further report or reconciliation can change it.
    Recorded,
    /// The settler was entitled to report and the host could not record it — no
    /// ledger index left, a failed durable write. **Not** a refusal: the binding
    /// stands and the attempt is still outstanding, so the same settlement can be
    /// sent again once the host has been reopened.
    NotRecorded,
    /// Nothing was recorded, at the coarseness the wire allows.
    Refused {
        /// Why.
        code: SettlementRefusal,
    },
}

/// Why a settlement was refused.
///
/// Every refusal about **authority** is the one `Runtime` code: a peer that is
/// not the attempt's settler, a settler since withdrawn, an attempt never bound,
/// one already settled or reconciled. A peer that could tell those apart could
/// probe which attempts exist and whose they are. The other two tell the settler
/// nothing it did not know: how it framed its bytes and which key it addressed.
///
/// There is no `Replayed`: a settlement is exactly-once by the runtime's own
/// record, durably, so a settlement sent twice is refused as settling an attempt
/// that is no longer outstanding, which is the `Runtime` code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettlementRefusal {
    /// The runtime refused. One code for every reason of authority.
    Runtime,
    /// The settlement did not decode.
    Malformed,
    /// The settlement named another host's key.
    Misaddressed,
}

impl SettlementRefusal {
    /// Stable diagnostic code, for a log rather than for the wire.
    pub fn code(self) -> &'static str {
        match self {
            Self::Runtime => "PTR_SETTLE_REFUSED",
            Self::Malformed => "PTR_SETTLE_MALFORMED",
            Self::Misaddressed => "PTR_SETTLE_MISADDRESSED",
        }
    }

    /// Every refusal this build can send, so a test can hold the table to account.
    pub const ALL: [Self; 3] = [Self::Runtime, Self::Malformed, Self::Misaddressed];
}

impl fmt::Display for SettlementRefusal {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// The digest a settlement receipt is bound to: taken over the bytes that arrived.
pub fn settlement_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SETTLEMENT_DIGEST_DOMAIN);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

/// The wire's name for a report kind. Zero is deliberately not one.
fn report_code(report: &AdapterReport) -> u8 {
    match report {
        AdapterReport::Applied { .. } => 1,
    }
}

/// The wire's name for what a host did with a settlement.
fn settlement_outcome_code(outcome: &SettlementOutcome) -> u8 {
    match outcome {
        SettlementOutcome::Recorded => 1,
        SettlementOutcome::NotRecorded => 2,
        SettlementOutcome::Refused { .. } => 3,
    }
}

/// The wire's name for a settlement refusal.
fn settlement_refusal_code(code: SettlementRefusal) -> u16 {
    match code {
        SettlementRefusal::Runtime => 1,
        SettlementRefusal::Malformed => 2,
        SettlementRefusal::Misaddressed => 3,
    }
}

/// Read a settlement refusal, or refuse a code this build has no meaning for.
fn settlement_refusal_from_code(code: u16) -> Result<SettlementRefusal, FrameError> {
    match code {
        1 => Ok(SettlementRefusal::Runtime),
        2 => Ok(SettlementRefusal::Malformed),
        3 => Ok(SettlementRefusal::Misaddressed),
        other => Err(FrameError::UnknownCode {
            field: "settlement_refusal",
            code: u64::from(other),
        }),
    }
}

/// Encode a settlement.
///
/// Layout, little-endian: magic `PTRSTREQ` (8), layout version (u16), the
/// length-prefixed `addressed_to`, `attempt` (u64), the report code (u8) and the
/// report's own field — for `Applied`, the length-prefixed response, bounded by
/// [`MAX_BODY_BYTES`].
pub fn encode_settlement(settlement: &WireSettlement) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(128);
    frame.extend_from_slice(SETTLEMENT_MAGIC);
    frame.extend_from_slice(&SETTLEMENT_FORMAT_V1.to_le_bytes());
    put_str(&mut frame, "addressed_to", &settlement.addressed_to)?;
    frame.extend_from_slice(&settlement.attempt.to_le_bytes());
    frame.push(report_code(&settlement.report));
    match &settlement.report {
        AdapterReport::Applied { response } => {
            put_bytes(&mut frame, "response", response, MAX_BODY_BYTES)?;
        }
    }
    bound_frame(frame)
}

/// Decode a settlement, or refuse it.
pub fn decode_settlement(bytes: &[u8]) -> Result<WireSettlement, FrameError> {
    let (mut at, format) = open_any(bytes, SETTLEMENT_MAGIC)?;
    if format != SETTLEMENT_FORMAT_V1 {
        return Err(FrameError::UnknownFormat { format });
    }
    let addressed_to = take_str(bytes, &mut at, "addressed_to")?;
    let attempt = take_u64(bytes, &mut at, "attempt")?;
    let report = match take_u8(bytes, &mut at, "report")? {
        1 => AdapterReport::Applied {
            response: take_bytes(bytes, &mut at, "response", MAX_BODY_BYTES)?,
        },
        other => {
            return Err(FrameError::UnknownCode {
                field: "report",
                code: u64::from(other),
            })
        }
    };
    close(bytes, at)?;
    Ok(WireSettlement {
        addressed_to,
        attempt,
        report,
    })
}

/// Encode a settlement receipt and sign it.
///
/// Layout, little-endian: magic `PTRSTRCP` (8), layout version (u16), the
/// length-prefixed `responder`, `attempt` (u64), `settlement_digest` (32), the
/// outcome code (u8) and, for a refusal, its code (u16), then the signature (64)
/// over every byte before it in [`SETTLEMENT_RECEIPT_DOMAIN`].
pub fn encode_settlement_receipt(
    receipt: &SettlementReceipt,
    signer: &NodeSigner,
) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(128 + SIGNATURE_BYTES);
    frame.extend_from_slice(SETTLEMENT_RECEIPT_MAGIC);
    frame.extend_from_slice(&SETTLEMENT_RECEIPT_FORMAT_V1.to_le_bytes());
    put_str(&mut frame, "responder", &receipt.responder)?;
    frame.extend_from_slice(&receipt.attempt.to_le_bytes());
    frame.extend_from_slice(&receipt.settlement_digest);
    frame.push(settlement_outcome_code(&receipt.outcome));
    match receipt.outcome {
        SettlementOutcome::Recorded | SettlementOutcome::NotRecorded => {}
        SettlementOutcome::Refused { code } => {
            frame.extend_from_slice(&settlement_refusal_code(code).to_le_bytes());
        }
    }
    let signature = signer.sign(SETTLEMENT_RECEIPT_DOMAIN, &frame);
    frame.extend_from_slice(&signature);
    bound_frame(frame)
}

/// Verify a settlement receipt as having been written by `responder`, and return
/// it.
///
/// The only way to read a settlement receipt out of bytes, and it needs no
/// connection: a receipt kept by the settler verifies offline against the host's
/// key alone.
///
/// **What a `Recorded` receipt that verifies attests:** that the runtime holding
/// the key `responder` recorded, as the settlement of its attempt `attempt`, the
/// settlement whose bytes have digest `settlement_digest`. Since the host accepts
/// a settlement only from the attempt's bound settler and only once, it also says
/// that the host treated the connection that sent those bytes as that settler.
///
/// **What it does not attest:** that the work applied — the settler said so, and
/// the receipt makes the *host's record* of that attributable, not true; who the
/// settler was — a settlement names no sender, so the receipt carries none, and a
/// verifier that needs it has the connection it arrived on; when; and that the
/// host is still trusted, which is the verifier's policy. A rotated key is a new
/// identity.
///
/// The checks run in the order they are worth doing: a frame that is not a signed
/// settlement receipt, then one naming another author, then the signature.
pub fn verify_settlement_receipt(
    bytes: &[u8],
    responder: &NodeId,
) -> Result<SettlementReceipt, ReceiptError> {
    let (receipt, signed, signature) = parse_settlement_receipt(bytes)?;
    if receipt.responder != responder.0 {
        return Err(ReceiptError::WrongResponder {
            expected: responder.0.clone(),
            claimed: receipt.responder,
        });
    }
    if !ptr_net::verify(&responder.0, SETTLEMENT_RECEIPT_DOMAIN, signed, &signature) {
        return Err(ReceiptError::BadSignature);
    }
    Ok(receipt)
}

/// A settlement receipt's fields, the bytes its signature covers, and the
/// signature — read without checking it. Private: see
/// [`verify_settlement_receipt`].
type ParsedSettlementReceipt<'a> = (SettlementReceipt, &'a [u8], [u8; SIGNATURE_BYTES]);

/// Read a signed settlement receipt without verifying it.
fn parse_settlement_receipt(bytes: &[u8]) -> Result<ParsedSettlementReceipt<'_>, FrameError> {
    let (mut at, format) = open_any(bytes, SETTLEMENT_RECEIPT_MAGIC)?;
    if format != SETTLEMENT_RECEIPT_FORMAT_V1 {
        return Err(FrameError::UnknownFormat { format });
    }
    let responder = take_str(bytes, &mut at, "responder")?;
    let attempt = take_u64(bytes, &mut at, "attempt")?;
    let settlement_digest = take_digest(bytes, &mut at, "settlement_digest")?;
    let outcome = match take_u8(bytes, &mut at, "settlement_outcome")? {
        1 => SettlementOutcome::Recorded,
        2 => SettlementOutcome::NotRecorded,
        3 => SettlementOutcome::Refused {
            code: settlement_refusal_from_code(take_u16(bytes, &mut at, "settlement_refusal")?)?,
        },
        other => {
            return Err(FrameError::UnknownCode {
                field: "settlement_outcome",
                code: u64::from(other),
            })
        }
    };
    let signed = &bytes[..at];
    let signature = take_fixed::<SIGNATURE_BYTES>(bytes, &mut at, "signature")?;
    close(bytes, at)?;
    Ok((
        SettlementReceipt {
            responder,
            attempt,
            settlement_digest,
            outcome,
        },
        signed,
        signature,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{
        decode_request, encode_receipt, encode_request, request_digest, verify_receipt,
        WireOutcome, WireReceipt, MAX_FRAME_BYTES, RECEIPT_DOMAIN,
    };

    fn signer() -> NodeSigner {
        NodeSigner::from_test_seed([7; 32])
    }

    fn stranger() -> NodeSigner {
        NodeSigner::from_test_seed([8; 32])
    }

    fn responder() -> NodeId {
        NodeId(signer().public_key().to_owned())
    }

    fn settlement() -> WireSettlement {
        WireSettlement {
            addressed_to: "key-of-the-host".to_owned(),
            attempt: 41,
            report: AdapterReport::Applied {
                response: b"queued and done".to_vec(),
            },
        }
    }

    fn receipt() -> SettlementReceipt {
        SettlementReceipt {
            responder: signer().public_key().to_owned(),
            attempt: 41,
            settlement_digest: settlement_digest(b"whatever arrived"),
            outcome: SettlementOutcome::Recorded,
        }
    }

    /// The bytes of `changed`, encoded, carrying the signature `original` was sent
    /// with.
    fn with_original_signature(changed: &SettlementReceipt, original: &[u8]) -> Vec<u8> {
        let mut frame = encode_settlement_receipt(changed, &signer()).unwrap();
        frame.truncate(frame.len() - SIGNATURE_BYTES);
        frame.extend_from_slice(&original[original.len() - SIGNATURE_BYTES..]);
        frame
    }

    #[test]
    fn a_settlement_and_every_settlement_receipt_round_trip() {
        let frame = encode_settlement(&settlement()).unwrap();
        assert_eq!(decode_settlement(&frame).unwrap(), settlement());
        let empty = WireSettlement {
            report: AdapterReport::Applied {
                response: Vec::new(),
            },
            ..settlement()
        };
        assert_eq!(
            decode_settlement(&encode_settlement(&empty).unwrap()).unwrap(),
            empty
        );
        let mut outcomes = vec![SettlementOutcome::Recorded, SettlementOutcome::NotRecorded];
        outcomes.extend(
            SettlementRefusal::ALL
                .iter()
                .map(|&code| SettlementOutcome::Refused { code }),
        );
        for outcome in outcomes {
            let answered = SettlementReceipt {
                outcome,
                ..receipt()
            };
            let frame = encode_settlement_receipt(&answered, &signer()).unwrap();
            assert_eq!(
                verify_settlement_receipt(&frame, &responder()).unwrap(),
                answered
            );
        }
    }

    #[test]
    fn a_settlement_is_never_read_as_a_request_nor_its_receipt_as_an_execution_receipt() {
        // Decided by the magic: a settlement reports an effect and a request asks
        // for one, so reading one as the other must not depend on the fields
        // failing to line up.
        let settle = encode_settlement(&settlement()).unwrap();
        assert_eq!(
            decode_request(&settle).expect_err("a settlement is not a request"),
            FrameError::WrongKind
        );
        let request = encode_request(&crate::frame::WireRequest {
            addressed_to: "key-of-the-host".to_owned(),
            request_id: 41,
            project: ptr_types::ProjectId::from("p"),
            action: ptr_core::action_head::ActionIr {
                operation: "write".into(),
                target: "capsule:a".into(),
                capability: ptr_types::CapabilityId::from("file.write"),
                effect: ptr_types::Effect::Mutation,
                input_type: ptr_types::TypeId::from("Bytes"),
                generation: ptr_types::Generation(1),
                revision: ptr_types::Revision(1),
                payload: b"verified payload".to_vec(),
            },
            once_key: None,
        })
        .unwrap();
        assert_eq!(
            decode_settlement(&request).expect_err("a request is not a settlement"),
            FrameError::WrongKind
        );
        // The two receipts: each refused as the other by its magic, and — so the
        // magic is not the only thing standing between them — a signature over one
        // kind is not valid in the other's domain.
        let settled = encode_settlement_receipt(&receipt(), &signer()).unwrap();
        assert_eq!(
            verify_receipt(&settled, &responder()),
            Err(ReceiptError::Frame(FrameError::WrongKind))
        );
        let executed = encode_receipt(
            &WireReceipt {
                responder: responder().0,
                request_id: 41,
                request_digest: request_digest(b"x"),
                outcome: WireOutcome::Pending { attempt: 41 },
            },
            &signer(),
        )
        .unwrap();
        assert_eq!(
            verify_settlement_receipt(&executed, &responder()),
            Err(ReceiptError::Frame(FrameError::WrongKind))
        );
        let body = &settled[..settled.len() - SIGNATURE_BYTES];
        let signature: [u8; SIGNATURE_BYTES] = settled[settled.len() - SIGNATURE_BYTES..]
            .try_into()
            .unwrap();
        assert!(ptr_net::verify(
            &responder().0,
            SETTLEMENT_RECEIPT_DOMAIN,
            body,
            &signature
        ));
        assert!(!ptr_net::verify(
            &responder().0,
            RECEIPT_DOMAIN,
            body,
            &signature
        ));
    }

    #[test]
    fn a_stored_settlement_receipt_verifies_offline_and_a_change_to_any_field_is_refused() {
        // The control: the stored bytes verify against the host's key alone.
        let stored = encode_settlement_receipt(&receipt(), &signer()).unwrap();
        assert_eq!(
            verify_settlement_receipt(&stored, &responder()).unwrap(),
            receipt()
        );
        let edits = [
            (
                "attempt",
                SettlementReceipt {
                    attempt: 42,
                    ..receipt()
                },
            ),
            (
                "settlement_digest",
                SettlementReceipt {
                    settlement_digest: settlement_digest(b"another settlement"),
                    ..receipt()
                },
            ),
            (
                "outcome",
                SettlementReceipt {
                    outcome: SettlementOutcome::NotRecorded,
                    ..receipt()
                },
            ),
            (
                "refusal",
                SettlementReceipt {
                    outcome: SettlementOutcome::Refused {
                        code: SettlementRefusal::Runtime,
                    },
                    ..receipt()
                },
            ),
        ];
        for (field, changed) in edits {
            assert_eq!(
                verify_settlement_receipt(
                    &with_original_signature(&changed, &stored),
                    &responder()
                ),
                Err(ReceiptError::BadSignature),
                "{field} edited after signing"
            );
        }
        for index in 0..stored.len() {
            for bit in 0..8 {
                let mut damaged = stored.clone();
                damaged[index] ^= 1 << bit;
                assert!(
                    verify_settlement_receipt(&damaged, &responder()).is_err(),
                    "byte {index} bit {bit} still verifies"
                );
            }
        }
        // Signed by another key, and presented as another host's: both refused.
        let forged = encode_settlement_receipt(&receipt(), &stranger()).unwrap();
        assert_eq!(
            verify_settlement_receipt(&forged, &responder()),
            Err(ReceiptError::BadSignature)
        );
        let other = NodeId(stranger().public_key().to_owned());
        assert!(matches!(
            verify_settlement_receipt(&stored, &other),
            Err(ReceiptError::WrongResponder { .. })
        ));
    }

    #[test]
    fn every_prefix_trailing_bytes_and_an_unknown_layout_are_refused() {
        for frame in [
            encode_settlement(&settlement()).unwrap(),
            encode_settlement_receipt(&receipt(), &signer()).unwrap(),
        ] {
            let is_settlement = decode_settlement(&frame).is_ok();
            for length in 0..frame.len() {
                let error = if is_settlement {
                    decode_settlement(&frame[..length]).expect_err("a prefix")
                } else {
                    match verify_settlement_receipt(&frame[..length], &responder()) {
                        Err(ReceiptError::Frame(error)) => error,
                        other => panic!("a prefix of {length}: {other:?}"),
                    }
                };
                assert!(
                    matches!(error, FrameError::WrongKind | FrameError::Truncated { .. }),
                    "prefix of {length}: {error:?}"
                );
            }
        }
        let mut trailing = encode_settlement(&settlement()).unwrap();
        trailing.push(0);
        assert_eq!(
            decode_settlement(&trailing),
            Err(FrameError::TrailingBytes { extra: 1 })
        );
        let mut future = encode_settlement(&settlement()).unwrap();
        future[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert_eq!(
            decode_settlement(&future),
            Err(FrameError::UnknownFormat { format: 2 })
        );
        let mut future = encode_settlement_receipt(&receipt(), &signer()).unwrap();
        future[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert_eq!(
            verify_settlement_receipt(&future, &responder()),
            Err(ReceiptError::Frame(FrameError::UnknownFormat { format: 2 }))
        );
        let oversize = vec![0_u8; MAX_FRAME_BYTES + 1];
        assert!(matches!(
            decode_settlement(&oversize),
            Err(FrameError::TooLarge { field: "frame", .. })
        ));
        let huge = WireSettlement {
            report: AdapterReport::Applied {
                response: vec![0; MAX_BODY_BYTES + 1],
            },
            ..settlement()
        };
        assert!(matches!(
            encode_settlement(&huge),
            Err(FrameError::TooLarge {
                field: "response",
                ..
            })
        ));
    }

    #[test]
    fn every_settlement_code_is_its_own_and_an_unknown_one_is_refused() {
        // The report kind: 1 is `Applied`, and nothing else is a kind, zero
        // included — a zeroed byte is what a damaged frame looks like.
        let frame = encode_settlement(&settlement()).unwrap();
        let kind_at = 8 + 2 + 4 + "key-of-the-host".len() + 8;
        assert_eq!(frame[kind_at], 1, "the report kind is where it is expected");
        for unknown in [0_u8, 2, 255] {
            let mut damaged = frame.clone();
            damaged[kind_at] = unknown;
            assert_eq!(
                decode_settlement(&damaged),
                Err(FrameError::UnknownCode {
                    field: "report",
                    code: u64::from(unknown)
                })
            );
        }
        // The outcome kinds and refusal codes: distinct, two-way, zero refused.
        let mut codes: Vec<u16> = SettlementRefusal::ALL
            .iter()
            .copied()
            .map(settlement_refusal_code)
            .collect();
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "two refusals share a wire code");
        for refusal in SettlementRefusal::ALL {
            assert_eq!(
                settlement_refusal_from_code(settlement_refusal_code(refusal)),
                Ok(refusal)
            );
        }
        assert!(settlement_refusal_from_code(0).is_err());
        let outcomes = [
            SettlementOutcome::Recorded,
            SettlementOutcome::NotRecorded,
            SettlementOutcome::Refused {
                code: SettlementRefusal::Runtime,
            },
        ];
        let mut kinds: Vec<u8> = outcomes.iter().map(settlement_outcome_code).collect();
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(
            kinds.len(),
            outcomes.len(),
            "two outcomes share a wire code"
        );
        let stored = encode_settlement_receipt(&receipt(), &signer()).unwrap();
        let outcome_at = stored.len() - SIGNATURE_BYTES - 1;
        let mut unknown = stored.clone();
        unknown[outcome_at] = 9;
        assert_eq!(
            verify_settlement_receipt(&unknown, &responder()),
            Err(ReceiptError::Frame(FrameError::UnknownCode {
                field: "settlement_outcome",
                code: 9
            }))
        );
        // Diagnostic codes are distinct from each other and from the execution
        // wire's, so a log line names which protocol refused.
        let mut names: Vec<&str> = SettlementRefusal::ALL.iter().map(|c| c.code()).collect();
        names.extend(crate::RefusalCode::ALL.iter().map(|c| c.code()));
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total);
    }
}
