pub mod acknowledged;
pub mod anchor;
pub mod compaction;
mod file;
pub mod integrity;
pub mod retention;
pub use acknowledged::{AcknowledgedError, AcknowledgedLedger, Split, TailPolicy, TailRecovery};
pub use compaction::{
    CompactionDecision, CompactionFault, CompactionOutcome, CompactionPlan, LogPaths,
    RetentionPolicy,
};
pub use file::{FileLedger, LegacyLog, RecoverableLog};
pub use retention::{retains, ErasureAudit, OutOfReach, Retainer};

use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, Generation, ProjectId, Revision,
    VerificationLevel,
};
use std::collections::BTreeSet;
use std::io;
#[cfg(feature = "raft-engine-backend")]
use std::path::Path;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LedgerEvent {
    /// Opaque versioned semantic transaction. ptr-runtime validates its schema
    /// and both revisions before append and again during replay.
    SemanticDeltaCommitted {
        base_revision: Revision,
        revision: Revision,
        encoded_delta: Vec<u8>,
        /// Why the write was allowed, recorded in the same append as the
        /// delta. [`SemanticOrigin::Legacy`] encodes as tag 8, byte for byte
        /// as before origins existed; every other origin encodes as tag 12.
        origin: SemanticOrigin,
    },
    CapsuleCommitted {
        project: ProjectId,
        capsule: CapsuleId,
        generation: Generation,
    },
    CapsuleSuperseded {
        capsule: CapsuleId,
        old: Generation,
        new: Generation,
    },
    Revoked {
        subject: String,
        generation: Generation,
    },
    HardConstraintCommitted {
        key: String,
        generation: Generation,
    },
    VerifierAttested {
        subject: String,
        passed: bool,
    },
    ProcedurePromoted {
        id: String,
        generation: Generation,
    },
    ProcedureRevoked {
        id: String,
        generation: Generation,
    },
    SnapshotCommitted {
        revision: u64,
        covers: CommitIndex,
    },
    /// Committed *before* an external effect is attempted, so a crash inside
    /// the effect window leaves a durable record that the effect may already
    /// have applied. An attempt with no matching settlement *is* the fence:
    /// there is no separate flag to lose, and replay rebuilds it by definition.
    EffectAttempted {
        /// Caller-supplied at-most-once key, absent when the caller asked for
        /// no such guarantee. Only a caller knows whether a repeated request
        /// means "the same one again" or "do it once more".
        key: Option<String>,
        project: ProjectId,
        principal: String,
        target: String,
        operation: String,
        capability: CapabilityId,
        effect: Effect,
        generation: Generation,
        revision: Revision,
        /// The level that admitted this action. A record only exists after the
        /// verifier passed, so there is no status field that could disagree.
        verification: VerificationLevel,
        action_digest: [u8; 32],
    },
    /// The runtime observed the effect's own outcome and recorded it.
    EffectSettled {
        attempt: CommitIndex,
        /// Absent once the response exceeds [`MAX_RETAINED_RESPONSE`]. The
        /// digest is recorded either way, so "not retained" never becomes
        /// "not known".
        response: Option<Vec<u8>>,
        response_digest: [u8; 32],
    },
    /// An operator resolved an ambiguous attempt with evidence this runtime
    /// cannot obtain by itself. Reconciliation never guesses and never retries.
    EffectReconciled {
        attempt: CommitIndex,
        applied: bool,
        evidence: String,
    },
}

/// Why a semantic write was allowed. ptr-runtime writes it in the same append
/// as the delta and checks it with the same function on append and on every
/// rebuild; the codec checks only its shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticOrigin {
    /// Written before origins were recorded: tag 8. Decoded only; a runtime
    /// never appends it, and replay accepts it only before the first
    /// attributed record. Not the unframed PTRLOG01 format that
    /// `migrate_legacy_log` reads: a record of that format becomes a framed
    /// tag-8 record when it is migrated.
    Legacy,
    /// The raw text of one request, written by ingest.
    Request { request: String },
    /// One Pod's output for a request, promoted after the Pod's verifier
    /// passed with no hard finding, at the level it reported.
    PodOutput {
        request: String,
        pod: String,
        level: VerificationLevel,
    },
    /// A write by the host, admitted by the installed grant's verifiers.
    Host {
        principal: String,
        verification: Attestation,
    },
    /// A sealed branch the runtime certified, verified and merged.
    Merge(MergeRecord),
}

/// What admitted an attributed write: the level required, the weakest level
/// the grant's verifiers reported, their names in grant order and their soft
/// finding codes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Attestation {
    /// `FullSemantic` or `Deterministic`.
    pub required: VerificationLevel,
    /// The weakest level among the verifiers.
    pub level: VerificationLevel,
    /// 1 to [`MAX_ATTESTATION_VERIFIERS`] names, in grant order.
    pub verifiers: Vec<String>,
    /// Up to [`MAX_ATTESTATION_FINDINGS`] soft-finding codes, sorted and
    /// distinct, each `<verifier>/<code>`.
    pub findings: Vec<String>,
}

/// What a merge record binds: the branch and its author, the seal, plan and
/// dependency digests the runtime computed, the keys certification rebased,
/// the verification and the authority that let the merge commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergeRecord {
    pub branch: String,
    pub author: String,
    /// The sealed branch's digest (`SealedBranch::seal_digest`).
    pub seal: [u8; 32],
    /// The merge plan's digest, recomputable from this record and its delta.
    pub plan: [u8; 32],
    pub dependencies: [u8; 32],
    /// Keys certification rebased onto the target, which the plan digest
    /// covers; empty for a clean merge. At most [`MAX_REBASED_KEYS`].
    pub rebased: BTreeSet<String>,
    pub verification: Attestation,
    pub authority: MergeAuthorityRecord,
}

/// Why a verified merge was allowed to commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MergeAuthorityRecord {
    /// The runtime's own triage under the grant's policy returned
    /// AutoPropose; the score it was given, as `f32` bits.
    Triage {
        policy_version: String,
        score_bits: u32,
    },
    /// A reviewer the grant lists approved exactly this plan.
    Reviewed { reviewer: String },
}

/// Most verifiers an [`Attestation`] names.
pub const MAX_ATTESTATION_VERIFIERS: usize = 16;
/// Most soft-finding codes an [`Attestation`] carries.
pub const MAX_ATTESTATION_FINDINGS: usize = 32;
/// Most rebased keys a [`MergeRecord`] carries: the most items a semantic
/// delta can hold (`ptr_semdb::MAX_DELTA_ITEMS`), since a key is rebased only
/// where the delta writes it or leaves it equal to the target.
pub const MAX_REBASED_KEYS: usize = 16_384;

/// Largest effect response copied into the journal. Above it only the digest is
/// kept: an unbounded history is its own failure, and a silently truncated
/// response would be a second, wrong answer rather than a missing one.
pub const MAX_RETAINED_RESPONSE: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedEvent {
    pub index: CommitIndex,
    pub event: LedgerEvent,
}

pub trait Ledger {
    /// Append one event and return the index it was committed at.
    ///
    /// Fallible because a commit index can run out. Returning the index
    /// unconditionally would force an implementation to invent one at the
    /// ceiling, and the only values available there repeat an index already
    /// handed out — which is worse than refusing, since two records would then
    /// claim the same position in a history that is supposed to order them.
    fn append(&mut self, event: LedgerEvent) -> io::Result<CommitIndex>;
    fn events(&self) -> &[CommittedEvent];
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryLedger {
    base: CommitIndex,
    events: Vec<CommittedEvent>,
}

impl InMemoryLedger {
    /// A ledger that continues above a compaction floor.
    ///
    /// Restoring from a compacted snapshot must not renumber the retained records
    /// from 1: their indices are part of the committed history the snapshot's
    /// floor refers to, and renumbering them would make the two disagree.
    pub fn resuming_above(base: CommitIndex) -> Self {
        Self {
            base,
            events: Vec::new(),
        }
    }

    /// The compaction floor this ledger continues above.
    pub fn base(&self) -> CommitIndex {
        self.base
    }
}

impl Ledger for InMemoryLedger {
    fn append(&mut self, event: LedgerEvent) -> io::Result<CommitIndex> {
        // Checked rather than saturating: saturation hands the ceiling out
        // twice, so the second record silently claims a position the first one
        // already holds. The index is computed before anything is stored, so a
        // refused append leaves the ledger exactly as it was.
        let index = self
            .base
            .0
            .checked_add(self.events.len() as u64)
            .and_then(|count| count.checked_add(1))
            .map(CommitIndex)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "PTR_LEDGER_INDEX_EXHAUSTED")
            })?;
        self.events.push(CommittedEvent { index, event });
        Ok(index)
    }

    fn events(&self) -> &[CommittedEvent] {
        &self.events
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactionBarrier {
    pub snapshot_covers: CommitIndex,
    pub all_consumers_caught_up: bool,
    pub unresolved_revocations: usize,
}

impl CompactionBarrier {
    pub fn safe(&self) -> bool {
        self.all_consumers_caught_up && self.unresolved_revocations == 0
    }
}

fn encode_event(event: &LedgerEvent) -> Vec<u8> {
    let mut out = Vec::new();
    match event {
        LedgerEvent::SemanticDeltaCommitted {
            base_revision,
            revision,
            encoded_delta,
            origin,
        } => {
            // Legacy stays tag 8, byte for byte, so every log and anchor
            // written before origins existed keeps its exact encoding.
            out.push(if *origin == SemanticOrigin::Legacy {
                8
            } else {
                12
            });
            put_u64(&mut out, base_revision.0);
            put_u64(&mut out, revision.0);
            put_bytes(&mut out, encoded_delta);
            put_origin(&mut out, origin);
        }
        LedgerEvent::CapsuleCommitted {
            project,
            capsule,
            generation,
        } => {
            out.push(0);
            put_string(&mut out, &project.to_string());
            put_string(&mut out, &capsule.to_string());
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::CapsuleSuperseded { capsule, old, new } => {
            out.push(1);
            put_string(&mut out, &capsule.to_string());
            put_u64(&mut out, old.0);
            put_u64(&mut out, new.0);
        }
        LedgerEvent::Revoked {
            subject,
            generation,
        } => {
            out.push(2);
            put_string(&mut out, subject);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::HardConstraintCommitted { key, generation } => {
            out.push(3);
            put_string(&mut out, key);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::VerifierAttested { subject, passed } => {
            out.push(4);
            put_string(&mut out, subject);
            out.push(u8::from(*passed));
        }
        LedgerEvent::ProcedurePromoted { id, generation } => {
            out.push(5);
            put_string(&mut out, id);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::ProcedureRevoked { id, generation } => {
            out.push(6);
            put_string(&mut out, id);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::SnapshotCommitted { revision, covers } => {
            out.push(7);
            put_u64(&mut out, *revision);
            put_u64(&mut out, covers.0);
        }
        LedgerEvent::EffectAttempted {
            key,
            project,
            principal,
            target,
            operation,
            capability,
            effect,
            generation,
            revision,
            verification,
            action_digest,
        } => {
            out.push(9);
            put_optional_string(&mut out, key.as_deref());
            put_string(&mut out, &project.to_string());
            put_string(&mut out, principal);
            put_string(&mut out, target);
            put_string(&mut out, operation);
            put_string(&mut out, &capability.to_string());
            out.push(effect_code(*effect));
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            out.push(verification_code(*verification));
            out.extend_from_slice(action_digest);
        }
        LedgerEvent::EffectSettled {
            attempt,
            response,
            response_digest,
        } => {
            out.push(10);
            put_u64(&mut out, attempt.0);
            match response {
                Some(bytes) => {
                    out.push(1);
                    put_bytes(&mut out, bytes);
                }
                None => out.push(0),
            }
            out.extend_from_slice(response_digest);
        }
        LedgerEvent::EffectReconciled {
            attempt,
            applied,
            evidence,
        } => {
            out.push(11);
            put_u64(&mut out, attempt.0);
            out.push(u8::from(*applied));
            put_string(&mut out, evidence);
        }
    }
    out
}

fn decode_event(payload: &[u8]) -> io::Result<LedgerEvent> {
    let mut cursor = Cursor::new(payload);
    let tag = cursor.u8()?;
    let event = match tag {
        8 => LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            encoded_delta: cursor.bytes()?.to_vec(),
            origin: SemanticOrigin::Legacy,
        },
        12 => LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            encoded_delta: cursor.bytes()?.to_vec(),
            origin: attributed_origin(&mut cursor)?,
        },
        0 => LedgerEvent::CapsuleCommitted {
            project: ProjectId(cursor.string()?),
            capsule: CapsuleId(cursor.string()?),
            generation: Generation(cursor.u64()?),
        },
        1 => LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId(cursor.string()?),
            old: Generation(cursor.u64()?),
            new: Generation(cursor.u64()?),
        },
        2 => LedgerEvent::Revoked {
            subject: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        3 => LedgerEvent::HardConstraintCommitted {
            key: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        4 => LedgerEvent::VerifierAttested {
            subject: cursor.string()?,
            passed: match cursor.u8()? {
                0 => false,
                1 => true,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid bool byte {other}"),
                    ))
                }
            },
        },
        5 => LedgerEvent::ProcedurePromoted {
            id: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        6 => LedgerEvent::ProcedureRevoked {
            id: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        7 => LedgerEvent::SnapshotCommitted {
            revision: cursor.u64()?,
            covers: CommitIndex(cursor.u64()?),
        },
        9 => LedgerEvent::EffectAttempted {
            key: cursor.optional_string()?,
            project: ProjectId(cursor.string()?),
            principal: cursor.string()?,
            target: cursor.string()?,
            operation: cursor.string()?,
            capability: CapabilityId(cursor.string()?),
            effect: effect_from_code(cursor.u8()?)?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            verification: verification_from_code(cursor.u8()?)?,
            action_digest: cursor.digest()?,
        },
        10 => LedgerEvent::EffectSettled {
            attempt: CommitIndex(cursor.u64()?),
            response: match cursor.u8()? {
                0 => None,
                1 => Some(cursor.bytes()?.to_vec()),
                other => return Err(presence_byte(other)),
            },
            response_digest: cursor.digest()?,
        },
        11 => LedgerEvent::EffectReconciled {
            attempt: CommitIndex(cursor.u64()?),
            applied: match cursor.u8()? {
                0 => false,
                1 => true,
                other => return Err(presence_byte(other)),
            },
            evidence: cursor.string()?,
        },
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown ledger event tag {other}"),
            ))
        }
    };

    if !cursor.finished() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ledger event contains trailing bytes",
        ));
    }

    Ok(event)
}

/// Explicit codes rather than `as` casts. A Rust discriminant follows
/// declaration order, so inserting one variant would renumber every effect
/// after it and every already-written record would then decode as a different
/// effect than it was committed with.
pub fn effect_code(effect: Effect) -> u8 {
    match effect {
        Effect::Pure => 0,
        Effect::Read => 1,
        Effect::Mutation => 2,
        Effect::External => 3,
        Effect::Irreversible => 4,
    }
}

fn effect_from_code(code: u8) -> io::Result<Effect> {
    match code {
        0 => Ok(Effect::Pure),
        1 => Ok(Effect::Read),
        2 => Ok(Effect::Mutation),
        3 => Ok(Effect::External),
        4 => Ok(Effect::Irreversible),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown effect code {other}"),
        )),
    }
}

pub fn verification_code(level: VerificationLevel) -> u8 {
    match level {
        VerificationLevel::Unverified => 0,
        VerificationLevel::LatentAgreement => 1,
        VerificationLevel::SampleVerified => 2,
        VerificationLevel::FullSemantic => 3,
        VerificationLevel::Deterministic => 4,
    }
}

fn verification_from_code(code: u8) -> io::Result<VerificationLevel> {
    match code {
        0 => Ok(VerificationLevel::Unverified),
        1 => Ok(VerificationLevel::LatentAgreement),
        2 => Ok(VerificationLevel::SampleVerified),
        3 => Ok(VerificationLevel::FullSemantic),
        4 => Ok(VerificationLevel::Deterministic),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown verification level code {other}"),
        )),
    }
}

/// Whether `event` is within every bound its encoding carries: the counts of
/// an attributed origin and the size of the encoded record
/// ([`integrity::MAX_RECORD_BYTES`]). An event outside them is refused with
/// `InvalidData` rather than encoded, since an encoding the decoder refuses
/// would leave a log that cannot be read back. The durable append paths
/// check it, and a writer can check it before appending to a memory ledger,
/// which encodes nothing.
pub fn check_encodable(event: &LedgerEvent) -> Result<(), io::Error> {
    check_origin_bounds(event)?;
    let payload = encode_event(event);
    if payload.len() > integrity::MAX_RECORD_BYTES {
        return Err(integrity::invalid("PTR_LOG_PAYLOAD_LIMIT"));
    }
    Ok(())
}

/// The counts an attributed origin's encoding carries in a byte or a bounded
/// word, checked before any encoding so an append path refuses rather than
/// panics.
pub(crate) fn check_origin_bounds(event: &LedgerEvent) -> Result<(), io::Error> {
    if let LedgerEvent::SemanticDeltaCommitted { origin, .. } = event {
        let attestation = match origin {
            SemanticOrigin::Legacy
            | SemanticOrigin::Request { .. }
            | SemanticOrigin::PodOutput { .. } => None,
            SemanticOrigin::Host { verification, .. } => Some(verification),
            SemanticOrigin::Merge(merge) => {
                if merge.rebased.len() > MAX_REBASED_KEYS {
                    return Err(integrity::invalid("PTR_LEDGER_REBASED_KEY_LIMIT"));
                }
                Some(&merge.verification)
            }
        };
        if let Some(attestation) = attestation {
            if attestation.verifiers.is_empty()
                || attestation.verifiers.len() > MAX_ATTESTATION_VERIFIERS
                || attestation.findings.len() > MAX_ATTESTATION_FINDINGS
            {
                return Err(integrity::invalid("PTR_LEDGER_ATTESTATION_LIMIT"));
            }
        }
    }
    Ok(())
}

/// Tag 12's origin after the delta: a kind byte, then the kind's fields.
/// Called only for an attributed origin; `Legacy` has nothing after the delta.
fn put_origin(out: &mut Vec<u8>, origin: &SemanticOrigin) {
    match origin {
        SemanticOrigin::Legacy => {}
        SemanticOrigin::Request { request } => {
            out.push(1);
            put_string(out, request);
        }
        SemanticOrigin::PodOutput {
            request,
            pod,
            level,
        } => {
            out.push(2);
            put_string(out, request);
            put_string(out, pod);
            out.push(verification_code(*level));
        }
        SemanticOrigin::Host {
            principal,
            verification,
        } => {
            out.push(3);
            put_string(out, principal);
            put_attestation(out, verification);
        }
        SemanticOrigin::Merge(merge) => {
            out.push(4);
            put_string(out, &merge.branch);
            put_string(out, &merge.author);
            out.extend_from_slice(&merge.seal);
            out.extend_from_slice(&merge.plan);
            out.extend_from_slice(&merge.dependencies);
            let count = u32::try_from(merge.rebased.len()).expect("rebased keys are bounded");
            out.extend_from_slice(&count.to_le_bytes());
            for key in &merge.rebased {
                put_string(out, key);
            }
            put_attestation(out, &merge.verification);
            match &merge.authority {
                MergeAuthorityRecord::Triage {
                    policy_version,
                    score_bits,
                } => {
                    out.push(1);
                    put_string(out, policy_version);
                    out.extend_from_slice(&score_bits.to_le_bytes());
                }
                MergeAuthorityRecord::Reviewed { reviewer } => {
                    out.push(2);
                    put_string(out, reviewer);
                }
            }
        }
    }
}

fn put_attestation(out: &mut Vec<u8>, attestation: &Attestation) {
    out.push(verification_code(attestation.required));
    out.push(verification_code(attestation.level));
    // `check_encodable` bounds both counts; a writer checks it before append.
    out.push(u8::try_from(attestation.verifiers.len()).expect("verifier count is bounded"));
    for name in &attestation.verifiers {
        put_string(out, name);
    }
    out.push(u8::try_from(attestation.findings.len()).expect("finding count is bounded"));
    for code in &attestation.findings {
        put_string(out, code);
    }
}

fn attributed_origin(cursor: &mut Cursor<'_>) -> io::Result<SemanticOrigin> {
    Ok(match cursor.u8()? {
        1 => SemanticOrigin::Request {
            request: cursor.string()?,
        },
        2 => SemanticOrigin::PodOutput {
            request: cursor.string()?,
            pod: cursor.string()?,
            level: verification_from_code(cursor.u8()?)?,
        },
        3 => SemanticOrigin::Host {
            principal: cursor.string()?,
            verification: attestation(cursor)?,
        },
        4 => {
            let branch = cursor.string()?;
            let author = cursor.string()?;
            let seal = cursor.digest()?;
            let plan = cursor.digest()?;
            let dependencies = cursor.digest()?;
            let count = cursor.u32()? as usize;
            if count > MAX_REBASED_KEYS {
                return Err(integrity::invalid("PTR_LEDGER_REBASED_KEY_LIMIT"));
            }
            let mut rebased = BTreeSet::new();
            let mut previous: Option<String> = None;
            for _ in 0..count {
                let key = cursor.string()?;
                // Strictly ascending, so one set has one encoding.
                if previous.as_ref().is_some_and(|previous| key <= *previous) {
                    return Err(integrity::invalid("PTR_LEDGER_REBASED_KEY_ORDER"));
                }
                previous = Some(key.clone());
                rebased.insert(key);
            }
            let verification = attestation(cursor)?;
            let authority = match cursor.u8()? {
                1 => MergeAuthorityRecord::Triage {
                    policy_version: cursor.string()?,
                    score_bits: cursor.u32()?,
                },
                2 => MergeAuthorityRecord::Reviewed {
                    reviewer: cursor.string()?,
                },
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown merge authority {other}"),
                    ))
                }
            };
            SemanticOrigin::Merge(MergeRecord {
                branch,
                author,
                seal,
                plan,
                dependencies,
                rebased,
                verification,
                authority,
            })
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown semantic origin {other}"),
            ))
        }
    })
}

fn attestation(cursor: &mut Cursor<'_>) -> io::Result<Attestation> {
    let required = verification_from_code(cursor.u8()?)?;
    if !matches!(
        required,
        VerificationLevel::FullSemantic | VerificationLevel::Deterministic
    ) {
        return Err(integrity::invalid("PTR_LEDGER_ATTESTATION_REQUIREMENT"));
    }
    let level = verification_from_code(cursor.u8()?)?;
    let count = usize::from(cursor.u8()?);
    if count == 0 || count > MAX_ATTESTATION_VERIFIERS {
        return Err(integrity::invalid("PTR_LEDGER_ATTESTATION_LIMIT"));
    }
    let verifiers = (0..count)
        .map(|_| cursor.string())
        .collect::<io::Result<Vec<_>>>()?;
    let count = usize::from(cursor.u8()?);
    if count > MAX_ATTESTATION_FINDINGS {
        return Err(integrity::invalid("PTR_LEDGER_ATTESTATION_LIMIT"));
    }
    let findings = (0..count)
        .map(|_| cursor.string())
        .collect::<io::Result<Vec<_>>>()?;
    Ok(Attestation {
        required,
        level,
        verifiers,
        findings,
    })
}

fn presence_byte(value: u8) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("invalid presence byte {value}"),
    )
}

fn put_optional_string(out: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            out.push(1);
            put_string(out, value);
        }
        None => out.push(0),
    }
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    let length = u32::try_from(bytes.len()).expect("string length fits u32");
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(bytes);
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn u8(&mut self) -> io::Result<u8> {
        let value = *self.bytes.get(self.offset).ok_or_else(truncated)?;
        self.offset += 1;
        Ok(value)
    }

    fn u32(&mut self) -> io::Result<u32> {
        let end = self.offset.saturating_add(4);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
    }

    fn u64(&mut self) -> io::Result<u64> {
        let end = self.offset.saturating_add(8);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
    }

    fn bytes(&mut self) -> io::Result<&'a [u8]> {
        let length = self.u32()? as usize;
        let end = self.offset.checked_add(length).ok_or_else(truncated)?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn string(&mut self) -> io::Result<String> {
        String::from_utf8(self.bytes()?.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 string"))
    }

    fn optional_string(&mut self) -> io::Result<Option<String>> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.string()?)),
            other => Err(presence_byte(other)),
        }
    }

    fn digest(&mut self) -> io::Result<[u8; 32]> {
        let end = self.offset.saturating_add(32);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(bytes.try_into().expect("thirty-two bytes"))
    }
}

fn truncated() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "truncated ledger record")
}

#[cfg(feature = "raft-engine-backend")]
mod raft_engine_backend {
    use super::*;
    use raft_engine::{Config, Engine, LogBatch};

    const GROUP_ID: u64 = 1;
    const KEY_PREFIX: &[u8] = b"ptr/event/";

    pub struct RaftEngineLedger {
        engine: Engine,
        events: Vec<CommittedEvent>,
    }

    impl RaftEngineLedger {
        pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
            let config = Config {
                dir: path.as_ref().to_string_lossy().into_owned(),
                ..Default::default()
            };
            let engine = Engine::open(config).map_err(engine_error)?;

            let mut encoded = Vec::<(Vec<u8>, Vec<u8>)>::new();
            engine
                .scan_raw_messages(GROUP_ID, None, None, false, |key, value| {
                    if key.starts_with(KEY_PREFIX) {
                        encoded.push((key.to_vec(), value.to_vec()));
                    }
                    true
                })
                .map_err(engine_error)?;
            encoded.sort_by(|left, right| left.0.cmp(&right.0));

            let mut events = Vec::with_capacity(encoded.len());
            for (offset, (key, payload)) in encoded.into_iter().enumerate() {
                let index = parse_event_key(&key)?;
                let expected = offset as u64 + 1;
                if index != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("raft-engine ledger gap: expected {expected}, found {index}"),
                    ));
                }
                events.push(CommittedEvent {
                    index: CommitIndex(index),
                    event: decode_event(&payload)?,
                });
            }

            Ok(Self { engine, events })
        }

        pub fn append_durable(&mut self, event: LedgerEvent) -> io::Result<CommitIndex> {
            check_encodable(&event)?;
            let index = CommitIndex(self.events.len() as u64 + 1);
            let mut batch = LogBatch::default();
            batch.put(
                GROUP_ID,
                event_key(index).into_bytes(),
                encode_event(&event),
            );
            self.engine.write(&mut batch, true).map_err(engine_error)?;
            self.events.push(CommittedEvent { index, event });
            Ok(index)
        }

        pub fn events(&self) -> &[CommittedEvent] {
            &self.events
        }

        pub fn sync(&self) -> io::Result<()> {
            self.engine.sync().map_err(engine_error)
        }

        pub fn purge_expired_files(&self) -> io::Result<Vec<u64>> {
            self.engine.purge_expired_files().map_err(engine_error)
        }
    }

    fn event_key(index: CommitIndex) -> String {
        format!("ptr/event/{:020}", index.0)
    }

    fn parse_event_key(key: &[u8]) -> io::Result<u64> {
        if !key.starts_with(KEY_PREFIX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected raft-engine ledger key prefix",
            ));
        }
        let suffix = std::str::from_utf8(&key[KEY_PREFIX.len()..])
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "ledger key is not UTF-8"))?;
        suffix
            .parse::<u64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid ledger event index"))
    }

    fn engine_error(error: raft_engine::Error) -> io::Error {
        io::Error::other(error.to_string())
    }
}

#[cfg(feature = "raft-engine-backend")]
pub use raft_engine_backend::RaftEngineLedger;

#[cfg(feature = "raft-rs-backend")]
pub mod raft_node;
#[cfg(feature = "raft-rs-backend")]
pub mod raft_storage;
#[cfg(feature = "raft-rs-backend")]
pub use raft_node::{
    decode_message, encode_message, InstalledSnapshot, MembershipChange, RaftMessage, RaftNode,
    RetainedSnapshotAnchor, SnapshotAnchorMismatch, MAX_MESSAGE_BYTES,
};
#[cfg(feature = "raft-rs-backend")]
pub use raft_storage::{FileRaftStorage, MAX_ENTRY_BYTES, MAX_SNAPSHOT_BYTES};

#[cfg(feature = "raft-rs-backend")]
mod raft_rs_backend {
    use super::*;
    use crate::raft_node::RaftNode;
    use std::path::Path;

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RaftCommitReceipt {
        pub raft_index: u64,
        pub commit_index: CommitIndex,
    }

    /// A one-member group over durable state.
    ///
    /// It is one member, so it demonstrates nothing about partitions or leader
    /// changes — [`RaftNode`] is what composes into a group. What this adds is the
    /// convenience a single member allows: a proposal is decided by the time
    /// `propose` returns, because there is nobody to hear from.
    ///
    /// It is the same node type underneath, so there is one place where entries are
    /// persisted, hard state is flushed and committed entries are applied. Two
    /// copies of that sequence would be two chances to get the ordering wrong.
    pub struct SingleNodeRaftConsensus {
        node: RaftNode,
    }

    impl SingleNodeRaftConsensus {
        /// Open durable state under `dir`, recovering committed history.
        pub fn open(dir: &Path, node_id: u64) -> Result<Self, String> {
            let mut node = RaftNode::open(dir, node_id, &[node_id])?;
            let outbound = node.campaign()?;
            if !outbound.is_empty() {
                return Err("a one-member group has nobody to send votes to".into());
            }
            if !node.is_leader() {
                return Err("single-node raft group failed to become leader".into());
            }
            Ok(Self { node })
        }

        pub fn is_leader(&self) -> bool {
            self.node.is_leader()
        }

        pub fn committed_events(&self) -> &[CommittedEvent] {
            self.node.committed_events()
        }

        /// The durable term this node has seen.
        pub fn term(&self) -> u64 {
            self.node.term()
        }

        pub fn propose(&mut self, event: LedgerEvent) -> Result<RaftCommitReceipt, String> {
            let before = self.node.committed_events().len();
            let outbound = self.node.propose(event)?;
            if !outbound.is_empty() {
                return Err(
                    "a one-member group produced a message with no peer to receive it".into(),
                );
            }

            let committed =
                self.node.committed_events().get(before).ok_or_else(|| {
                    "proposal was not committed in single-node raft group".to_owned()
                })?;
            Ok(RaftCommitReceipt {
                raft_index: self.node.raft_committed(),
                commit_index: committed.index,
            })
        }
    }
}

#[cfg(feature = "raft-rs-backend")]
pub use raft_rs_backend::{RaftCommitReceipt, SingleNodeRaftConsensus};

#[cfg(test)]
mod effect_record_tests {
    use super::*;

    fn attempt_keyed(key: Option<&str>) -> LedgerEvent {
        LedgerEvent::EffectAttempted {
            key: key.map(str::to_owned),
            project: ProjectId::from("p"),
            principal: "alice".into(),
            target: "capsule:a".into(),
            operation: "write".into(),
            capability: CapabilityId::from("file.write"),
            effect: Effect::External,
            generation: Generation(3),
            revision: Revision(9),
            verification: VerificationLevel::FullSemantic,
            action_digest: [7; 32],
        }
    }

    fn attempt() -> LedgerEvent {
        attempt_keyed(Some("invoice-7"))
    }

    /// The codes are data, not discriminants. Inserting a variant must break a
    /// build here rather than silently renumber records that are already stored.
    #[test]
    fn every_effect_and_verification_member_has_a_pinned_code() {
        for (effect, code) in [
            (Effect::Pure, 0),
            (Effect::Read, 1),
            (Effect::Mutation, 2),
            (Effect::External, 3),
            (Effect::Irreversible, 4),
        ] {
            assert_eq!(effect_code(effect), code);
            assert_eq!(effect_from_code(code).unwrap(), effect);
        }
        for (level, code) in [
            (VerificationLevel::Unverified, 0),
            (VerificationLevel::LatentAgreement, 1),
            (VerificationLevel::SampleVerified, 2),
            (VerificationLevel::FullSemantic, 3),
            (VerificationLevel::Deterministic, 4),
        ] {
            assert_eq!(verification_code(level), code);
            assert_eq!(verification_from_code(code).unwrap(), level);
        }
    }

    #[test]
    fn effect_records_round_trip_through_their_canonical_encoding() {
        let records = [
            attempt(),
            attempt_keyed(None),
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(4),
                response: Some(b"executed".to_vec()),
                response_digest: [1; 32],
            },
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(4),
                response: None,
                response_digest: [2; 32],
            },
            LedgerEvent::EffectReconciled {
                attempt: CommitIndex(4),
                applied: true,
                evidence: "operator checked".into(),
            },
            LedgerEvent::EffectReconciled {
                attempt: CommitIndex(4),
                applied: false,
                evidence: "no record downstream".into(),
            },
        ];
        for record in records {
            let encoded = encode_event(&record);
            assert_eq!(decode_event(&encoded).unwrap(), record);
            // A record that decodes from a prefix of itself would mean the
            // framing does not pin its own length.
            assert!(decode_event(&encoded[..encoded.len() - 1]).is_err());
        }
    }

    /// The only byte two records differ in is the field that differs, which locates
    /// it without hand-counting offsets that would rot with the layout.
    fn sole_difference(left: &[u8], right: &[u8]) -> usize {
        assert_eq!(left.len(), right.len());
        let differing: Vec<usize> = (0..left.len()).filter(|i| left[*i] != right[*i]).collect();
        assert_eq!(differing.len(), 1, "expected exactly one differing byte");
        differing[0]
    }

    fn attempt_effect(effect: Effect) -> LedgerEvent {
        match attempt() {
            LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                capability,
                generation,
                revision,
                verification,
                action_digest,
                ..
            } => LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                capability,
                effect,
                generation,
                revision,
                verification,
                action_digest,
            },
            other => other,
        }
    }

    fn attempt_verification(verification: VerificationLevel) -> LedgerEvent {
        match attempt() {
            LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                capability,
                effect,
                generation,
                revision,
                action_digest,
                ..
            } => LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                capability,
                effect,
                generation,
                revision,
                verification,
                action_digest,
            },
            other => other,
        }
    }

    #[test]
    fn unknown_effect_and_verification_codes_are_refused() {
        assert!(effect_from_code(5).is_err());
        assert!(verification_from_code(5).is_err());

        // A code this build does not assign must stop the record rather than
        // decode as whichever member happens to sit nearby.
        let effect_offset = sole_difference(
            &encode_event(&attempt_effect(Effect::Pure)),
            &encode_event(&attempt_effect(Effect::Read)),
        );
        let mut encoded = encode_event(&attempt());
        assert_eq!(encoded[effect_offset], effect_code(Effect::External));
        encoded[effect_offset] = 5;
        assert!(decode_event(&encoded).is_err());

        let verification_offset = sole_difference(
            &encode_event(&attempt_verification(VerificationLevel::Unverified)),
            &encode_event(&attempt_verification(VerificationLevel::Deterministic)),
        );
        assert_ne!(verification_offset, effect_offset);
        let mut encoded = encode_event(&attempt());
        assert_eq!(
            encoded[verification_offset],
            verification_code(VerificationLevel::FullSemantic)
        );
        encoded[verification_offset] = 5;
        assert!(decode_event(&encoded).is_err());
    }

    #[test]
    fn an_invalid_presence_or_boolean_byte_is_refused() {
        // A third value for a two-valued field would be a second encoding of one
        // of the two meanings.
        let mut encoded = encode_event(&attempt());
        assert_eq!(encoded[1], 1, "the key's presence byte follows the tag");
        encoded[1] = 2;
        assert!(decode_event(&encoded).is_err());

        let mut encoded = encode_event(&LedgerEvent::EffectSettled {
            attempt: CommitIndex(4),
            response: None,
            response_digest: [0; 32],
        });
        assert_eq!(
            encoded[9], 0,
            "the response presence byte follows the index"
        );
        encoded[9] = 2;
        assert!(decode_event(&encoded).is_err());

        let mut encoded = encode_event(&LedgerEvent::EffectReconciled {
            attempt: CommitIndex(4),
            applied: true,
            evidence: "checked".into(),
        });
        assert_eq!(encoded[9], 1, "the applied byte follows the index");
        encoded[9] = 2;
        assert!(decode_event(&encoded).is_err());
    }

    #[test]
    fn a_truncated_digest_is_refused_rather_than_zero_filled() {
        let encoded = encode_event(&attempt());
        for missing in 1..=32 {
            assert!(decode_event(&encoded[..encoded.len() - missing]).is_err());
        }
    }

    #[test]
    fn trailing_bytes_after_an_effect_record_are_refused() {
        for record in [
            attempt(),
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(1),
                response: None,
                response_digest: [0; 32],
            },
            LedgerEvent::EffectReconciled {
                attempt: CommitIndex(1),
                applied: false,
                evidence: "checked".into(),
            },
        ] {
            let mut encoded = encode_event(&record);
            encoded.push(0);
            assert!(decode_event(&encoded).is_err());
        }
    }
}

#[cfg(test)]
mod semantic_origin_tests {
    use super::*;

    fn attestation() -> Attestation {
        Attestation {
            required: VerificationLevel::FullSemantic,
            level: VerificationLevel::Deterministic,
            verifiers: vec!["schema".into(), "replay".into()],
            findings: vec!["replay/slow".into()],
        }
    }

    fn merge(authority: MergeAuthorityRecord, rebased: &[&str]) -> SemanticOrigin {
        SemanticOrigin::Merge(MergeRecord {
            branch: "branch-7".into(),
            author: "alice".into(),
            seal: [1; 32],
            plan: [2; 32],
            dependencies: [3; 32],
            rebased: rebased.iter().map(|key| (*key).to_owned()).collect(),
            verification: attestation(),
            authority,
        })
    }

    fn committed(origin: SemanticOrigin) -> LedgerEvent {
        LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(2),
            revision: Revision(3),
            encoded_delta: b"abc".to_vec(),
            origin,
        }
    }

    /// One origin of every attributed kind, every Pod level, both merge
    /// authorities, and an attestation at each of its bounds.
    fn attributed_origins() -> Vec<SemanticOrigin> {
        let mut origins = vec![SemanticOrigin::Request {
            request: "request:1".into(),
        }];
        for level in [
            VerificationLevel::Unverified,
            VerificationLevel::LatentAgreement,
            VerificationLevel::SampleVerified,
            VerificationLevel::FullSemantic,
            VerificationLevel::Deterministic,
        ] {
            origins.push(SemanticOrigin::PodOutput {
                request: "request:1".into(),
                pod: "pod-a".into(),
                level,
            });
        }
        origins.push(SemanticOrigin::Host {
            principal: "operator".into(),
            verification: attestation(),
        });
        origins.push(SemanticOrigin::Host {
            principal: "operator".into(),
            verification: Attestation {
                required: VerificationLevel::Deterministic,
                level: VerificationLevel::Unverified,
                verifiers: (0..MAX_ATTESTATION_VERIFIERS)
                    .map(|n| format!("verifier-{n:02}"))
                    .collect(),
                findings: (0..MAX_ATTESTATION_FINDINGS)
                    .map(|n| format!("verifier-00/finding-{n:02}"))
                    .collect(),
            },
        });
        origins.push(merge(
            MergeAuthorityRecord::Triage {
                policy_version: "policy-3".into(),
                score_bits: 0.75f32.to_bits(),
            },
            &[],
        ));
        origins.push(merge(
            MergeAuthorityRecord::Reviewed {
                reviewer: "bob".into(),
            },
            &["doc:a", "doc:b", "fact:z"],
        ));
        origins
    }

    /// A length-prefixed string, written by hand rather than by the encoder.
    fn text(value: &str) -> Vec<u8> {
        let mut out = u32::try_from(value.len()).unwrap().to_le_bytes().to_vec();
        out.extend_from_slice(value.as_bytes());
        out
    }

    /// The part of `committed`'s encoding before the origin: the tag, both
    /// revisions and the delta.
    fn prefix(tag: u8) -> Vec<u8> {
        let mut out = vec![tag];
        out.extend_from_slice(&2u64.to_le_bytes());
        out.extend_from_slice(&3u64.to_le_bytes());
        out.extend(text("abc"));
        out
    }

    /// An attestation with the counts its lists give, so a test can write
    /// counts the encoder never would.
    fn raw_attestation(required: u8, level: u8, verifiers: &[&str], findings: &[&str]) -> Vec<u8> {
        let mut out = vec![required, level, u8::try_from(verifiers.len()).unwrap()];
        for name in verifiers {
            out.extend(text(name));
        }
        out.push(u8::try_from(findings.len()).unwrap());
        for code in findings {
            out.extend(text(code));
        }
        out
    }

    fn host(attestation: Vec<u8>) -> Vec<u8> {
        let mut out = prefix(12);
        out.push(3);
        out.extend(text("operator"));
        out.extend(attestation);
        out
    }

    /// A merge origin up to its rebased keys, with a raw count.
    fn merge_head(count: u32, rebased: &[&str]) -> Vec<u8> {
        let mut out = prefix(12);
        out.push(4);
        out.extend(text("branch-7"));
        out.extend(text("alice"));
        out.extend_from_slice(&[1; 32]);
        out.extend_from_slice(&[2; 32]);
        out.extend_from_slice(&[3; 32]);
        out.extend_from_slice(&count.to_le_bytes());
        for key in rebased {
            out.extend(text(key));
        }
        out
    }

    fn refusal(payload: &[u8]) -> String {
        decode_event(payload)
            .expect_err("the payload must be refused")
            .to_string()
    }

    #[test]
    fn a_legacy_origin_encodes_as_tag_eight_byte_for_byte() {
        // What every build before origins wrote for this record: the tag, both
        // revisions, the length-prefixed delta, and nothing after it.
        let golden = [
            8, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, b'a', b'b', b'c',
        ];
        let legacy = committed(SemanticOrigin::Legacy);
        assert_eq!(encode_event(&legacy), golden);
        assert_eq!(decode_event(&golden).unwrap(), legacy);
        // Nothing may follow a tag-8 delta, not even a well-formed origin.
        let mut followed = golden.to_vec();
        followed.push(1);
        followed.extend(text("request:1"));
        assert_eq!(refusal(&followed), "ledger event contains trailing bytes");
    }

    #[test]
    fn every_attested_origin_round_trips_through_tag_twelve() {
        for origin in attributed_origins() {
            let event = committed(origin);
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], 12, "{event:?}");
            assert_eq!(decode_event(&encoded).unwrap(), event);
        }

        // The layout of each kind, written out by hand.
        let mut request = prefix(12);
        request.push(1);
        request.extend(text("request:1"));
        assert_eq!(
            encode_event(&committed(SemanticOrigin::Request {
                request: "request:1".into()
            })),
            request
        );

        let mut pod = prefix(12);
        pod.push(2);
        pod.extend(text("request:1"));
        pod.extend(text("pod-a"));
        pod.push(2);
        assert_eq!(
            encode_event(&committed(SemanticOrigin::PodOutput {
                request: "request:1".into(),
                pod: "pod-a".into(),
                level: VerificationLevel::SampleVerified,
            })),
            pod
        );

        let host_layout = host(raw_attestation(
            3,
            4,
            &["schema", "replay"],
            &["replay/slow"],
        ));
        assert_eq!(
            encode_event(&committed(SemanticOrigin::Host {
                principal: "operator".into(),
                verification: attestation(),
            })),
            host_layout
        );

        let mut reviewed = merge_head(3, &["doc:a", "doc:b", "fact:z"]);
        reviewed.extend(raw_attestation(
            3,
            4,
            &["schema", "replay"],
            &["replay/slow"],
        ));
        reviewed.push(2);
        reviewed.extend(text("bob"));
        assert_eq!(
            encode_event(&committed(merge(
                MergeAuthorityRecord::Reviewed {
                    reviewer: "bob".into()
                },
                &["fact:z", "doc:b", "doc:a"],
            ))),
            reviewed,
            "rebased keys are written in ascending order"
        );

        let mut triage = merge_head(0, &[]);
        triage.extend(raw_attestation(
            3,
            4,
            &["schema", "replay"],
            &["replay/slow"],
        ));
        triage.push(1);
        triage.extend(text("policy-3"));
        triage.extend_from_slice(&0.75f32.to_bits().to_le_bytes());
        assert_eq!(
            encode_event(&committed(merge(
                MergeAuthorityRecord::Triage {
                    policy_version: "policy-3".into(),
                    score_bits: 0.75f32.to_bits(),
                },
                &[],
            ))),
            triage
        );
    }

    #[test]
    fn unknown_kinds_levels_counts_key_order_and_trailing_bytes_are_refused() {
        // An origin kind outside 1..=4, or none at all.
        for kind in [0, 5, 255] {
            let mut payload = prefix(12);
            payload.push(kind);
            payload.extend(text("request:1"));
            assert_eq!(refusal(&payload), format!("unknown semantic origin {kind}"));
        }
        assert_eq!(
            decode_event(&prefix(12)).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );

        // A Pod level outside the five codes.
        let mut pod = prefix(12);
        pod.push(2);
        pod.extend(text("request:1"));
        pod.extend(text("pod-a"));
        pod.push(5);
        assert_eq!(refusal(&pod), "unknown verification level code 5");

        // A requirement below FullSemantic, or a level code outside the five.
        for required in [0, 1, 2] {
            assert_eq!(
                refusal(&host(raw_attestation(required, 4, &["v"], &[]))),
                "PTR_LEDGER_ATTESTATION_REQUIREMENT"
            );
        }
        for required in [3, 4] {
            assert!(decode_event(&host(raw_attestation(required, 0, &["v"], &[]))).is_ok());
        }
        assert_eq!(
            refusal(&host(raw_attestation(5, 4, &["v"], &[]))),
            "unknown verification level code 5"
        );
        assert_eq!(
            refusal(&host(raw_attestation(3, 5, &["v"], &[]))),
            "unknown verification level code 5"
        );

        // Verifier counts outside 1..=16 and finding counts above 32.
        let names: Vec<String> = (0..=MAX_ATTESTATION_VERIFIERS)
            .map(|n| format!("v{n}"))
            .collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        assert_eq!(
            refusal(&host(raw_attestation(3, 4, &[], &[]))),
            "PTR_LEDGER_ATTESTATION_LIMIT"
        );
        assert!(decode_event(&host(raw_attestation(
            3,
            4,
            &names[..MAX_ATTESTATION_VERIFIERS],
            &[]
        )))
        .is_ok());
        assert_eq!(
            refusal(&host(raw_attestation(3, 4, &names, &[]))),
            "PTR_LEDGER_ATTESTATION_LIMIT"
        );
        let codes: Vec<String> = (0..=MAX_ATTESTATION_FINDINGS)
            .map(|n| format!("v0/f{n}"))
            .collect();
        let codes: Vec<&str> = codes.iter().map(String::as_str).collect();
        assert!(decode_event(&host(raw_attestation(
            3,
            4,
            &["v0"],
            &codes[..MAX_ATTESTATION_FINDINGS]
        )))
        .is_ok());
        assert_eq!(
            refusal(&host(raw_attestation(3, 4, &["v0"], &codes))),
            "PTR_LEDGER_ATTESTATION_LIMIT"
        );

        // Rebased keys out of order, duplicated, or more than a delta holds.
        // The count is refused before any key is read.
        let tail = |mut head: Vec<u8>| {
            head.extend(raw_attestation(3, 4, &["v"], &[]));
            head.push(2);
            head.extend(text("bob"));
            head
        };
        assert!(decode_event(&tail(merge_head(2, &["a", "b"]))).is_ok());
        assert_eq!(
            refusal(&tail(merge_head(2, &["b", "a"]))),
            "PTR_LEDGER_REBASED_KEY_ORDER"
        );
        assert_eq!(
            refusal(&tail(merge_head(2, &["a", "a"]))),
            "PTR_LEDGER_REBASED_KEY_ORDER"
        );
        let too_many = u32::try_from(MAX_REBASED_KEYS + 1).unwrap();
        assert_eq!(
            refusal(&merge_head(too_many, &[])),
            "PTR_LEDGER_REBASED_KEY_LIMIT"
        );

        // A merge authority outside 1..=2.
        for authority in [0, 3] {
            let mut payload = merge_head(0, &[]);
            payload.extend(raw_attestation(3, 4, &["v"], &[]));
            payload.push(authority);
            payload.extend(text("bob"));
            assert_eq!(
                refusal(&payload),
                format!("unknown merge authority {authority}")
            );
        }

        // Every attributed record refuses a trailing byte and every cut.
        for origin in attributed_origins() {
            let encoded = encode_event(&committed(origin));
            let mut followed = encoded.clone();
            followed.push(0);
            assert_eq!(refusal(&followed), "ledger event contains trailing bytes");
            for end in 0..encoded.len() {
                assert!(decode_event(&encoded[..end]).is_err(), "cut at {end}");
            }
        }
    }

    #[test]
    fn an_origin_outside_its_bounds_is_refused_before_it_is_encoded() {
        let host = |verifiers: usize, findings: usize| {
            committed(SemanticOrigin::Host {
                principal: "operator".into(),
                verification: Attestation {
                    verifiers: (0..verifiers).map(|n| format!("v{n}")).collect(),
                    findings: (0..findings).map(|n| format!("v0/f{n}")).collect(),
                    ..attestation()
                },
            })
        };
        for (verifiers, findings) in [
            (0, 0),
            (MAX_ATTESTATION_VERIFIERS + 1, 0),
            (1, MAX_ATTESTATION_FINDINGS + 1),
        ] {
            let error = check_encodable(&host(verifiers, findings)).unwrap_err();
            assert_eq!(error.to_string(), "PTR_LEDGER_ATTESTATION_LIMIT");
        }
        assert!(
            check_encodable(&host(MAX_ATTESTATION_VERIFIERS, MAX_ATTESTATION_FINDINGS)).is_ok()
        );

        let SemanticOrigin::Merge(mut record) = merge(
            MergeAuthorityRecord::Reviewed {
                reviewer: "bob".into(),
            },
            &[],
        ) else {
            unreachable!("merge builds a merge origin")
        };
        record.rebased = (0..=MAX_REBASED_KEYS).map(|n| format!("k{n:05}")).collect();
        let error = check_encodable(&committed(SemanticOrigin::Merge(record.clone()))).unwrap_err();
        assert_eq!(error.to_string(), "PTR_LEDGER_REBASED_KEY_LIMIT");
        record.rebased.pop_last();
        assert!(check_encodable(&committed(SemanticOrigin::Merge(record))).is_ok());

        let mut empty_merge = merge(
            MergeAuthorityRecord::Reviewed {
                reviewer: "bob".into(),
            },
            &[],
        );
        if let SemanticOrigin::Merge(record) = &mut empty_merge {
            record.verification.verifiers.clear();
        }
        let error = check_encodable(&committed(empty_merge)).unwrap_err();
        assert_eq!(error.to_string(), "PTR_LEDGER_ATTESTATION_LIMIT");
    }
}
