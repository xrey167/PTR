//! What a cycle produces: refusals, slot content, retirements and the immutable snapshot.

use super::gate::{level_rank, status_code, GateRefusal};
use super::proposal::{Bid, Encoder, PayloadRef, Proposal, SlotIndex, WorkspaceProposalId};
use super::WorkspaceRevision;
use ptr_types::{Digest, Generation, PodId, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport};

use super::config::FindingVisibility;

const SNAPSHOT_DOMAIN: &[u8] = b"ptr.pods.workspace.snapshot.v2";
const REPORT_DOMAIN: &[u8] = b"ptr.pods.workspace.report.v1";

/// Findings kept per refusal or slot; the rest are counted in [`Refusal::findings_dropped`].
pub const MAX_FINDINGS_PER_REFUSAL: usize = 32;
/// Findings kept in one snapshot across all refusals; the rest are counted as dropped.
pub const MAX_SNAPSHOT_FINDINGS: usize = 2048;
/// Bytes kept of one finding message (host-side, and in snapshots only when
/// [`FindingVisibility::Public`]).
pub const MAX_FINDING_MESSAGE_BYTES: usize = 256;
/// Bytes kept of one finding code.
pub const MAX_FINDING_CODE_BYTES: usize = 64;
/// Longest verifier identity name.
pub const MAX_VERIFIER_NAME_BYTES: usize = 64;

/// What a slot's content is: workspace state derived from verified proposals. It is never
/// committed semantic state (invariant 4) and must not be handed to `ptr-state` or `ptr-runtime`
/// as such.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Authority {
    /// Provisional, workspace-local, non-authoritative.
    Provisional,
}

/// Identity of the host verifier that judged a cycle, recorded in the snapshot so it is
/// auditable. The verifier must be a deterministic function of the [`Proposal`] for replay
/// (invariant 14) and order independence to hold; the identity names which one it was.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifierIdentity {
    name: String,
    config_digest: Digest,
}

impl VerifierIdentity {
    /// A name of 1 to [`MAX_VERIFIER_NAME_BYTES`] bytes and a digest of the verifier's
    /// configuration.
    pub fn new(name: &str, config_digest: Digest) -> Option<Self> {
        if name.trim().is_empty() || name.len() > MAX_VERIFIER_NAME_BYTES {
            return None;
        }
        Some(Self {
            name: name.to_owned(),
            config_digest,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config_digest(&self) -> &Digest {
        &self.config_digest
    }
}

/// Coarse class of a [`RefusalReason`], used for counting.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum RefusalKind {
    Stale,
    AuthorRevoked,
    NotPassed,
    HardFinding,
    BelowRequiredLevel,
    Duplicate,
    AuthorSlotCap,
    Outbid,
}

impl RefusalKind {
    pub const ALL: [RefusalKind; 8] = [
        Self::Stale,
        Self::AuthorRevoked,
        Self::NotPassed,
        Self::HardFinding,
        Self::BelowRequiredLevel,
        Self::Duplicate,
        Self::AuthorSlotCap,
        Self::Outbid,
    ];

    pub(super) fn index(self) -> usize {
        match self {
            Self::Stale => 0,
            Self::AuthorRevoked => 1,
            Self::NotPassed => 2,
            Self::HardFinding => 3,
            Self::BelowRequiredLevel => 4,
            Self::Duplicate => 5,
            Self::AuthorSlotCap => 6,
            Self::Outbid => 7,
        }
    }
}

/// Why a proposal did not enter a slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RefusalReason {
    /// Computed against a revision too far behind, or a slot changed since.
    Stale {
        computed_against: WorkspaceRevision,
        current: WorkspaceRevision,
    },
    /// The author's generation is revoked or superseded according to the host registry.
    AuthorRevoked,
    /// The verifier gate refused it.
    Gate(GateRefusal),
    /// The same content digest already occupies a slot or won this cycle.
    Duplicate,
    /// The author already holds its maximum number of slots.
    AuthorSlotCap { cap: usize },
    /// It passed the gate but lost the write competition.
    Outbid { free_slots: usize },
}

impl RefusalReason {
    pub fn kind(&self) -> RefusalKind {
        match self {
            Self::Stale { .. } => RefusalKind::Stale,
            Self::AuthorRevoked => RefusalKind::AuthorRevoked,
            Self::Gate(GateRefusal::NotPassed(_)) => RefusalKind::NotPassed,
            Self::Gate(GateRefusal::HardFinding) => RefusalKind::HardFinding,
            Self::Gate(GateRefusal::BelowRequiredLevel { .. }) => RefusalKind::BelowRequiredLevel,
            Self::Duplicate => RefusalKind::Duplicate,
            Self::AuthorSlotCap { .. } => RefusalKind::AuthorSlotCap,
            Self::Outbid { .. } => RefusalKind::Outbid,
        }
    }

    fn encode(&self, e: &mut Encoder) {
        e.u8(self.kind().index() as u8);
        match self {
            Self::Stale {
                computed_against,
                current,
            } => {
                e.u64(computed_against.0);
                e.u64(current.0);
            }
            Self::Gate(GateRefusal::NotPassed(status)) => e.u8(status_code(*status)),
            Self::Gate(GateRefusal::BelowRequiredLevel { required, reached }) => {
                e.u8(level_rank(*required));
                e.u8(level_rank(*reached));
            }
            Self::AuthorSlotCap { cap } => e.u64(*cap as u64),
            Self::Outbid { free_slots } => e.u64(*free_slots as u64),
            Self::AuthorRevoked | Self::Duplicate | Self::Gate(_) => {}
        }
    }
}

/// A verifier finding as broadcast: bounded code, hard flag, and the message only under
/// [`FindingVisibility::Public`] (invariant 13).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FindingRecord {
    code: String,
    hard: bool,
    detail: Option<String>,
}

impl FindingRecord {
    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn hard(&self) -> bool {
        self.hard
    }

    /// The bounded message, present only when the workspace publishes finding messages. Untrusted
    /// text: it may echo proposal-derived content.
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    pub(super) fn from_finding(finding: &Finding, visibility: FindingVisibility) -> Self {
        Self {
            code: bounded_code(&finding.code),
            hard: finding.hard,
            detail: match visibility {
                FindingVisibility::HostOnly => None,
                FindingVisibility::Public => Some(bounded_message(&finding.message)),
            },
        }
    }

    fn encode(&self, e: &mut Encoder) {
        e.bytes(self.code.as_bytes());
        e.u8(u8::from(self.hard));
        match &self.detail {
            None => e.u8(0),
            Some(d) => {
                e.u8(1);
                e.bytes(d.as_bytes());
            }
        }
    }
}

/// A finding with its message, handed to the host only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostFinding {
    pub proposal: WorkspaceProposalId,
    pub code: String,
    pub message: String,
    pub hard: bool,
}

pub(super) fn bounded_code(code: &str) -> String {
    let cleaned: String = code
        .chars()
        .take(MAX_FINDING_CODE_BYTES)
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "unnamed".to_owned()
    } else {
        cleaned
    }
}

pub(super) fn bounded_message(message: &str) -> String {
    let mut end = message.len().min(MAX_FINDING_MESSAGE_BYTES);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message[..end].to_owned()
}

/// Findings of one report in broadcast form: hard first (report order within each group), at
/// most [`MAX_FINDINGS_PER_REFUSAL`], and how many were left out.
pub(super) fn records(
    findings: &[Finding],
    visibility: FindingVisibility,
) -> (Vec<FindingRecord>, usize) {
    let mut ordered: Vec<&Finding> = findings.iter().collect();
    ordered.sort_by_key(|f| !f.hard);
    let dropped = ordered.len().saturating_sub(MAX_FINDINGS_PER_REFUSAL);
    let kept = ordered
        .into_iter()
        .take(MAX_FINDINGS_PER_REFUSAL)
        .map(|f| FindingRecord::from_finding(f, visibility))
        .collect();
    (kept, dropped)
}

/// Digest of a verifier report for audit: status, level, score bits and finding codes with hard
/// flags. Messages are excluded on purpose: a hash of a secret-bearing message is still a leak
/// oracle for guessable values.
pub(super) fn report_digest(report: &VerificationReport) -> Digest {
    let mut e = Encoder::new(REPORT_DOMAIN);
    e.u8(status_code(report.status));
    e.u8(level_rank(report.level));
    e.u64(u64::from(report.score.get().to_bits()));
    e.u64(report.findings.len() as u64);
    for f in &report.findings {
        e.bytes(f.code.as_bytes());
        e.u8(u8::from(f.hard));
    }
    e.finish()
}

/// The explicit record of a proposal that did not enter a slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Refusal {
    pub(super) proposal: WorkspaceProposalId,
    pub(super) author: PodId,
    pub(super) author_generation: Generation,
    pub(super) content: Digest,
    pub(super) reason: RefusalReason,
    pub(super) findings: Vec<FindingRecord>,
    pub(super) findings_dropped: usize,
}

impl Refusal {
    pub fn proposal(&self) -> WorkspaceProposalId {
        self.proposal
    }

    pub fn author(&self) -> &PodId {
        &self.author
    }

    pub fn author_generation(&self) -> Generation {
        self.author_generation
    }

    pub fn content(&self) -> &Digest {
        &self.content
    }

    pub fn reason(&self) -> &RefusalReason {
        &self.reason
    }

    /// Findings the verifier raised, hard ones first. Empty when no verifier ran (a stale or
    /// revoked-author proposal).
    pub fn findings(&self) -> &[FindingRecord] {
        &self.findings
    }

    /// Findings beyond the per-refusal or per-snapshot bound that were left out.
    pub fn findings_dropped(&self) -> usize {
        self.findings_dropped
    }
}

/// What occupies a slot. Provisional workspace state, see [`Authority`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlotContent {
    pub(super) slot: SlotIndex,
    pub(super) origin: Proposal,
    pub(super) bid: Bid,
    pub(super) level: VerificationLevel,
    pub(super) admitted_at: WorkspaceRevision,
    pub(super) report_digest: Digest,
    pub(super) findings: Vec<FindingRecord>,
}

impl SlotContent {
    pub fn slot(&self) -> SlotIndex {
        self.slot
    }

    pub fn authority(&self) -> Authority {
        Authority::Provisional
    }

    pub fn proposal(&self) -> WorkspaceProposalId {
        self.origin.id
    }

    pub fn author(&self) -> &PodId {
        &self.origin.author
    }

    pub fn author_generation(&self) -> Generation {
        self.origin.author_generation
    }

    pub fn content(&self) -> &Digest {
        &self.origin.content
    }

    pub fn payload(&self) -> &PayloadRef {
        &self.origin.payload
    }

    /// The effective bid it won with (after clamping).
    pub fn bid(&self) -> Bid {
        self.bid
    }

    /// The verification level the admitting report reached.
    pub fn level(&self) -> VerificationLevel {
        self.level
    }

    /// The revision of the snapshot that first held this content.
    pub fn admitted_at(&self) -> WorkspaceRevision {
        self.admitted_at
    }

    /// Digest of the admitting report (see the module notes on what it covers).
    pub fn report_digest(&self) -> &Digest {
        &self.report_digest
    }

    /// Soft findings of the admitting report.
    pub fn findings(&self) -> &[FindingRecord] {
        &self.findings
    }

    fn encode(&self, e: &mut Encoder) {
        e.u64(self.slot.get() as u64);
        e.raw(&self.origin.id.0);
        e.bytes(self.origin.author.0.as_bytes());
        e.u64(self.origin.author_generation.0);
        e.raw(&self.origin.content);
        e.bytes(self.origin.payload.type_id.0.as_bytes());
        e.raw(&self.origin.payload.digest);
        e.u64(u64::from(self.bid.get()));
        e.u8(level_rank(self.level));
        e.u64(self.admitted_at.0);
        e.raw(&self.report_digest);
        e.u64(self.findings.len() as u64);
        for f in &self.findings {
            f.encode(e);
        }
    }
}

/// Why content left a slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RetireReason {
    /// The residency limit elapsed.
    Aged,
    /// Requested through [`Workspace::request_retire`](super::Workspace::request_retire).
    Explicit,
    /// The author's generation was revoked or superseded.
    AuthorRevoked,
    /// Revalidation failed the gate: a later contradiction.
    Contradicted,
    /// A better challenger took the slot under the displacement policy.
    Displaced { by: WorkspaceProposalId },
}

impl RetireReason {
    fn encode(&self, e: &mut Encoder) {
        match self {
            Self::Aged => e.u8(0),
            Self::Explicit => e.u8(1),
            Self::AuthorRevoked => e.u8(2),
            Self::Contradicted => e.u8(3),
            Self::Displaced { by } => {
                e.u8(4);
                e.raw(&by.0);
            }
        }
    }
}

/// Content that left a slot, kept so retirement is never silent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetiredRecord {
    pub(super) content: SlotContent,
    pub(super) reason: RetireReason,
    pub(super) retired_at: WorkspaceRevision,
    pub(super) findings: Vec<FindingRecord>,
}

impl RetiredRecord {
    pub fn content(&self) -> &SlotContent {
        &self.content
    }

    pub fn reason(&self) -> RetireReason {
        self.reason
    }

    /// The revision of the snapshot that first reported the retirement.
    pub fn retired_at(&self) -> WorkspaceRevision {
        self.retired_at
    }

    /// Findings that caused the retirement (for [`RetireReason::Contradicted`]).
    pub fn findings(&self) -> &[FindingRecord] {
        &self.findings
    }
}

/// The immutable, versioned view every subscriber receives after a cycle. Its digest covers
/// everything it exposes, so two snapshots with equal digests are equal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceSnapshot {
    pub(super) revision: WorkspaceRevision,
    pub(super) slots: Vec<Option<SlotContent>>,
    pub(super) admitted: Vec<WorkspaceProposalId>,
    pub(super) refusals: Vec<Refusal>,
    pub(super) retired: Vec<RetiredRecord>,
    pub(super) verifier: VerifierIdentity,
    pub(super) digest: Digest,
}

impl WorkspaceSnapshot {
    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn authority(&self) -> Authority {
        Authority::Provisional
    }

    /// One entry per slot, in slot order; `None` is a free slot.
    pub fn slots(&self) -> &[Option<SlotContent>] {
        &self.slots
    }

    /// Proposals admitted by the cycle that produced this snapshot, in assignment order.
    pub fn admitted(&self) -> &[WorkspaceProposalId] {
        &self.admitted
    }

    /// Every proposal this cycle judged and did not admit, in proposal-id order. This is where
    /// verifier findings (codes) are broadcast.
    pub fn refusals(&self) -> &[Refusal] {
        &self.refusals
    }

    /// Content retired by this cycle.
    pub fn retired(&self) -> &[RetiredRecord] {
        &self.retired
    }

    /// The verifier that judged the cycle.
    pub fn verifier(&self) -> &VerifierIdentity {
        &self.verifier
    }

    pub fn occupied(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    /// SHA-256 over a canonical encoding of the whole snapshot.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }
}

pub(super) struct SnapshotParts {
    pub revision: WorkspaceRevision,
    pub slots: Vec<Option<SlotContent>>,
    pub admitted: Vec<WorkspaceProposalId>,
    pub refusals: Vec<Refusal>,
    pub retired: Vec<RetiredRecord>,
    pub verifier: VerifierIdentity,
}

pub(super) fn build_snapshot(parts: SnapshotParts) -> WorkspaceSnapshot {
    let SnapshotParts {
        revision,
        slots,
        admitted,
        mut refusals,
        retired,
        verifier,
    } = parts;
    // Bound the findings of the whole snapshot; what is cut is counted, not hidden.
    let mut budget = MAX_SNAPSHOT_FINDINGS;
    for refusal in &mut refusals {
        if refusal.findings.len() > budget {
            refusal.findings_dropped += refusal.findings.len() - budget;
            refusal.findings.truncate(budget);
        }
        budget -= refusal.findings.len();
    }
    let mut e = Encoder::new(SNAPSHOT_DOMAIN);
    e.u8(0); // Authority::Provisional
    e.u64(revision.0);
    e.bytes(verifier.name.as_bytes());
    e.raw(&verifier.config_digest);
    e.u64(slots.len() as u64);
    for slot in &slots {
        match slot {
            None => e.u8(0),
            Some(content) => {
                e.u8(1);
                content.encode(&mut e);
            }
        }
    }
    e.u64(admitted.len() as u64);
    for id in &admitted {
        e.raw(&id.0);
    }
    e.u64(refusals.len() as u64);
    for r in &refusals {
        e.raw(&r.proposal.0);
        e.bytes(r.author.0.as_bytes());
        e.u64(r.author_generation.0);
        e.raw(&r.content);
        r.reason.encode(&mut e);
        e.u64(r.findings.len() as u64);
        for f in &r.findings {
            f.encode(&mut e);
        }
        e.u64(r.findings_dropped as u64);
    }
    e.u64(retired.len() as u64);
    for record in &retired {
        record.content.encode(&mut e);
        record.reason.encode(&mut e);
        e.u64(record.retired_at.0);
        e.u64(record.findings.len() as u64);
        for f in &record.findings {
            f.encode(&mut e);
        }
    }
    let digest = e.finish();
    WorkspaceSnapshot {
        revision,
        slots,
        admitted,
        refusals,
        retired,
        verifier,
        digest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finding_messages_are_cut_on_a_char_boundary() {
        let m = bounded_message(&"é".repeat(MAX_FINDING_MESSAGE_BYTES));
        assert!(m.len() <= MAX_FINDING_MESSAGE_BYTES);
        assert!(m.chars().all(|c| c == 'é'));
    }

    #[test]
    fn finding_codes_are_bounded_and_sanitised() {
        let long = bounded_code(&"a".repeat(10_000));
        assert_eq!(long.len(), MAX_FINDING_CODE_BYTES);
        assert_eq!(bounded_code("a b/é"), "a_b__");
        assert_eq!(bounded_code(""), "unnamed");
    }

    #[test]
    fn host_only_records_carry_no_message() {
        let f = Finding {
            code: "c".into(),
            message: "SECRET".into(),
            hard: true,
        };
        let hidden = FindingRecord::from_finding(&f, FindingVisibility::HostOnly);
        assert!(hidden.detail().is_none());
        assert!(!format!("{hidden:?}").contains("SECRET"));
        let shown = FindingRecord::from_finding(&f, FindingVisibility::Public);
        assert_eq!(shown.detail(), Some("SECRET"));
    }
}
