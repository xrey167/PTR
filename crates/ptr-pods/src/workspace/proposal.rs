//! Identifiers, proposals and the canonical encoder.

use ptr_types::{Digest, Generation, PodId, TypeId};
use sha2::{Digest as ShaDigest, Sha256};
use std::fmt;

const PROPOSAL_DOMAIN: &[u8] = b"ptr.pods.workspace.proposal.v2";

/// The workspace revision: starts at 0 and advances by exactly one per cycle, never back. It has
/// no public constructor: read it from [`Workspace::revision`](super::Workspace::revision) or a
/// snapshot, so it cannot be mistaken for, or forged as, a ledger revision.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WorkspaceRevision(pub(super) u64);

impl WorkspaceRevision {
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Write priority a proposal asks for. It is a request: the effective bid is clamped by the
/// workspace's `max_bid` and by the host registry's per-author ceiling, and it orders only
/// proposals that already passed the verifier gate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Bid(u32);

impl Bid {
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// Index of a slot, `0..slots`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SlotIndex(usize);

impl SlotIndex {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn get(self) -> usize {
        self.0
    }
}

/// Typed reference to a payload held elsewhere: the type and the digest of its bytes. The
/// workspace never holds payload bytes, which keeps its state bounded.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PayloadRef {
    pub(super) type_id: TypeId,
    pub(super) digest: Digest,
}

impl PayloadRef {
    pub fn new(type_id: TypeId, digest: Digest) -> Self {
        Self { type_id, digest }
    }

    pub fn type_id(&self) -> &TypeId {
        &self.type_id
    }

    pub fn digest(&self) -> &Digest {
        &self.digest
    }
}

/// Deterministic identity of a proposal: SHA-256 over every field, domain separated.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WorkspaceProposalId(pub(super) Digest);

impl WorkspaceProposalId {
    pub fn as_bytes(&self) -> &Digest {
        &self.0
    }
}

/// Identity of a subscription, minted by [`Workspace::subscribe`](super::Workspace::subscribe).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SubscriberId(pub(super) u64);

impl SubscriberId {
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Unvalidated proposal fields; see [`Proposal::new`]. There is deliberately no field for a
/// verification report, level, threshold or verifier: the workspace's host decides all of those.
#[derive(Clone, Debug)]
pub struct ProposalInit {
    pub author: PodId,
    pub author_generation: Generation,
    pub content: Digest,
    pub payload: PayloadRef,
    pub computed_against: WorkspaceRevision,
    pub bid: Bid,
}

/// A rejected [`ProposalInit`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkspaceProposalError {
    BlankAuthor,
    ZeroContent,
    ZeroPayloadDigest,
    BlankPayloadType,
}

impl fmt::Display for WorkspaceProposalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BlankAuthor => "author pod id is blank",
            Self::ZeroContent => "content digest is all zero",
            Self::ZeroPayloadDigest => "payload digest is all zero",
            Self::BlankPayloadType => "payload type id is blank",
        })
    }
}

impl std::error::Error for WorkspaceProposalError {}

/// A Pod's request to write into the workspace. Not authoritative: it waits in the pending area
/// until a cycle judges it.
///
/// The host verifier sees all of these fields, and the author, payload type and content are
/// chosen by the proposer. A host verifier that dispatches on the payload type must treat an
/// unknown type as `Unknown` or `Fail`, never as a weaker check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Proposal {
    pub(super) author: PodId,
    pub(super) author_generation: Generation,
    pub(super) content: Digest,
    pub(super) payload: PayloadRef,
    pub(super) computed_against: WorkspaceRevision,
    pub(super) bid: Bid,
    pub(super) id: WorkspaceProposalId,
}

impl Proposal {
    pub fn new(init: ProposalInit) -> Result<Self, WorkspaceProposalError> {
        if init.author.0.trim().is_empty() {
            return Err(WorkspaceProposalError::BlankAuthor);
        }
        if init.content == [0; 32] {
            return Err(WorkspaceProposalError::ZeroContent);
        }
        if init.payload.digest == [0; 32] {
            return Err(WorkspaceProposalError::ZeroPayloadDigest);
        }
        if init.payload.type_id.0.trim().is_empty() {
            return Err(WorkspaceProposalError::BlankPayloadType);
        }
        let mut e = Encoder::new(PROPOSAL_DOMAIN);
        e.bytes(init.author.0.as_bytes());
        e.u64(init.author_generation.0);
        e.raw(&init.content);
        e.bytes(init.payload.type_id.0.as_bytes());
        e.raw(&init.payload.digest);
        e.u64(init.computed_against.0);
        e.u64(u64::from(init.bid.0));
        let id = WorkspaceProposalId(e.finish());
        Ok(Self {
            author: init.author,
            author_generation: init.author_generation,
            content: init.content,
            payload: init.payload,
            computed_against: init.computed_against,
            bid: init.bid,
            id,
        })
    }

    pub fn id(&self) -> WorkspaceProposalId {
        self.id
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

    pub fn payload(&self) -> &PayloadRef {
        &self.payload
    }

    pub fn computed_against(&self) -> WorkspaceRevision {
        self.computed_against
    }

    /// The bid the proposer asked for; the effective bid may be lower.
    pub fn bid(&self) -> Bid {
        self.bid
    }
}

/// Length-prefixed canonical encoder hashed with SHA-256.
pub(super) struct Encoder(Sha256);

impl Encoder {
    pub(super) fn new(domain: &[u8]) -> Self {
        let mut e = Self(Sha256::new());
        e.bytes(domain);
        e
    }

    pub(super) fn u8(&mut self, v: u8) {
        self.0.update([v]);
    }

    pub(super) fn u64(&mut self, v: u64) {
        self.0.update(v.to_le_bytes());
    }

    pub(super) fn raw(&mut self, v: &[u8]) {
        self.0.update(v);
    }

    pub(super) fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.0.update(v);
    }

    pub(super) fn finish(self) -> Digest {
        self.0.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init(tag: u8, bid: u32) -> ProposalInit {
        ProposalInit {
            author: PodId::from("p"),
            author_generation: Generation(1),
            content: [tag; 32],
            payload: PayloadRef::new(TypeId::from("t"), [tag; 32]),
            computed_against: WorkspaceRevision(0),
            bid: Bid::new(bid),
        }
    }

    #[test]
    fn proposal_id_changes_with_every_field() {
        let base = Proposal::new(init(1, 5)).unwrap().id();
        assert_eq!(base, Proposal::new(init(1, 5)).unwrap().id());
        assert_ne!(base, Proposal::new(init(2, 5)).unwrap().id());
        assert_ne!(base, Proposal::new(init(1, 6)).unwrap().id());
        let mut other = init(1, 5);
        other.computed_against = WorkspaceRevision(1);
        assert_ne!(base, Proposal::new(other).unwrap().id());
        let mut other = init(1, 5);
        other.author_generation = Generation(2);
        assert_ne!(base, Proposal::new(other).unwrap().id());
    }

    #[test]
    fn proposal_validation_rejects_blank_and_zero_fields() {
        assert!(Proposal::new(init(1, 0)).is_ok());
        let mut a = init(1, 0);
        a.author = PodId::from(" ");
        assert_eq!(
            Proposal::new(a).unwrap_err(),
            WorkspaceProposalError::BlankAuthor
        );
        let mut c = init(1, 0);
        c.content = [0; 32];
        assert_eq!(
            Proposal::new(c).unwrap_err(),
            WorkspaceProposalError::ZeroContent
        );
        let mut d = init(1, 0);
        d.payload = PayloadRef::new(TypeId::from("t"), [0; 32]);
        assert_eq!(
            Proposal::new(d).unwrap_err(),
            WorkspaceProposalError::ZeroPayloadDigest
        );
        let mut t = init(1, 0);
        t.payload = PayloadRef::new(TypeId::from(""), [1; 32]);
        assert_eq!(
            Proposal::new(t).unwrap_err(),
            WorkspaceProposalError::BlankPayloadType
        );
    }
}
