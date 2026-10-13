//! Exact-source resolution and deterministic claim verification: the two steps that
//! stand between a search hit and `VerifiedKnown` (invariant 6, evidence promotion).
//!
//! A retrieval candidate names a source: a raw event, a byte span of it and the text it
//! says it quotes ([`SourceCitation`]). Nothing the retriever says is evidence. The chain
//! is:
//!
//! 1. [`ExactSourceResolver::resolve`] checks the citation against the raw event the
//!    store holds: the event exists, its digest still matches its bytes, the span lies
//!    inside it and the span's bytes equal the quote, byte for byte. Only then does it
//!    hand out a [`ResolvedSource`], which also records which capsule generations the
//!    event belongs to ([`SourceOwnership`]).
//! 2. [`SearchHit::observe_resolved`](crate::SearchHit::observe_resolved) takes that
//!    proof, asks the lifecycle authority about the hit's own generation, checks that the
//!    event belongs to that very capsule generation and moves the hit to `Observed`. A
//!    hit labelled with a live generation that cites an event of a superseded or revoked
//!    one is refused with `SourceNotOwnedByHit`.
//! 3. [`verify_claim`] compares a structured [`Claim`] with the resolved text under the
//!    statement format below and hands out a [`ClaimProof`].
//! 4. [`SearchHit::promote_verified_with`](crate::SearchHit::promote_verified_with) takes
//!    that proof, checks that it is about the very source the hit was observed with and
//!    moves the hit to `VerifiedKnown`.
//!
//! [`ResolvedSource`] and [`ClaimProof`] have private fields and no public constructor:
//! outside this crate a value of either type exists only because the resolver or the
//! verifier produced it, so a caller cannot pass `true` for a step it did not take.
//!
//! ```compile_fail,E0451
//! // A resolution cannot be written down by hand: its fields are private.
//! use ptr_search::ResolvedSource;
//! use ptr_types::RawEventId;
//! let _forged = ResolvedSource {
//!     raw_id: RawEventId::from("raw-1"),
//!     start: 0,
//!     end: 4,
//!     digest: [0; 32],
//!     text: "forged".to_owned(),
//!     owners: Vec::new(),
//! };
//! ```
//!
//! ```compile_fail,E0451
//! // Neither can a claim proof.
//! use ptr_search::{ClaimProof, Claim};
//! let _forged = ClaimProof {
//!     source: todo!(),
//!     claim: Claim::new("a", "b", "c"),
//! };
//! ```
//!
//! The same imports compile when the value comes from the resolver and the verifier:
//!
//! ```
//! use ptr_memory::{KnowledgeStore, RawEvent};
//! use ptr_search::{verify_claim, Claim, ExactSourceResolver, SourceCitation, SourceOwners};
//! use ptr_types::{CapsuleId, EventRole, Generation, RawEventId, SessionId, Timestamp};
//!
//! let text = "ent-1: status is active";
//! let mut store = KnowledgeStore::default();
//! store
//!     .append_raw(RawEvent::new(
//!         RawEventId::from("raw-1"),
//!         SessionId::from("s"),
//!         EventRole::User,
//!         text.as_bytes().to_vec(),
//!         Timestamp(1),
//!     ))
//!     .unwrap();
//! let mut owners = SourceOwners::default();
//! owners.record(RawEventId::from("raw-1"), CapsuleId::from("cap-1"), Generation(1));
//! let citation = SourceCitation {
//!     raw_id: RawEventId::from("raw-1"),
//!     start: 0,
//!     end: text.len(),
//!     quote: text.to_owned(),
//! };
//! let resolved = ExactSourceResolver::new(&store, &owners).resolve(&citation).unwrap();
//! let proof = verify_claim(&Claim::new("ent-1", "status", "active"), &resolved).unwrap();
//! assert_eq!(proof.claim().value, "active");
//! ```
//!
//! # Statement format
//!
//! The resolved text must be exactly one statement, and nothing else:
//!
//! ```text
//! statement := subject ": " predicate " is " [ "not " ] value
//! subject, predicate, value := token
//! token := one or more characters, none of them whitespace
//! ```
//!
//! So `ent-0012: status is active` states that subject `ent-0012` has the value `active`
//! for the predicate `status`, and `ent-0012: status is not active` states the contrary.
//! The text is split on single ASCII spaces into exactly `subject:`, `predicate`, `is`,
//! an optional `not` and the value (four or five parts). The bare value `not`
//! (`ent-1: status is not`) is refused as malformed: it is indistinguishable from a
//! negation cut off before its value, so a truncated span can never verify. Anything else is malformed:
//! leading or trailing space or newline, a doubled space, a missing colon, a second
//! statement. A trailing full stop is part of the value token (`active.` is not
//! `active`). Tokens are compared byte for byte, with no case
//! folding and no Unicode normalisation, so a subject that only looks like the claimed
//! one (`ent-O012` for `ent-0012`, a Cyrillic letter for a Latin one) is a different
//! subject. A [`Claim`] whose own fields are not tokens can match no statement and is
//! refused as malformed.
//!
//! The verifier checks, in this order: the claim is well formed, the text is a
//! statement, the subject, the predicate and the value are equal to the claim's, and the
//! statement is not negated. Each refusal is a [`ClaimError`] variant.
//!
//! # Trust roots
//!
//! The proof says that *this claim is what this span of this intact raw event states*,
//! and the resolution records which capsule generations the event belongs to. Two things
//! are taken from the caller and are the trust roots of the whole chain:
//!
//! - the [`RawEventSource`]: the digest check shows that a source is internally
//!   consistent (an event's bytes hash to the digest it carries), not that the source is
//!   the store of record. A caller can implement the trait over fabricated events, so the
//!   trait is open on purpose and the source must be the store that admitted the events.
//!   (Sealing it would not help: the P3 test double of experiment Q002 is such an
//!   implementation, and any build that enables a test-support feature would unseal it for
//!   every crate in the dependency graph.)
//! - the [`SourceOwnership`]: which capsule generation admitted which raw event. It must
//!   come from the record that admitted the events (for a knowledge object, its
//!   `sources`), never from the retriever's index, which is the thing under suspicion.
//!
//! Given truthful roots, a hit can be promoted only with a source that belongs to the
//! capsule generation the hit names, and that generation must be live. The lifecycle
//! closure handed to `observe_resolved` is a root as well: the guarantee about
//! superseded, revoked and disputed generations holds exactly as far as it answers
//! truthfully, and nothing here checks it. The claim itself is the caller's: the hit
//! carries no assertion of its own that the proof is held against.
//!
//! It does not judge free text (a paraphrase, a statement in another format, a span that
//! holds several statements are all refused, never interpreted) and it does not decide
//! which of two live statements that disagree is right.

use std::fmt;

use ptr_memory::{KnowledgeStore, RawEvent};
use ptr_types::{CapsuleId, Generation, RawEventId};
use std::collections::BTreeMap;

/// Where the resolver reads raw events from. [`KnowledgeStore`] is the store of record;
/// the trait lets a test or an experiment hand the resolver a source that breaks the
/// store's own invariants (an event whose bytes no longer match its digest), which the
/// resolver must not trust anyway.
pub trait RawEventSource {
    /// The raw event with this id, as the source holds it.
    fn raw_event(&self, id: &RawEventId) -> Option<&RawEvent>;
}

impl RawEventSource for KnowledgeStore {
    fn raw_event(&self, id: &RawEventId) -> Option<&RawEvent> {
        self.raw(id)
    }
}

/// Which capsule generations a raw event belongs to: the record that admitted the event.
/// An event may belong to several (a later generation can cite the sources of an earlier
/// one) and to none (an event nothing admitted). It must be the admitting record, never the
/// retriever's index: a hit is promotable only with an event that belongs to the capsule
/// generation the hit names.
pub trait SourceOwnership {
    /// The capsule generations the raw event with this id belongs to; empty when none does.
    fn owners(&self, id: &RawEventId) -> Vec<(CapsuleId, Generation)>;
}

/// A [`SourceOwnership`] kept as a map; fill it from the records that admitted the events.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SourceOwners {
    owners: BTreeMap<RawEventId, Vec<(CapsuleId, Generation)>>,
}

impl SourceOwners {
    /// Record that the raw event `id` belongs to `generation` of `capsule`. Recording the
    /// same triple twice changes nothing.
    pub fn record(&mut self, id: RawEventId, capsule: CapsuleId, generation: Generation) {
        let entry = self.owners.entry(id).or_default();
        let owner = (capsule, generation);
        if !entry.contains(&owner) {
            entry.push(owner);
        }
    }
}

impl SourceOwnership for SourceOwners {
    fn owners(&self, id: &RawEventId) -> Vec<(CapsuleId, Generation)> {
        self.owners.get(id).cloned().unwrap_or_default()
    }
}

/// What a retriever claims to quote: bytes `start..end` of a raw event, said to equal
/// `quote`. Plain data; the resolver is what checks it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceCitation {
    pub raw_id: RawEventId,
    /// Byte offset of the first byte of the span.
    pub start: usize,
    /// Byte offset one past the last byte of the span.
    pub end: usize,
    /// The text the retriever says is at `start..end`.
    pub quote: String,
}

/// Why a citation did not resolve. Each reason has its own variant; none carries source
/// text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceError {
    /// The source holds no raw event with this id.
    UnknownRawEvent { raw_id: RawEventId },
    /// The raw event's bytes no longer match the digest it carries.
    DigestMismatch { raw_id: RawEventId },
    /// The span holds no byte (`start >= end`).
    EmptySpan { start: usize, end: usize },
    /// The span ends beyond the raw event's last byte.
    SpanOutOfRange {
        start: usize,
        end: usize,
        len: usize,
    },
    /// The bytes at the span are not the quote.
    QuoteMismatch {
        raw_id: RawEventId,
        start: usize,
        end: usize,
    },
}

impl SourceError {
    /// A stable code for each refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnknownRawEvent { .. } => "PTR_SEARCH_UNKNOWN_RAW_EVENT",
            Self::DigestMismatch { .. } => "PTR_SEARCH_RAW_DIGEST_MISMATCH",
            Self::EmptySpan { .. } => "PTR_SEARCH_EMPTY_SPAN",
            Self::SpanOutOfRange { .. } => "PTR_SEARCH_SPAN_OUT_OF_RANGE",
            Self::QuoteMismatch { .. } => "PTR_SEARCH_QUOTE_MISMATCH",
        }
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownRawEvent { raw_id } => write!(formatter, "no raw event {raw_id}"),
            Self::DigestMismatch { raw_id } => {
                write!(formatter, "raw event {raw_id} does not match its digest")
            }
            Self::EmptySpan { start, end } => write!(formatter, "empty span {start}..{end}"),
            Self::SpanOutOfRange { start, end, len } => {
                write!(
                    formatter,
                    "span {start}..{end} exceeds the {len} bytes of the event"
                )
            }
            Self::QuoteMismatch {
                raw_id, start, end, ..
            } => write!(
                formatter,
                "the bytes {start}..{end} of raw event {raw_id} are not the quote"
            ),
        }
    }
}

impl std::error::Error for SourceError {}

/// A citation the resolver confirmed against an intact raw event: the proof that
/// `text` is, byte for byte, what `start..end` of `raw_id` holds. Fields are private
/// and there is no public constructor, so only [`ExactSourceResolver::resolve`] makes
/// one. It also records the capsule generations the event belongs to, as the resolver's
/// [`SourceOwnership`] answered when it was made.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedSource {
    raw_id: RawEventId,
    start: usize,
    end: usize,
    digest: [u8; 32],
    text: String,
    owners: Vec<(CapsuleId, Generation)>,
}

impl ResolvedSource {
    pub fn raw_id(&self) -> &RawEventId {
        &self.raw_id
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    /// The digest of the whole raw event the span was read from.
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The text of the span.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The capsule generations the raw event belongs to; empty when nothing admitted it.
    pub fn owners(&self) -> &[(CapsuleId, Generation)] {
        &self.owners
    }

    /// Whether the raw event belongs to exactly this capsule and this generation.
    pub fn is_owned_by(&self, capsule: &CapsuleId, generation: Generation) -> bool {
        self.owners
            .iter()
            .any(|(owner, owned)| owner == capsule && *owned == generation)
    }
}

/// Resolves citations against the raw events of a source (by default a
/// [`KnowledgeStore`]), re-checking each event's digest.
#[derive(Clone, Copy, Debug)]
pub struct ExactSourceResolver<
    'store,
    S: RawEventSource + ?Sized = KnowledgeStore,
    O: SourceOwnership + ?Sized = SourceOwners,
> {
    source: &'store S,
    ownership: &'store O,
}

impl<'store, S: RawEventSource + ?Sized, O: SourceOwnership + ?Sized>
    ExactSourceResolver<'store, S, O>
{
    /// A resolver over `source`, recording ownership as `ownership` answers.
    pub fn new(source: &'store S, ownership: &'store O) -> Self {
        Self { source, ownership }
    }

    /// Confirm `citation` against the raw event it names.
    ///
    /// The checks run in this order and the first failure is returned: the event exists,
    /// its bytes still hash to its digest (the store checks this when it appends, the
    /// resolver does not rely on that), the span is not empty and lies inside the event,
    /// and the span's bytes equal the quote exactly. The offset is part of the citation:
    /// a quote that occurs elsewhere in the event does not resolve at a span that does
    /// not hold it. The capsule generations the event belongs to are recorded in the
    /// result; an event nothing admitted still resolves, with no owner, and no hit can
    /// then be observed with it.
    ///
    /// # Errors
    /// `UnknownRawEvent`, `DigestMismatch`, `EmptySpan`, `SpanOutOfRange` or
    /// `QuoteMismatch`, in that order of precedence.
    pub fn resolve(&self, citation: &SourceCitation) -> Result<ResolvedSource, SourceError> {
        let event = self.source.raw_event(&citation.raw_id).ok_or_else(|| {
            SourceError::UnknownRawEvent {
                raw_id: citation.raw_id.clone(),
            }
        })?;
        event
            .validate_digest()
            .map_err(|_| SourceError::DigestMismatch {
                raw_id: citation.raw_id.clone(),
            })?;
        if citation.start >= citation.end {
            return Err(SourceError::EmptySpan {
                start: citation.start,
                end: citation.end,
            });
        }
        if citation.end > event.content.len() {
            return Err(SourceError::SpanOutOfRange {
                start: citation.start,
                end: citation.end,
                len: event.content.len(),
            });
        }
        if event.content[citation.start..citation.end] != *citation.quote.as_bytes() {
            return Err(SourceError::QuoteMismatch {
                raw_id: citation.raw_id.clone(),
                start: citation.start,
                end: citation.end,
            });
        }
        Ok(ResolvedSource {
            raw_id: citation.raw_id.clone(),
            start: citation.start,
            end: citation.end,
            digest: event.digest,
            text: citation.quote.clone(),
            owners: self.ownership.owners(&citation.raw_id),
        })
    }
}

/// What a hit claims: `subject` has `value` for `predicate`. Plain data; see the module
/// documentation for the statement format a source must state it in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Claim {
    pub subject: String,
    pub predicate: String,
    pub value: String,
}

impl Claim {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            value: value.into(),
        }
    }
}

/// Why a claim was not verified against a resolved source. No variant carries text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimError {
    /// A field of the claim is not a token (empty or holding whitespace), so no
    /// statement can state it.
    MalformedClaim,
    /// The resolved text is not exactly one statement of the format.
    MalformedStatement,
    /// The statement is about another subject, however alike the names look.
    SubjectMismatch,
    /// The statement is about another predicate.
    PredicateMismatch,
    /// The statement gives another value.
    ValueMismatch,
    /// The statement says the claim does not hold.
    NegatedStatement,
}

impl ClaimError {
    /// A stable code for each refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::MalformedClaim => "PTR_SEARCH_MALFORMED_CLAIM",
            Self::MalformedStatement => "PTR_SEARCH_MALFORMED_STATEMENT",
            Self::SubjectMismatch => "PTR_SEARCH_SUBJECT_MISMATCH",
            Self::PredicateMismatch => "PTR_SEARCH_PREDICATE_MISMATCH",
            Self::ValueMismatch => "PTR_SEARCH_VALUE_MISMATCH",
            Self::NegatedStatement => "PTR_SEARCH_NEGATED_STATEMENT",
        }
    }
}

impl fmt::Display for ClaimError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::MalformedClaim => "a field of the claim is not a token",
            Self::MalformedStatement => "the resolved text is not exactly one statement",
            Self::SubjectMismatch => "the statement is about another subject",
            Self::PredicateMismatch => "the statement is about another predicate",
            Self::ValueMismatch => "the statement gives another value",
            Self::NegatedStatement => "the statement negates the claim",
        })
    }
}

impl std::error::Error for ClaimError {}

/// The proof that a claim is what a resolved source states. Fields are private and there
/// is no public constructor, so only [`verify_claim`] makes one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimProof {
    source: ResolvedSource,
    claim: Claim,
}

impl ClaimProof {
    /// The source the claim was verified against.
    pub fn source(&self) -> &ResolvedSource {
        &self.source
    }

    /// The claim that was verified.
    pub fn claim(&self) -> &Claim {
        &self.claim
    }
}

/// A token: not empty, no whitespace.
fn is_token(token: &str) -> bool {
    !token.is_empty() && !token.chars().any(char::is_whitespace)
}

/// A parsed statement, borrowed from the text.
struct Statement<'text> {
    subject: &'text str,
    predicate: &'text str,
    negated: bool,
    value: &'text str,
}

fn parse_statement(text: &str) -> Result<Statement<'_>, ClaimError> {
    let parts: Vec<&str> = text.split(' ').collect();
    let (subject, predicate, negated, value) = match parts[..] {
        // `x: p is not` is a negation cut off before its value, not the value "not".
        [_, _, "is", "not"] => return Err(ClaimError::MalformedStatement),
        [subject, predicate, "is", value] => (subject, predicate, false, value),
        [subject, predicate, "is", "not", value] => (subject, predicate, true, value),
        _ => return Err(ClaimError::MalformedStatement),
    };
    let subject = subject
        .strip_suffix(':')
        .ok_or(ClaimError::MalformedStatement)?;
    if ![subject, predicate, value].into_iter().all(is_token) {
        return Err(ClaimError::MalformedStatement);
    }
    Ok(Statement {
        subject,
        predicate,
        negated,
        value,
    })
}

/// Compare `claim` with the text of `source` under the statement format of the module
/// documentation.
///
/// Deterministic: the same claim and the same text always give the same answer, and
/// nothing is scored or learned.
///
/// # Errors
/// `MalformedClaim` when a field of the claim is not a token; `MalformedStatement` when
/// the text is not exactly one statement (a bare `not` value included); then
/// `SubjectMismatch`, `PredicateMismatch` and `ValueMismatch` in that order; and `NegatedStatement` when subject, predicate and
/// value are the claim's but the statement is negated.
pub fn verify_claim(claim: &Claim, source: &ResolvedSource) -> Result<ClaimProof, ClaimError> {
    if ![&claim.subject, &claim.predicate, &claim.value]
        .into_iter()
        .all(|field| is_token(field))
    {
        return Err(ClaimError::MalformedClaim);
    }
    let statement = parse_statement(&source.text)?;
    if statement.subject != claim.subject {
        return Err(ClaimError::SubjectMismatch);
    }
    if statement.predicate != claim.predicate {
        return Err(ClaimError::PredicateMismatch);
    }
    if statement.value != claim.value {
        return Err(ClaimError::ValueMismatch);
    }
    if statement.negated {
        return Err(ClaimError::NegatedStatement);
    }
    Ok(ClaimProof {
        source: source.clone(),
        claim: claim.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::{EventRole, SessionId, Timestamp};

    const BODY: &str =
        "ent-0001: status is active\nent-0002: tier is not gold\nent-0003: color is red";

    fn store_with(id: &str, body: &str) -> KnowledgeStore {
        let mut store = KnowledgeStore::default();
        store
            .append_raw(RawEvent::new(
                RawEventId::from(id),
                SessionId::from("s"),
                EventRole::User,
                body.as_bytes().to_vec(),
                Timestamp(1),
            ))
            .expect("a well-formed raw event");
        store
    }

    fn citation(id: &str, start: usize, end: usize, quote: &str) -> SourceCitation {
        SourceCitation {
            raw_id: RawEventId::from(id),
            start,
            end,
            quote: quote.to_owned(),
        }
    }

    /// The ownership the unit tests use: the raw event `r` belongs to generation 1 of `cap`.
    fn owners_of(id: &str) -> SourceOwners {
        let mut owners = SourceOwners::default();
        owners.record(RawEventId::from(id), CapsuleId::from("cap"), Generation(1));
        owners
    }

    fn resolved(text: &str) -> ResolvedSource {
        let store = store_with("r", text);
        let owners = owners_of("r");
        ExactSourceResolver::new(&store, &owners)
            .resolve(&citation("r", 0, text.len(), text))
            .expect("the whole event resolves")
    }

    /// A source that holds an event whose bytes were altered after its digest was taken,
    /// which `KnowledgeStore` itself cannot hold.
    struct Corrupt(RawEvent);

    impl RawEventSource for Corrupt {
        fn raw_event(&self, id: &RawEventId) -> Option<&RawEvent> {
            (id == &self.0.id).then_some(&self.0)
        }
    }

    mod resolver {
        use super::*;

        #[test]
        fn a_span_that_holds_the_quote_resolves_to_its_text() {
            let store = store_with("r", BODY);
            let owners = owners_of("r");
            let resolver = ExactSourceResolver::new(&store, &owners);
            let line = "ent-0002: tier is not gold";
            let start = BODY.find(line).expect("line present");
            let source = resolver
                .resolve(&citation("r", start, start + line.len(), line))
                .expect("resolves");
            assert_eq!(source.text(), line);
            assert_eq!(source.raw_id(), &RawEventId::from("r"));
            assert_eq!((source.start(), source.end()), (start, start + line.len()));
            assert_eq!(source.digest(), &RawEvent::digest_for(BODY.as_bytes()));
        }

        #[test]
        fn the_resolution_records_every_owner_of_the_event_and_none_for_an_unowned_one() {
            let store = store_with("r", BODY);
            let mut owners = owners_of("r");
            owners.record(RawEventId::from("r"), CapsuleId::from("cap"), Generation(1));
            owners.record(RawEventId::from("r"), CapsuleId::from("cap"), Generation(2));
            owners.record(
                RawEventId::from("r"),
                CapsuleId::from("other"),
                Generation(1),
            );
            let source = ExactSourceResolver::new(&store, &owners)
                .resolve(&citation("r", 0, 26, &BODY[..26]))
                .expect("resolves");
            assert_eq!(source.owners().len(), 3);
            assert!(source.is_owned_by(&CapsuleId::from("cap"), Generation(1)));
            assert!(source.is_owned_by(&CapsuleId::from("cap"), Generation(2)));
            assert!(source.is_owned_by(&CapsuleId::from("other"), Generation(1)));
            // Both halves of the pair must match.
            assert!(!source.is_owned_by(&CapsuleId::from("other"), Generation(2)));
            assert!(!source.is_owned_by(&CapsuleId::from("cap"), Generation(3)));
            assert!(!source.is_owned_by(&CapsuleId::from("nobody"), Generation(1)));
            // An event nothing admitted still resolves, with no owner.
            let nobody = SourceOwners::default();
            let orphan = ExactSourceResolver::new(&store, &nobody)
                .resolve(&citation("r", 0, 26, &BODY[..26]))
                .expect("resolves");
            assert!(orphan.owners().is_empty());
            assert!(!orphan.is_owned_by(&CapsuleId::from("cap"), Generation(1)));
        }

        #[test]
        fn an_unknown_raw_event_is_refused() {
            let store = store_with("r", BODY);
            let owners = owners_of("r");
            assert_eq!(
                ExactSourceResolver::new(&store, &owners).resolve(&citation("nope", 0, 4, "ent-")),
                Err(SourceError::UnknownRawEvent {
                    raw_id: RawEventId::from("nope")
                })
            );
        }

        #[test]
        fn an_event_whose_bytes_no_longer_match_its_digest_is_refused_even_if_the_quote_matches() {
            let mut event = RawEvent::new(
                RawEventId::from("r"),
                SessionId::from("s"),
                EventRole::User,
                BODY.as_bytes().to_vec(),
                Timestamp(1),
            );
            event.content[11] = b'X';
            let source = Corrupt(event);
            let tampered = String::from_utf8(source.0.content.clone()).expect("ascii");
            let owners = owners_of("r");
            let outcome = ExactSourceResolver::new(&source, &owners).resolve(&citation(
                "r",
                0,
                26,
                &tampered[..26],
            ));
            assert_eq!(
                outcome,
                Err(SourceError::DigestMismatch {
                    raw_id: RawEventId::from("r")
                })
            );
        }

        #[test]
        fn an_empty_or_reversed_span_is_refused() {
            let store = store_with("r", BODY);
            let owners = owners_of("r");
            let resolver = ExactSourceResolver::new(&store, &owners);
            assert_eq!(
                resolver.resolve(&citation("r", 4, 4, "")),
                Err(SourceError::EmptySpan { start: 4, end: 4 })
            );
            assert_eq!(
                resolver.resolve(&citation("r", 9, 4, "ent-")),
                Err(SourceError::EmptySpan { start: 9, end: 4 })
            );
        }

        #[test]
        fn a_span_past_the_end_is_refused() {
            let store = store_with("r", BODY);
            let owners = owners_of("r");
            let resolver = ExactSourceResolver::new(&store, &owners);
            let past = BODY.len() + 1;
            assert_eq!(
                resolver.resolve(&citation("r", 0, past, BODY)),
                Err(SourceError::SpanOutOfRange {
                    start: 0,
                    end: past,
                    len: BODY.len()
                })
            );
            assert_eq!(
                resolver.resolve(&citation("r", usize::MAX - 1, usize::MAX, "x")),
                Err(SourceError::SpanOutOfRange {
                    start: usize::MAX - 1,
                    end: usize::MAX,
                    len: BODY.len()
                })
            );
        }

        #[test]
        fn a_quote_that_is_not_the_bytes_at_the_span_is_refused() {
            let store = store_with("r", BODY);
            let owners = owners_of("r");
            let resolver = ExactSourceResolver::new(&store, &owners);
            let line = "ent-0001: status is active";
            // Wrong tail with the same length.
            assert!(matches!(
                resolver.resolve(&citation("r", 0, line.len(), "ent-0001: status is actiXX")),
                Err(SourceError::QuoteMismatch { .. })
            ));
            // A doctored quote that is longer than the span and shares its prefix.
            assert!(matches!(
                resolver.resolve(&citation("r", 0, line.len(), "ent-0001: status is blue")),
                Err(SourceError::QuoteMismatch { .. })
            ));
            // A prefix of the real bytes is not the bytes.
            assert!(matches!(
                resolver.resolve(&citation("r", 0, line.len(), "ent-0001: status")),
                Err(SourceError::QuoteMismatch { .. })
            ));
        }

        #[test]
        fn the_offset_is_part_of_the_citation() {
            let store = store_with("r", BODY);
            let owners = owners_of("r");
            let resolver = ExactSourceResolver::new(&store, &owners);
            let line = "ent-0003: color is red";
            let start = BODY.find(line).expect("line present");
            // The quote occurs in the event, but not at the cited span.
            assert!(matches!(
                resolver.resolve(&citation("r", start - 1, start - 1 + line.len(), line)),
                Err(SourceError::QuoteMismatch { .. })
            ));
            assert!(resolver
                .resolve(&citation("r", start, start + line.len(), line))
                .is_ok());
        }

        #[test]
        fn every_refusal_has_a_distinct_stable_code() {
            let errors = [
                SourceError::UnknownRawEvent {
                    raw_id: RawEventId::from("a"),
                },
                SourceError::DigestMismatch {
                    raw_id: RawEventId::from("a"),
                },
                SourceError::EmptySpan { start: 0, end: 0 },
                SourceError::SpanOutOfRange {
                    start: 0,
                    end: 1,
                    len: 0,
                },
                SourceError::QuoteMismatch {
                    raw_id: RawEventId::from("a"),
                    start: 0,
                    end: 1,
                },
            ];
            let codes: std::collections::BTreeSet<&str> =
                errors.iter().map(SourceError::code).collect();
            assert_eq!(codes.len(), errors.len());
            assert!(errors.iter().all(|error| !error.to_string().is_empty()));
        }
    }

    mod verifier {
        use super::*;

        fn verify(claim: (&str, &str, &str), text: &str) -> Result<ClaimProof, ClaimError> {
            verify_claim(&Claim::new(claim.0, claim.1, claim.2), &resolved(text))
        }

        #[test]
        fn a_claim_equal_to_the_statement_is_verified() {
            let proof =
                verify(("ent-1", "status", "active"), "ent-1: status is active").expect("verified");
            assert_eq!(proof.claim(), &Claim::new("ent-1", "status", "active"));
            assert_eq!(proof.source().text(), "ent-1: status is active");
        }

        #[test]
        fn another_subject_predicate_or_value_is_refused_in_that_order() {
            let text = "ent-1: status is active";
            assert_eq!(
                verify(("ent-2", "status", "active"), text).unwrap_err(),
                ClaimError::SubjectMismatch
            );
            assert_eq!(
                verify(("ent-1", "tier", "active"), text).unwrap_err(),
                ClaimError::PredicateMismatch
            );
            assert_eq!(
                verify(("ent-1", "status", "idle"), text).unwrap_err(),
                ClaimError::ValueMismatch
            );
            // A full stop is part of the value token.
            assert_eq!(
                verify(("ent-1", "status", "active"), "ent-1: status is active.").unwrap_err(),
                ClaimError::ValueMismatch
            );
            // Everything differs: the subject is reported.
            assert_eq!(
                verify(("ent-2", "tier", "idle"), text).unwrap_err(),
                ClaimError::SubjectMismatch
            );
        }

        #[test]
        fn lookalike_subjects_are_different_subjects() {
            let text = "ent-0012: status is active";
            for lookalike in [
                "ent-O012",
                "ent-0012 ",
                "ENT-0012",
                "\u{435}nt-0012",
                "ent\u{2010}0012",
                "ent-0012\u{200b}",
            ] {
                let outcome = verify((lookalike, "status", "active"), text);
                assert!(
                    matches!(
                        outcome,
                        Err(ClaimError::SubjectMismatch | ClaimError::MalformedClaim)
                    ),
                    "{lookalike:?} -> {outcome:?}"
                );
            }
            // The statement may be the one with the lookalike subject.
            assert_eq!(
                verify(
                    ("ent-0012", "status", "active"),
                    "ent-O012: status is active"
                )
                .unwrap_err(),
                ClaimError::SubjectMismatch
            );
        }

        #[test]
        fn a_negated_statement_does_not_verify_the_claim_it_negates() {
            assert_eq!(
                verify(("ent-1", "status", "active"), "ent-1: status is not active").unwrap_err(),
                ClaimError::NegatedStatement
            );
            // A negated statement about another value says nothing for the claim.
            assert_eq!(
                verify(("ent-1", "status", "idle"), "ent-1: status is not active").unwrap_err(),
                ClaimError::ValueMismatch
            );
            // A token that merely contains `not` is an ordinary value.
            for value in ["nothing", "knot", "cannot", "not-yet", "note"] {
                let text = format!("ent-1: flag is {value}");
                assert!(verify(("ent-1", "flag", value), &text).is_ok(), "{value}");
            }
            // The bare value `not` is a negation cut off before its value: it verifies
            // nothing, however the claim is worded.
            for claim in [("ent-1", "flag", "not"), ("ent-1", "flag", "active")] {
                assert_eq!(
                    verify(claim, "ent-1: flag is not").unwrap_err(),
                    ClaimError::MalformedStatement,
                    "{claim:?}"
                );
            }
        }

        #[test]
        fn text_that_is_not_exactly_one_statement_is_malformed() {
            let claim = ("ent-1", "status", "active");
            for text in [
                "ent-1: status is active\n",
                " ent-1: status is active",
                "ent-1: status is active ",
                "ent-1:  status is active",
                "ent-1 status is active",
                "ent-1: status = active",
                "ent-1: status is",
                "ent-1: status is active\nent-2: status is idle",
                "ent-1: status is very active",
                "ent-1: status IS active",
                ": status is active",
            ] {
                assert_eq!(
                    verify(claim, text).unwrap_err(),
                    ClaimError::MalformedStatement,
                    "{text:?}"
                );
            }
        }

        #[test]
        fn a_claim_that_is_not_made_of_tokens_is_malformed() {
            let text = "ent-1: status is active";
            for claim in [
                ("", "status", "active"),
                ("ent-1", "", "active"),
                ("ent-1", "status", ""),
                ("ent-1", "sta tus", "active"),
                ("ent-1", "status", "act\nive"),
                ("ent-1 ", "status", "active"),
            ] {
                assert_eq!(verify(claim, text).unwrap_err(), ClaimError::MalformedClaim);
            }
        }

        #[test]
        fn every_refusal_has_a_distinct_stable_code() {
            let errors = [
                ClaimError::MalformedClaim,
                ClaimError::MalformedStatement,
                ClaimError::SubjectMismatch,
                ClaimError::PredicateMismatch,
                ClaimError::ValueMismatch,
                ClaimError::NegatedStatement,
            ];
            let codes: std::collections::BTreeSet<&str> =
                errors.iter().map(ClaimError::code).collect();
            assert_eq!(codes.len(), errors.len());
            assert!(errors.iter().all(|error| !error.to_string().is_empty()));
        }

        #[test]
        fn verification_is_deterministic() {
            let claim = Claim::new("ent-1", "status", "active");
            let source = resolved("ent-1: status is active");
            assert_eq!(verify_claim(&claim, &source), verify_claim(&claim, &source));
        }
    }
}
