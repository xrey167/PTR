//! Typed memory records keyed by [`MemoryClass`] and their lifecycle state machine.

use crate::codec::{bounded_capacity, CodecError, Reader, Writer, MIN_STRING_BYTES};
use crate::MemoryClass;
use ptr_types::Revision;
use std::fmt;

/// Identifier of a [`MemoryRecord`], unique across all four memory classes.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MemoryRecordId(pub String);

impl From<&str> for MemoryRecordId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for MemoryRecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Lifecycle state of a memory record.
///
/// ```text
/// Provisional -> Authoritative -> NeedsRevalidation -> Authoritative
///      |               |                 |  \-> Provisional
///      +---------------+-----------------+--> Retired --> Revoked
///      +---------------+-----------------+--> Revoked (terminal)
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RecordState {
    /// Proposed or observed but not (yet) authoritative. Every record starts here.
    Provisional,
    /// Backed by at least one observation tied to a committed revision (Invariant 4).
    Authoritative,
    /// An ancestor was revoked, or a verifier contradicted the record; it must not be used as
    /// truth until revalidated.
    NeedsRevalidation,
    /// Forgotten by retention. Raw history elsewhere is untouched; the record is not revoked.
    Retired,
    /// Revoked. Terminal: a revoked record never becomes usable again (Invariant 3).
    Revoked,
}

impl RecordState {
    /// Whether the state machine allows `self -> to`. Self-transitions are not allowed.
    pub fn can_transition_to(self, to: Self) -> bool {
        matches!(
            (self, to),
            (Self::Provisional, Self::Authoritative)
                | (Self::Provisional, Self::Retired)
                | (Self::Provisional, Self::Revoked)
                | (Self::Authoritative, Self::NeedsRevalidation)
                | (Self::Authoritative, Self::Retired)
                | (Self::Authoritative, Self::Revoked)
                | (Self::NeedsRevalidation, Self::Authoritative)
                | (Self::NeedsRevalidation, Self::Provisional)
                | (Self::NeedsRevalidation, Self::Retired)
                | (Self::NeedsRevalidation, Self::Revoked)
                | (Self::Retired, Self::Revoked)
        )
    }

    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::Provisional => 0,
            Self::Authoritative => 1,
            Self::NeedsRevalidation => 2,
            Self::Retired => 3,
            Self::Revoked => 4,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Result<Self, CodecError> {
        match tag {
            0 => Ok(Self::Provisional),
            1 => Ok(Self::Authoritative),
            2 => Ok(Self::NeedsRevalidation),
            3 => Ok(Self::Retired),
            4 => Ok(Self::Revoked),
            tag => Err(CodecError::InvalidTag {
                field: "record state",
                tag,
            }),
        }
    }
}

/// How a piece of support for a record came about.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SupportKind {
    /// An observation of `source` that belongs to the given revision. It only counts as
    /// committed support once that revision is at or below the store's committed horizon.
    Observation { revision: Revision },
    /// Derived by inference from other memory. Never sufficient for authority (Invariant 5).
    Derived,
    /// Surfaced by search/index. Never sufficient for authority (Invariant 5).
    SearchHit,
}

/// One independent piece of support, attributed to a source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Support {
    pub source: String,
    pub kind: SupportKind,
}

impl Support {
    /// Whether this support is an observation at or below `horizon`.
    pub fn is_committed(&self, horizon: Revision) -> bool {
        matches!(self.kind, SupportKind::Observation { revision } if revision <= horizon)
    }
}

/// A typed record of one memory class.
#[derive(Clone, Eq, PartialEq)]
pub struct MemoryRecord {
    pub id: MemoryRecordId,
    pub class: MemoryClass,
    /// Free-text body. Redacted by `Debug` and by the human-readable mirror when `sensitive`.
    pub body: String,
    /// Flags the body as secret/private (Invariant 13).
    pub sensitive: bool,
    /// Pinned records are never forgotten by retention.
    pub pinned: bool,
    pub state: RecordState,
    pub supports: Vec<Support>,
    /// Records this one was derived from; revoking any of them propagates here.
    pub derived_from: Vec<MemoryRecordId>,
    /// Revoked records whose revocation reached this one by ancestry (sorted, unique).
    pub revoked_ancestors: Vec<MemoryRecordId>,
    /// Store logical clock tick at creation.
    pub created_at: u64,
    /// Store logical clock tick of the last change.
    pub updated_at: u64,
}

impl fmt::Debug for MemoryRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let body: &dyn fmt::Debug = if self.sensitive {
            &"[REDACTED]"
        } else {
            &self.body
        };
        f.debug_struct("MemoryRecord")
            .field("id", &self.id)
            .field("class", &self.class)
            .field("body", body)
            .field("sensitive", &self.sensitive)
            .field("pinned", &self.pinned)
            .field("state", &self.state)
            .field("supports", &self.supports)
            .field("derived_from", &self.derived_from)
            .field("revoked_ancestors", &self.revoked_ancestors)
            .field("created_at", &self.created_at)
            .field("updated_at", &self.updated_at)
            .finish()
    }
}

impl MemoryRecord {
    /// Number of distinct sources among committed observations at `horizon`.
    pub fn independent_committed_sources(&self, horizon: Revision) -> usize {
        let mut sources: Vec<&str> = self
            .supports
            .iter()
            .filter(|support| support.is_committed(horizon))
            .map(|support| support.source.as_str())
            .collect();
        sources.sort_unstable();
        sources.dedup();
        sources.len()
    }

    /// Whether any support is an observation tied to a revision at or below `horizon`.
    pub fn has_committed_support(&self, horizon: Revision) -> bool {
        self.supports
            .iter()
            .any(|support| support.is_committed(horizon))
    }

    /// Whether the record has support, none of which is an observation (only derived/search).
    pub fn only_derived_or_search_support(&self) -> bool {
        !self.supports.is_empty()
            && self
                .supports
                .iter()
                .all(|support| !matches!(support.kind, SupportKind::Observation { .. }))
    }
}

pub(crate) fn class_tag(class: MemoryClass) -> u8 {
    match class {
        MemoryClass::Semantic => 0,
        MemoryClass::Episodic => 1,
        MemoryClass::Procedural => 2,
        MemoryClass::Epistemic => 3,
    }
}

pub(crate) fn class_from_tag(tag: u8) -> Result<MemoryClass, CodecError> {
    match tag {
        0 => Ok(MemoryClass::Semantic),
        1 => Ok(MemoryClass::Episodic),
        2 => Ok(MemoryClass::Procedural),
        3 => Ok(MemoryClass::Epistemic),
        tag => Err(CodecError::InvalidTag {
            field: "memory class",
            tag,
        }),
    }
}

pub(crate) fn class_name(class: MemoryClass) -> &'static str {
    match class {
        MemoryClass::Semantic => "semantic",
        MemoryClass::Episodic => "episodic",
        MemoryClass::Procedural => "procedural",
        MemoryClass::Epistemic => "epistemic",
    }
}

pub(crate) fn write_record(w: &mut Writer, record: &MemoryRecord) {
    w.str(&record.id.0);
    w.u8(class_tag(record.class));
    w.str(&record.body);
    w.bool(record.sensitive);
    w.bool(record.pinned);
    w.u8(record.state.tag());
    w.count(record.supports.len());
    for support in &record.supports {
        w.str(&support.source);
        match support.kind {
            SupportKind::Observation { revision } => {
                w.u8(0);
                w.u64(revision.0);
            }
            SupportKind::Derived => w.u8(1),
            SupportKind::SearchHit => w.u8(2),
        }
    }
    w.count(record.derived_from.len());
    for id in &record.derived_from {
        w.str(&id.0);
    }
    w.count(record.revoked_ancestors.len());
    for id in &record.revoked_ancestors {
        w.str(&id.0);
    }
    w.u64(record.created_at);
    w.u64(record.updated_at);
}

/// Minimum encoded size of a [`Support`] (empty source string plus the kind tag).
pub(crate) const MIN_SUPPORT_BYTES: usize = MIN_STRING_BYTES + 1;

/// Minimum encoded size of a [`MemoryRecord`] (all strings empty, all lists empty).
pub(crate) const MIN_RECORD_BYTES: usize = 40;

pub(crate) fn read_record(r: &mut Reader<'_>) -> Result<MemoryRecord, CodecError> {
    let id = MemoryRecordId(r.string()?);
    let class = class_from_tag(r.u8()?)?;
    let body = r.string()?;
    let sensitive = r.bool()?;
    let pinned = r.bool()?;
    let state = RecordState::from_tag(r.u8()?)?;
    let support_count = r.count_of(MIN_SUPPORT_BYTES)?;
    let mut supports = Vec::with_capacity(bounded_capacity(support_count));
    for _ in 0..support_count {
        let source = r.string()?;
        let kind = match r.u8()? {
            0 => SupportKind::Observation {
                revision: Revision(r.u64()?),
            },
            1 => SupportKind::Derived,
            2 => SupportKind::SearchHit,
            tag => {
                return Err(CodecError::InvalidTag {
                    field: "support kind",
                    tag,
                })
            }
        };
        supports.push(Support { source, kind });
    }
    let derived_count = r.count_of(MIN_STRING_BYTES)?;
    let mut derived_from = Vec::with_capacity(bounded_capacity(derived_count));
    for _ in 0..derived_count {
        derived_from.push(MemoryRecordId(r.string()?));
    }
    let ancestor_count = r.count_of(MIN_STRING_BYTES)?;
    let mut revoked_ancestors = Vec::with_capacity(bounded_capacity(ancestor_count));
    for _ in 0..ancestor_count {
        revoked_ancestors.push(MemoryRecordId(r.string()?));
    }
    Ok(MemoryRecord {
        id,
        class,
        body,
        sensitive,
        pinned,
        state,
        supports,
        derived_from,
        revoked_ancestors,
        created_at: r.u64()?,
        updated_at: r.u64()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [RecordState; 5] = [
        RecordState::Provisional,
        RecordState::Authoritative,
        RecordState::NeedsRevalidation,
        RecordState::Retired,
        RecordState::Revoked,
    ];

    fn encoded(record: &MemoryRecord) -> usize {
        let mut w = Writer::new();
        write_record(&mut w, record);
        w.finish().len()
    }

    fn smallest_record() -> MemoryRecord {
        MemoryRecord {
            id: "".into(),
            class: MemoryClass::Semantic,
            body: String::new(),
            sensitive: false,
            pinned: false,
            state: RecordState::Provisional,
            supports: Vec::new(),
            derived_from: Vec::new(),
            revoked_ancestors: Vec::new(),
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn min_sizes_equal_the_encoding_of_the_smallest_elements() {
        assert_eq!(encoded(&smallest_record()), MIN_RECORD_BYTES);
        let mut with_supports = smallest_record();
        for kind in [SupportKind::Derived, SupportKind::SearchHit] {
            with_supports.supports = vec![Support {
                source: String::new(),
                kind,
            }];
            assert_eq!(
                encoded(&with_supports) - MIN_RECORD_BYTES,
                MIN_SUPPORT_BYTES
            );
        }
        let mut with_ids = smallest_record();
        with_ids.derived_from = vec!["".into()];
        assert_eq!(encoded(&with_ids) - MIN_RECORD_BYTES, MIN_STRING_BYTES);
    }

    #[test]
    fn an_inflated_element_count_is_rejected_before_reserving() {
        // supports count = every remaining byte, but each support needs at least 5.
        let mut w = Writer::new();
        write_record(&mut w, &smallest_record());
        let mut bytes = w.finish();
        let supports_at = 4 + 1 + 4 + 3; // after id, class, body, flags/state
        let remaining = u32::try_from(bytes.len() - supports_at - 4).unwrap();
        bytes[supports_at..supports_at + 4].copy_from_slice(&remaining.to_le_bytes());
        assert_eq!(
            read_record(&mut Reader::new(&bytes)).unwrap_err(),
            CodecError::LengthTooLarge
        );
    }

    #[test]
    fn revoked_is_terminal_and_self_transitions_are_refused() {
        for state in ALL {
            assert!(!state.can_transition_to(state));
            assert!(!RecordState::Revoked.can_transition_to(state));
        }
        assert!(!RecordState::Retired.can_transition_to(RecordState::Authoritative));
        assert!(!RecordState::Provisional.can_transition_to(RecordState::NeedsRevalidation));
    }

    #[test]
    fn every_state_tag_round_trips() {
        for state in ALL {
            assert_eq!(RecordState::from_tag(state.tag()), Ok(state));
        }
        assert!(RecordState::from_tag(9).is_err());
    }

    #[test]
    fn debug_redacts_sensitive_body() {
        let record = MemoryRecord {
            id: "r".into(),
            class: MemoryClass::Semantic,
            body: "hunter2".to_owned(),
            sensitive: true,
            pinned: false,
            state: RecordState::Provisional,
            supports: Vec::new(),
            derived_from: Vec::new(),
            revoked_ancestors: Vec::new(),
            created_at: 0,
            updated_at: 0,
        };
        let text = format!("{record:?}");
        assert!(!text.contains("hunter2"));
        assert!(text.contains("[REDACTED]"));
    }
}
