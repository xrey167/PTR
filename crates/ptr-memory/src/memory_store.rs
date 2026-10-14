//! `DurableMemoryStore`: durable semantic, episodic, procedural and epistemic memory records
//! with an enforced lifecycle state machine and revocation propagation.
//!
//! Every mutation is computed completely first, appended to the journal as one atomic frame
//! (`MemoryDelta`: committed horizon, logical clock and the full new value of every changed
//! record) and only then applied in memory. A failed journal write therefore leaves memory
//! unchanged, and replaying the journal reproduces the exact state.
//!
//! Authority rules enforced here, independent of any policy object:
//!
//! * Invariant 4: a record becomes `Authoritative` only with at least one observation whose
//!   revision is at or below the store's committed horizon ([`DurableMemoryStore::advance_committed_revision`]).
//! * Invariant 5: derived and search support never count, so a record supported only by them
//!   can never become authoritative.
//! * Invariant 3: `Revoked` is terminal, and revoking a record flags every transitive dependent
//!   (`derived_from`) as `NeedsRevalidation`.
//! * Invariant 8: record counts per class, field sizes and the revocation traversal are bounded.

use crate::codec::{CodecError, Reader, Writer};
use crate::consolidation::{
    ConsolidationPolicy, Contradiction, PromotionContext, PromotionDecision, RetentionVerdict,
};
use crate::journal::{JournalError, JournalKind, JournalOptions, MemoryJournal};
use crate::record::{read_record, write_record};
use crate::record::{MemoryRecord, MemoryRecordId, RecordState, Support};
use crate::MemoryClass;
use ptr_types::Revision;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const DELTA_TAG: u8 = 0x10;

/// Size and traversal bounds of a [`DurableMemoryStore`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryLimits {
    pub max_records_per_class: usize,
    pub max_id_bytes: usize,
    pub max_body_bytes: usize,
    pub max_supports: usize,
    pub max_derived_from: usize,
    /// Deepest dependent chain a revocation will follow.
    pub max_revocation_depth: usize,
    /// Most dependents a single revocation will visit.
    pub max_revocation_visited: usize,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_records_per_class: 10_000,
            max_id_bytes: 256,
            max_body_bytes: 64 * 1024,
            max_supports: 64,
            max_derived_from: 64,
            max_revocation_depth: 64,
            max_revocation_visited: 10_000,
        }
    }
}

/// Options of a [`DurableMemoryStore`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryStoreOptions {
    pub journal: JournalOptions,
    pub limits: MemoryLimits,
    /// Compact automatically once this many frames were appended since the last compaction.
    pub compact_after_records: Option<u64>,
}

impl Default for MemoryStoreOptions {
    fn default() -> Self {
        Self {
            journal: JournalOptions::default(),
            limits: MemoryLimits::default(),
            compact_after_records: Some(10_000),
        }
    }
}

/// Input for creating a record. New records always start `Provisional`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NewMemoryRecord {
    pub id: MemoryRecordId,
    pub class: MemoryClass,
    pub body: String,
    pub sensitive: bool,
    pub pinned: bool,
    pub supports: Vec<Support>,
    pub derived_from: Vec<MemoryRecordId>,
}

/// A violated input bound or shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordViolation {
    EmptyId,
    IdTooLong,
    BodyTooLarge,
    TooManySupports,
    TooManyDerivations,
    EmptySupportSource,
}

/// Which revocation traversal bound was hit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraversalLimit {
    Depth(usize),
    Visited(usize),
}

/// Failures of [`DurableMemoryStore`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryStoreError {
    Journal(JournalError),
    /// A journal record passed its CRC but could not be decoded.
    Decode {
        record_index: usize,
        error: CodecError,
    },
    DuplicateRecord(MemoryRecordId),
    UnknownRecord(MemoryRecordId),
    InvalidRecord(RecordViolation),
    CapacityExceeded {
        class: MemoryClassName,
        limit: usize,
    },
    /// The lifecycle state machine does not allow `from -> to`.
    InvalidTransition {
        id: MemoryRecordId,
        from: RecordState,
        to: RecordState,
    },
    /// Authority gate (Invariants 4 and 5): no observation at or below the committed horizon.
    NoCommittedSupport(MemoryRecordId),
    /// The record (or a revoked ancestor it names) is revoked, so it cannot gain authority.
    RevokedAncestor {
        id: MemoryRecordId,
        ancestor: MemoryRecordId,
    },
    /// Retired and revoked records are immutable.
    TerminalRecord(MemoryRecordId),
    SelfDerivation(MemoryRecordId),
    /// Revocation would exceed a traversal bound; nothing was changed.
    TraversalBoundExceeded(TraversalLimit),
    /// The committed horizon never moves backwards.
    HorizonRegression {
        current: Revision,
        requested: Revision,
    },
}

impl From<JournalError> for MemoryStoreError {
    fn from(error: JournalError) -> Self {
        Self::Journal(error)
    }
}

/// Printable class name used in errors (keeps [`MemoryClass`] untouched).
pub type MemoryClassName = &'static str;

/// What a revocation did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RevocationReport {
    /// The record that was revoked.
    pub revoked: MemoryRecordId,
    /// Dependents newly moved to `NeedsRevalidation`, sorted by id.
    pub marked_for_revalidation: Vec<MemoryRecordId>,
    /// Dependents visited (including terminal ones that were left as they are).
    pub visited: usize,
}

/// A retention action that was applied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppliedRetention {
    pub id: MemoryRecordId,
    pub verdict: RetentionVerdict,
}

/// Journaled memory store for all four memory classes.
pub struct DurableMemoryStore {
    records: BTreeMap<MemoryRecordId, MemoryRecord>,
    horizon: Revision,
    clock: u64,
    journal: MemoryJournal,
    options: MemoryStoreOptions,
    recovered_truncated_bytes: u64,
    compaction_failures: u64,
    last_compaction_error: Option<MemoryStoreError>,
}

impl DurableMemoryStore {
    /// Opens (creating) the journal at `path` and replays it.
    pub fn open(
        path: impl AsRef<Path>,
        options: MemoryStoreOptions,
    ) -> Result<Self, MemoryStoreError> {
        let (journal, recovery) = MemoryJournal::open(path, JournalKind::Memory, options.journal)?;
        let mut records = BTreeMap::new();
        let mut horizon = Revision(0);
        let mut clock = 0;
        for (record_index, payload) in recovery.records.iter().enumerate() {
            let delta = decode_delta(payload).map_err(|error| MemoryStoreError::Decode {
                record_index,
                error,
            })?;
            horizon = delta.horizon;
            clock = delta.clock;
            for record in delta.records {
                records.insert(record.id.clone(), record);
            }
        }
        Ok(Self {
            records,
            horizon,
            clock,
            journal,
            options,
            recovered_truncated_bytes: recovery.truncated_bytes,
            compaction_failures: 0,
            last_compaction_error: None,
        })
    }

    /// Number of automatic compactions that failed after their triggering mutation was already
    /// durable. Such a failure never fails the mutation; compaction is retried on later commits.
    pub fn compaction_failures(&self) -> u64 {
        self.compaction_failures
    }

    /// The most recent automatic-compaction failure, cleared by the next successful compaction.
    pub fn last_compaction_error(&self) -> Option<&MemoryStoreError> {
        self.last_compaction_error.as_ref()
    }

    /// Bytes of a torn tail removed when this store was opened.
    pub fn recovered_truncated_bytes(&self) -> u64 {
        self.recovered_truncated_bytes
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Highest revision the caller declared committed.
    pub fn committed_revision(&self) -> Revision {
        self.horizon
    }

    /// Logical clock: the number of committed mutations.
    pub fn clock(&self) -> u64 {
        self.clock
    }

    pub fn get(&self, id: &MemoryRecordId) -> Option<&MemoryRecord> {
        self.records.get(id)
    }

    /// All records sorted by id.
    pub fn records(&self) -> impl Iterator<Item = &MemoryRecord> {
        self.records.values()
    }

    /// Records of one class sorted by id.
    pub fn records_of(&self, class: MemoryClass) -> impl Iterator<Item = &MemoryRecord> {
        self.records
            .values()
            .filter(move |record| record.class == class)
    }

    fn class_count(&self, class: MemoryClass) -> usize {
        self.records_of(class).count()
    }

    fn require(&self, id: &MemoryRecordId) -> Result<&MemoryRecord, MemoryStoreError> {
        self.records
            .get(id)
            .ok_or_else(|| MemoryStoreError::UnknownRecord(id.clone()))
    }

    fn require_mutable(&self, id: &MemoryRecordId) -> Result<MemoryRecord, MemoryStoreError> {
        let record = self.require(id)?;
        match record.state {
            RecordState::Retired | RecordState::Revoked => {
                Err(MemoryStoreError::TerminalRecord(id.clone()))
            }
            RecordState::Provisional
            | RecordState::Authoritative
            | RecordState::NeedsRevalidation => Ok(record.clone()),
        }
    }

    fn check_support(&self, support: &Support) -> Result<(), MemoryStoreError> {
        if support.source.is_empty() {
            return Err(MemoryStoreError::InvalidRecord(
                RecordViolation::EmptySupportSource,
            ));
        }
        Ok(())
    }

    /// Creates a `Provisional` record.
    pub fn insert(&mut self, new: NewMemoryRecord) -> Result<(), MemoryStoreError> {
        let limits = self.options.limits;
        if new.id.0.is_empty() {
            return Err(MemoryStoreError::InvalidRecord(RecordViolation::EmptyId));
        }
        if new.id.0.len() > limits.max_id_bytes {
            return Err(MemoryStoreError::InvalidRecord(RecordViolation::IdTooLong));
        }
        if new.body.len() > limits.max_body_bytes {
            return Err(MemoryStoreError::InvalidRecord(
                RecordViolation::BodyTooLarge,
            ));
        }
        if new.supports.len() > limits.max_supports {
            return Err(MemoryStoreError::InvalidRecord(
                RecordViolation::TooManySupports,
            ));
        }
        if new.derived_from.len() > limits.max_derived_from {
            return Err(MemoryStoreError::InvalidRecord(
                RecordViolation::TooManyDerivations,
            ));
        }
        for support in &new.supports {
            self.check_support(support)?;
        }
        if self.records.contains_key(&new.id) {
            return Err(MemoryStoreError::DuplicateRecord(new.id));
        }
        if self.class_count(new.class) >= limits.max_records_per_class {
            return Err(MemoryStoreError::CapacityExceeded {
                class: crate::record::class_name(new.class),
                limit: limits.max_records_per_class,
            });
        }
        let mut derived_from: Vec<MemoryRecordId> = Vec::with_capacity(new.derived_from.len());
        for parent in new.derived_from {
            if parent == new.id {
                return Err(MemoryStoreError::SelfDerivation(new.id));
            }
            if self.require(&parent)?.state == RecordState::Revoked {
                return Err(MemoryStoreError::RevokedAncestor {
                    id: new.id,
                    ancestor: parent,
                });
            }
            if !derived_from.contains(&parent) {
                derived_from.push(parent);
            }
        }
        let record = MemoryRecord {
            id: new.id,
            class: new.class,
            body: new.body,
            sensitive: new.sensitive,
            pinned: new.pinned,
            state: RecordState::Provisional,
            supports: new.supports,
            derived_from,
            revoked_ancestors: Vec::new(),
            created_at: self.clock + 1,
            updated_at: self.clock + 1,
        };
        self.commit(vec![record], self.horizon)
    }

    /// Adds a piece of support to a live record.
    pub fn add_support(
        &mut self,
        id: &MemoryRecordId,
        support: Support,
    ) -> Result<(), MemoryStoreError> {
        self.check_support(&support)?;
        let mut record = self.require_mutable(id)?;
        if record.supports.len() >= self.options.limits.max_supports {
            return Err(MemoryStoreError::InvalidRecord(
                RecordViolation::TooManySupports,
            ));
        }
        record.supports.push(support);
        self.commit(vec![record], self.horizon)
    }

    /// Records that `id` is derived from `from`. Cycles between records are allowed; revocation
    /// traversal is cycle-safe.
    pub fn add_derivation(
        &mut self,
        id: &MemoryRecordId,
        from: &MemoryRecordId,
    ) -> Result<(), MemoryStoreError> {
        if id == from {
            return Err(MemoryStoreError::SelfDerivation(id.clone()));
        }
        let mut record = self.require_mutable(id)?;
        if self.require(from)?.state == RecordState::Revoked {
            return Err(MemoryStoreError::RevokedAncestor {
                id: id.clone(),
                ancestor: from.clone(),
            });
        }
        if record.derived_from.contains(from) {
            return Ok(());
        }
        if record.derived_from.len() >= self.options.limits.max_derived_from {
            return Err(MemoryStoreError::InvalidRecord(
                RecordViolation::TooManyDerivations,
            ));
        }
        record.derived_from.push(from.clone());
        self.commit(vec![record], self.horizon)
    }

    /// Declares that all revisions up to and including `revision` are committed. Only
    /// observations at or below this horizon count as authority.
    pub fn advance_committed_revision(
        &mut self,
        revision: Revision,
    ) -> Result<(), MemoryStoreError> {
        if revision < self.horizon {
            return Err(MemoryStoreError::HorizonRegression {
                current: self.horizon,
                requested: revision,
            });
        }
        if revision == self.horizon {
            return Ok(());
        }
        self.commit(Vec::new(), revision)
    }

    fn authority_gate(&self, record: &MemoryRecord) -> Result<(), MemoryStoreError> {
        if !record.has_committed_support(self.horizon) {
            return Err(MemoryStoreError::NoCommittedSupport(record.id.clone()));
        }
        for ancestor in record.derived_from.iter().chain(&record.revoked_ancestors) {
            if self.require(ancestor)?.state == RecordState::Revoked {
                return Err(MemoryStoreError::RevokedAncestor {
                    id: record.id.clone(),
                    ancestor: ancestor.clone(),
                });
            }
        }
        Ok(())
    }

    /// Performs a lifecycle transition. Only transitions allowed by
    /// [`RecordState::can_transition_to`] succeed; `Authoritative` additionally passes the
    /// authority gate, and `Revoked` propagates exactly like [`DurableMemoryStore::revoke`].
    pub fn transition(
        &mut self,
        id: &MemoryRecordId,
        to: RecordState,
    ) -> Result<(), MemoryStoreError> {
        let from = self.require(id)?.state;
        if !from.can_transition_to(to) {
            return Err(MemoryStoreError::InvalidTransition {
                id: id.clone(),
                from,
                to,
            });
        }
        match to {
            RecordState::Revoked => self.revoke(id).map(|_| ()),
            RecordState::Authoritative => {
                let mut record = self.require(id)?.clone();
                self.authority_gate(&record)?;
                record.state = RecordState::Authoritative;
                record.revoked_ancestors.clear();
                self.commit(vec![record], self.horizon)
            }
            RecordState::Provisional | RecordState::NeedsRevalidation | RecordState::Retired => {
                let mut record = self.require(id)?.clone();
                record.state = to;
                self.commit(vec![record], self.horizon)
            }
        }
    }

    /// Revokes `id` and flags every transitive dependent as `NeedsRevalidation`.
    ///
    /// The traversal is breadth-first over the reverse `derived_from` edges with a visited set
    /// (cycle-safe) and the bounds in [`MemoryLimits`]; when a bound would be exceeded nothing
    /// is changed. Revoking an already revoked record is an idempotent no-op.
    pub fn revoke(&mut self, id: &MemoryRecordId) -> Result<RevocationReport, MemoryStoreError> {
        let root = self.require(id)?;
        if root.state == RecordState::Revoked {
            return Ok(RevocationReport {
                revoked: id.clone(),
                marked_for_revalidation: Vec::new(),
                visited: 0,
            });
        }
        let dependents = self.dependents_closure(id)?;
        let mut changed = Vec::new();
        let mut revoked_root = self.require(id)?.clone();
        revoked_root.state = RecordState::Revoked;
        changed.push(revoked_root);
        let mut marked = Vec::new();
        for dependent in &dependents {
            let mut record = self.require(dependent)?.clone();
            let before = record.clone();
            match record.state {
                RecordState::Provisional | RecordState::Authoritative => {
                    record.state = RecordState::NeedsRevalidation;
                    marked.push(record.id.clone());
                }
                RecordState::NeedsRevalidation => {}
                RecordState::Retired | RecordState::Revoked => continue,
            }
            if !record.revoked_ancestors.contains(id) {
                record.revoked_ancestors.push(id.clone());
                record.revoked_ancestors.sort();
            }
            if record != before {
                changed.push(record);
            }
        }
        marked.sort();
        self.commit(changed, self.horizon)?;
        Ok(RevocationReport {
            revoked: id.clone(),
            marked_for_revalidation: marked,
            visited: dependents.len(),
        })
    }

    /// Transitive dependents of `root` (excluding `root`), breadth-first, bounded.
    fn dependents_closure(
        &self,
        root: &MemoryRecordId,
    ) -> Result<Vec<MemoryRecordId>, MemoryStoreError> {
        let limits = self.options.limits;
        let mut reverse: BTreeMap<&MemoryRecordId, Vec<&MemoryRecordId>> = BTreeMap::new();
        for record in self.records.values() {
            for parent in &record.derived_from {
                reverse.entry(parent).or_default().push(&record.id);
            }
        }
        let mut seen: BTreeSet<&MemoryRecordId> = BTreeSet::new();
        seen.insert(root);
        let mut order: Vec<MemoryRecordId> = Vec::new();
        let mut frontier: Vec<&MemoryRecordId> = vec![root];
        let mut depth = 0_usize;
        while !frontier.is_empty() {
            depth += 1;
            let mut next: Vec<&MemoryRecordId> = Vec::new();
            for node in &frontier {
                let Some(children) = reverse.get(node) else {
                    continue;
                };
                for child in children {
                    if seen.insert(child) {
                        if order.len() >= limits.max_revocation_visited {
                            return Err(MemoryStoreError::TraversalBoundExceeded(
                                TraversalLimit::Visited(limits.max_revocation_visited),
                            ));
                        }
                        order.push((*child).clone());
                        next.push(child);
                    }
                }
            }
            if !next.is_empty() && depth > limits.max_revocation_depth {
                return Err(MemoryStoreError::TraversalBoundExceeded(
                    TraversalLimit::Depth(limits.max_revocation_depth),
                ));
            }
            frontier = next;
        }
        Ok(order)
    }

    /// Applies `policy` to an episodic record: on `Promote` the record becomes `Semantic` and
    /// `Authoritative`. Any other decision leaves the store untouched and is returned.
    pub fn promote(
        &mut self,
        id: &MemoryRecordId,
        policy: &ConsolidationPolicy,
        context: &PromotionContext,
    ) -> Result<PromotionDecision, MemoryStoreError> {
        let record = self.require(id)?;
        let decision = policy.decide_promotion(record, self.horizon, context);
        if decision != PromotionDecision::Promote {
            return Ok(decision);
        }
        let mut record = record.clone();
        if record.state != RecordState::Authoritative {
            self.authority_gate(&record)?;
            record.state = RecordState::Authoritative;
            record.revoked_ancestors.clear();
        }
        if self.class_count(MemoryClass::Semantic) >= self.options.limits.max_records_per_class {
            return Err(MemoryStoreError::CapacityExceeded {
                class: crate::record::class_name(MemoryClass::Semantic),
                limit: self.options.limits.max_records_per_class,
            });
        }
        record.class = MemoryClass::Semantic;
        self.commit(vec![record], self.horizon)?;
        Ok(decision)
    }

    /// Applies the retention policy to every record in one atomic step and returns what changed.
    pub fn apply_retention(
        &mut self,
        policy: &ConsolidationPolicy,
        contradictions: &[Contradiction],
    ) -> Result<Vec<AppliedRetention>, MemoryStoreError> {
        let mut applied = Vec::new();
        let mut changed = Vec::new();
        for record in self.records.values() {
            let verdict = policy.decide_retention(record, self.clock, contradictions);
            let to = match verdict {
                RetentionVerdict::Keep => continue,
                RetentionVerdict::Demote => RecordState::NeedsRevalidation,
                RetentionVerdict::Forget => RecordState::Retired,
            };
            if !record.state.can_transition_to(to) {
                continue;
            }
            let mut next = record.clone();
            next.state = to;
            changed.push(next);
            applied.push(AppliedRetention {
                id: record.id.clone(),
                verdict,
            });
        }
        if !changed.is_empty() {
            self.commit(changed, self.horizon)?;
        }
        Ok(applied)
    }

    /// Rewrites the journal as a minimal snapshot (temp file + rename).
    pub fn compact(&mut self) -> Result<(), MemoryStoreError> {
        let mut payloads = vec![encode_delta(self.horizon, self.clock, &[])];
        for record in self.records.values() {
            payloads.push(encode_delta(
                self.horizon,
                self.clock,
                std::slice::from_ref(record),
            ));
        }
        self.journal.rewrite(&payloads)?;
        self.last_compaction_error = None;
        Ok(())
    }

    /// Journals the delta, then applies it. Memory is only touched after the append succeeded.
    fn commit(
        &mut self,
        mut changed: Vec<MemoryRecord>,
        horizon: Revision,
    ) -> Result<(), MemoryStoreError> {
        let clock = self.clock + 1;
        for record in &mut changed {
            record.updated_at = clock;
        }
        let payload = encode_delta(horizon, clock, &changed);
        self.journal.append(&payload)?;
        for record in changed {
            self.records.insert(record.id.clone(), record);
        }
        self.horizon = horizon;
        self.clock = clock;
        if let Some(limit) = self.options.compact_after_records {
            if self.journal.appended_since_rewrite() >= limit {
                // The mutation is already durable and applied: a failed rewrite must not be
                // reported as a failed mutation. Record it; the next commit retries.
                if let Err(error) = self.compact() {
                    self.compaction_failures += 1;
                    self.last_compaction_error = Some(error);
                }
            }
        }
        Ok(())
    }
}

struct Delta {
    horizon: Revision,
    clock: u64,
    records: Vec<MemoryRecord>,
}

fn encode_delta(horizon: Revision, clock: u64, records: &[MemoryRecord]) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(DELTA_TAG);
    w.u64(horizon.0);
    w.u64(clock);
    w.count(records.len());
    for record in records {
        write_record(&mut w, record);
    }
    w.finish()
}

fn decode_delta(payload: &[u8]) -> Result<Delta, CodecError> {
    let mut r = Reader::new(payload);
    let tag = r.u8()?;
    if tag != DELTA_TAG {
        return Err(CodecError::InvalidTag {
            field: "memory frame",
            tag,
        });
    }
    let horizon = Revision(r.u64()?);
    let clock = r.u64()?;
    let count = r.count()?;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        records.push(read_record(&mut r)?);
    }
    r.finish()?;
    Ok(Delta {
        horizon,
        clock,
        records,
    })
}
