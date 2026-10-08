//! The evidence-authority epoch: which gate configuration and which evaluator admission evidence
//! must come from, and what a change of either does to the experts admitted under the old one.
//!
//! Two authority kinds exist ([`ExpertAuthorityKind`]): the gate configuration and the evaluator.
//! Each has a *current standing*, an epoch and an identity digest, moved only by a committed
//! [`LedgerEvent::ExpertAuthorityChanged`] (ledger tag 31) that [`PtrRuntime::change_expert_authority`]
//! writes. A kind that no event ever moved has no standing and is not enforced, so a history
//! written before this module existed replays unchanged.
//!
//! # Evidence admission
//!
//! Once a kind has a standing, the one shared check [`ExpertRegistryProjection::check_authority`]
//! decides whether evidence is current. Admission of a structural proposal, readmission of a
//! fenced expert and replay of either all call it:
//!
//! * gate configuration: the gate epoch the evidence was produced at must equal the current
//!   epoch, and the required gate configuration digest must equal the current identity;
//! * evaluator: the grant must name the evaluator its report comes from
//!   ([`crate::expert::ExpertAdmissionGrant::with_evaluator`],
//!   [`crate::expert::ExpertFenceGrant::with_evaluator`]) and it must equal the current identity.
//!   A gate report carries no evaluator field, so the name is the host's assertion, not something
//!   the report proves; the committed record does not carry it either, so replay does not
//!   re-check it.
//!
//! A mismatch is the typed refusal [`ExpertAdmissionError::StalePolicyEvidence`] and nothing is
//! appended.
//!
//! # Fencing
//!
//! The registry remembers, per expert, the epoch of each kind it was admitted (or last
//! readmitted) under: the standing epoch at that point of the history, `0` when the kind had no
//! standing yet. A `Revoke` or `TighteningRotation` to epoch `E` fences, in the same event, every
//! `Serving` or `Frozen` expert whose epoch of that kind is below `E`, with cause
//! `AuthorityQuarantine`, and moves its fence epoch up by one. An expert admitted before the kind
//! had any standing is therefore fenced by the first such event: evidence from before the regime
//! cannot show it meets it. A `NonTightening` rotation fences nothing, and the ledger codec
//! refuses one that claims to. The event commits to the set by count and a domain-separated
//! digest ([`expert_authority_fence_set_digest`]); commit and replay both re-derive the set and
//! refuse a different one ([`ExpertAdmissionError::FenceSetMismatch`]).
//!
//! A quarantined expert is readmitted through the existing gate path, and only on evidence under
//! the current standing; the readmission records the current epochs as the expert's.
//!
//! # Limits
//!
//! The authority is runtime-wide: an event reaches the experts of every project, so it needs the
//! runtime-wide [`ExpertFenceAuthority::GlobalRevocation`] grant and refuses a project-scoped
//! one. The evaluator identity is a host assertion that replay does not re-check, and an expert
//! admitted before a kind had a standing counts as epoch 0, fenced by the first `Revoke` or
//! tightening of that kind. The grant is not recorded, so a ledger writer that bypasses
//! the public API is not defended against. Only the gate configuration and the evaluator are
//! kinds; attestor and verifier-set authorities do not exist.

use super::{
    fence_set_epochs, ExpertAdmissionError, ExpertFenceAuthority, ExpertFenceGrant,
    ExpertFenceSetMember, ExpertRegistryProjection, FenceSet,
};
use crate::{PtrRuntime, RuntimeError};
pub use ptr_ledger::{ExpertAuthorityClass, ExpertAuthorityKind};
use ptr_ledger::{ExpertFenceAction, ExpertFenceCause, ExpertFenceResume, LedgerEvent};
use ptr_pods::ExpertState;
use ptr_types::{CommitIndex, PodId, ProjectId};
use std::collections::BTreeMap;

/// The current epoch and identity of one authority kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertAuthorityStanding {
    epoch: u64,
    id_digest: [u8; 32],
}

impl ExpertAuthorityStanding {
    pub(crate) fn new(epoch: u64, id_digest: [u8; 32]) -> Self {
        Self { epoch, id_digest }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn id_digest(&self) -> &[u8; 32] {
        &self.id_digest
    }
}

/// The epochs an expert was admitted (or last readmitted) under, one per authority kind. `0`
/// means the kind had no standing at that point of the history.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertAdmissionEpochs {
    gate_config: u64,
    evaluator: u64,
}

impl ExpertAdmissionEpochs {
    pub(crate) fn new(gate_config: u64, evaluator: u64) -> Self {
        Self {
            gate_config,
            evaluator,
        }
    }

    /// The epoch of `kind` the expert's evidence was issued under.
    pub fn epoch(&self, kind: ExpertAuthorityKind) -> u64 {
        match kind {
            ExpertAuthorityKind::GateConfig => self.gate_config,
            ExpertAuthorityKind::Evaluator => self.evaluator,
        }
    }

    pub(crate) fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// The authority state of a registry: the standing of each kind and the epochs each expert was
/// admitted under. Projected from `ExpertAuthorityChanged` events and structural and readmission
/// records, so it is replayed and compacted.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct AuthorityState {
    gate_config: Option<ExpertAuthorityStanding>,
    evaluator: Option<ExpertAuthorityStanding>,
    /// Experts with a non-default entry only, so one canonical form exists.
    admissions: BTreeMap<(String, String), ExpertAdmissionEpochs>,
}

impl AuthorityState {
    pub(crate) fn from_parts(
        gate_config: Option<ExpertAuthorityStanding>,
        evaluator: Option<ExpertAuthorityStanding>,
        admissions: BTreeMap<(String, String), ExpertAdmissionEpochs>,
    ) -> Self {
        Self {
            gate_config,
            evaluator,
            admissions,
        }
    }

    /// Whether no kind has a standing: the state every history without a tag-31 event has.
    pub(crate) fn is_default(&self) -> bool {
        self.gate_config.is_none() && self.evaluator.is_none() && self.admissions.is_empty()
    }

    pub(crate) fn standing(&self, kind: ExpertAuthorityKind) -> Option<ExpertAuthorityStanding> {
        match kind {
            ExpertAuthorityKind::GateConfig => self.gate_config,
            ExpertAuthorityKind::Evaluator => self.evaluator,
        }
    }

    fn standing_slot(&mut self, kind: ExpertAuthorityKind) -> &mut Option<ExpertAuthorityStanding> {
        match kind {
            ExpertAuthorityKind::GateConfig => &mut self.gate_config,
            ExpertAuthorityKind::Evaluator => &mut self.evaluator,
        }
    }

    pub(crate) fn admission_entries(
        &self,
    ) -> impl Iterator<Item = (&str, &str, ExpertAdmissionEpochs)> {
        self.admissions
            .iter()
            .map(|((project, pod), epochs)| (project.as_str(), pod.as_str(), *epochs))
    }

    pub(crate) fn admission(&self, key: &(String, String)) -> ExpertAdmissionEpochs {
        self.admissions.get(key).copied().unwrap_or_default()
    }

    /// The epochs a piece of evidence admitted now is issued under: the current standing of
    /// each kind.
    fn epochs_now(&self) -> ExpertAdmissionEpochs {
        ExpertAdmissionEpochs::new(
            self.gate_config.map_or(0, |standing| standing.epoch),
            self.evaluator.map_or(0, |standing| standing.epoch),
        )
    }

    /// Record that the expert `key` was (re)admitted on verified evidence now, or on none (a
    /// legacy record), which leaves it with no epoch at all.
    pub(crate) fn record_admission(&mut self, key: &(String, String), verified: bool) {
        let epochs = if verified {
            self.epochs_now()
        } else {
            ExpertAdmissionEpochs::default()
        };
        if epochs.is_default() {
            self.admissions.remove(key);
        } else {
            self.admissions.insert(key.clone(), epochs);
        }
    }
}

/// The evaluator a piece of evidence is claimed to come from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EvaluatorClaim {
    /// Replay: the grant that named the evaluator is not recorded, so it is not re-checked.
    NotRechecked,
    /// Commit: the evaluator the grant names, if any.
    Named(Option<[u8; 32]>),
}

/// What a piece of evidence claims about the authority it was issued under.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EvidenceClaim {
    pub(crate) gate_epoch: u64,
    pub(crate) gate_config_digest: [u8; 32],
    pub(crate) evaluator: EvaluatorClaim,
}

/// Why evidence is not current, in [`ExpertAdmissionError::StalePolicyEvidence`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaleEvidenceReason {
    /// The evidence was produced at another epoch than the current one.
    EpochMismatch { current: u64, evidence: u64 },
    /// The evidence names another identity than the current one.
    IdentityMismatch,
    /// The kind has a standing but the grant names no identity for it.
    IdentityNotNamed,
}

/// One change of an authority: the kind, the epoch and identity it moves to and what it does to
/// evidence issued under earlier epochs. See [`PtrRuntime::change_expert_authority`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertAuthorityChange {
    kind: ExpertAuthorityKind,
    id_digest: [u8; 32],
    epoch: u64,
    class: ExpertAuthorityClass,
}

impl ExpertAuthorityChange {
    /// Move `kind` to `epoch` and `id_digest`, classified as `class`.
    pub fn new(
        kind: ExpertAuthorityKind,
        id_digest: [u8; 32],
        epoch: u64,
        class: ExpertAuthorityClass,
    ) -> Self {
        Self {
            kind,
            id_digest,
            epoch,
            class,
        }
    }

    pub fn kind(&self) -> ExpertAuthorityKind {
        self.kind
    }

    pub fn id_digest(&self) -> &[u8; 32] {
        &self.id_digest
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn class(&self) -> ExpertAuthorityClass {
        self.class
    }
}

/// What a committed [`PtrRuntime::change_expert_authority`] did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertAuthorityOutcome {
    index: CommitIndex,
    change: ExpertAuthorityChange,
    fence_digest: [u8; 32],
    fenced: Vec<ExpertFenceSetMember>,
}

impl ExpertAuthorityOutcome {
    /// The commit index of the one `ExpertAuthorityChanged` event.
    pub fn index(&self) -> CommitIndex {
        self.index
    }

    pub fn change(&self) -> &ExpertAuthorityChange {
        &self.change
    }

    /// The number of experts the event fenced, as the event carries it.
    pub fn fence_count(&self) -> u32 {
        // The count was checked to fit the event's `u32` before the event was built.
        self.fenced.len() as u32
    }

    /// The digest the event carries over the fence set.
    pub fn fence_digest(&self) -> &[u8; 32] {
        &self.fence_digest
    }

    /// The fenced experts in canonical order. Held for the caller only: the event does not list
    /// them.
    pub fn fenced(&self) -> &[ExpertFenceSetMember] {
        &self.fenced
    }

    /// The fence epoch now in force per `expert:{project}/{pod}` identity, the map a standalone
    /// KV backend takes (`ReferenceKvReuseBackend::apply_authority_change`).
    pub fn fence_epochs(&self) -> BTreeMap<String, u64> {
        fence_set_epochs(&self.fenced)
    }
}

const AUTHORITY_FENCE_SET_DOMAIN: &str = "ptr.expert.authority-change.fence-set.v1";

/// The digest an `ExpertAuthorityChanged` event carries over its fence set: SHA-256 over the
/// length-prefixed domain `ptr.expert.authority-change.fence-set.v1`, the kind code, the epoch,
/// the class code, the member count and, for each member in the order given (canonical order is
/// ascending `(project, pod)`), the length-prefixed project and pod, the expert generation, the
/// expected state version, the action code and the resume code. Integers are little-endian `u64`,
/// codes are one byte. A harness uses this to check the digest of an event against the set it
/// independently derived.
pub fn expert_authority_fence_set_digest(
    kind: ExpertAuthorityKind,
    epoch: u64,
    class: ExpertAuthorityClass,
    members: &[ExpertFenceSetMember],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    super::put_text(&mut bytes, AUTHORITY_FENCE_SET_DOMAIN);
    bytes.push(kind.code());
    bytes.extend_from_slice(&epoch.to_le_bytes());
    bytes.push(class.code());
    bytes.extend_from_slice(&(members.len() as u64).to_le_bytes());
    for member in members {
        super::put_text(&mut bytes, &member.project.0);
        super::put_text(&mut bytes, &member.pod.0);
        bytes.extend_from_slice(&member.generation.0.to_le_bytes());
        bytes.extend_from_slice(&member.expected_state_version.to_le_bytes());
        bytes.push(member.action.code());
        bytes.push(member.resume.code());
    }
    ptr_ledger::integrity::sha256(&bytes)
}

impl ExpertRegistryProjection {
    /// The current standing of `kind`, or `None` when no event ever moved it (it is then not
    /// enforced).
    pub fn authority(&self, kind: ExpertAuthorityKind) -> Option<ExpertAuthorityStanding> {
        self.authority.standing(kind)
    }

    /// The epochs the expert `pod` of `project` was admitted (or last readmitted) under, or
    /// `None` when the registry holds no such expert.
    pub fn admission_epochs(
        &self,
        project: &ProjectId,
        pod: &PodId,
    ) -> Option<ExpertAdmissionEpochs> {
        let key = (project.0.clone(), pod.0.clone());
        self.entries
            .contains_key(&key)
            .then(|| self.authority.admission(&key))
    }

    /// The shared evidence-currency check for admission, readmission and replay; see the module
    /// documentation. A kind without a standing is not enforced.
    pub(crate) fn check_authority(
        &self,
        claim: &EvidenceClaim,
    ) -> Result<(), ExpertAdmissionError> {
        let stale = |kind, reason| ExpertAdmissionError::StalePolicyEvidence { kind, reason };
        if let Some(current) = self.authority.gate_config {
            if claim.gate_epoch != current.epoch {
                return Err(stale(
                    ExpertAuthorityKind::GateConfig,
                    StaleEvidenceReason::EpochMismatch {
                        current: current.epoch,
                        evidence: claim.gate_epoch,
                    },
                ));
            }
            if claim.gate_config_digest != current.id_digest {
                return Err(stale(
                    ExpertAuthorityKind::GateConfig,
                    StaleEvidenceReason::IdentityMismatch,
                ));
            }
        }
        if let (Some(current), EvaluatorClaim::Named(named)) =
            (self.authority.evaluator, claim.evaluator)
        {
            match named {
                None => {
                    return Err(stale(
                        ExpertAuthorityKind::Evaluator,
                        StaleEvidenceReason::IdentityNotNamed,
                    ))
                }
                Some(id) if id != current.id_digest => {
                    return Err(stale(
                        ExpertAuthorityKind::Evaluator,
                        StaleEvidenceReason::IdentityMismatch,
                    ))
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// The experts a change to `epoch` of `kind` classified `class` fences, derived from the
    /// registry as it stands, in canonical `(project, pod)` order, with the digest the event
    /// commits to. Only a `Serving` or `Frozen` expert whose admission epoch of `kind` is below
    /// `epoch` is a member; an expert already fenced, retired or not yet serving is not.
    pub(crate) fn authority_fence_set(
        &self,
        kind: ExpertAuthorityKind,
        epoch: u64,
        class: ExpertAuthorityClass,
    ) -> Result<FenceSet, ExpertAdmissionError> {
        let mut members = Vec::new();
        if class.fences_earlier_evidence() {
            for (key, entry) in &self.entries {
                let resume = match entry.state {
                    ExpertState::Serving => ExpertFenceResume::Serving,
                    ExpertState::Frozen => ExpertFenceResume::Frozen,
                    ExpertState::Candidate
                    | ExpertState::Gated
                    | ExpertState::Fenced
                    | ExpertState::Superseded
                    | ExpertState::Retired(_) => continue,
                };
                if self.authority.admission(key).epoch(kind) >= epoch {
                    continue;
                }
                if entry.state_version.checked_add(1).is_none() {
                    return Err(ExpertAdmissionError::FenceEpochMismatch { pod: key.1.clone() });
                }
                members.push(ExpertFenceSetMember {
                    project: ProjectId(key.0.clone()),
                    pod: PodId(key.1.clone()),
                    generation: entry.generation,
                    expected_state_version: entry.state_version,
                    action: ExpertFenceAction::Fence,
                    resume,
                });
            }
        }
        let count = u32::try_from(members.len()).map_err(|_| ExpertAdmissionError::Unencodable)?;
        let digest = expert_authority_fence_set_digest(kind, epoch, class, &members);
        Ok(FenceSet {
            members,
            count,
            digest,
        })
    }

    /// Apply a validated change: the new standing, then the fences of `set` in one step with
    /// cause `AuthorityQuarantine`.
    pub(crate) fn apply_authority_change(
        &mut self,
        change: &ExpertAuthorityChange,
        set: &FenceSet,
    ) {
        *self.authority.standing_slot(change.kind) = Some(ExpertAuthorityStanding {
            epoch: change.epoch,
            id_digest: change.id_digest,
        });
        self.apply_fence_set(set, ExpertFenceCause::AuthorityQuarantine);
    }
}

impl PtrRuntime {
    /// The current standing of `kind`, or `None` when no committed change ever moved it.
    pub fn expert_authority(&self, kind: ExpertAuthorityKind) -> Option<ExpertAuthorityStanding> {
        self.experts.authority(kind)
    }

    /// Move the authority `change` names and, for a `Revoke` or a `TighteningRotation`, fence
    /// every expert admitted under an earlier epoch of that kind, in ONE committed
    /// [`LedgerEvent::ExpertAuthorityChanged`]. Needs a grant of
    /// [`ExpertFenceAuthority::GlobalRevocation`] ([`ExpertFenceGrant::global_revocation`]),
    /// revalidated at commit like a fence: the change fences experts of every project, so a
    /// project-scoped [`ExpertFenceAuthority::RuntimeRevocation`] grant is refused.
    ///
    /// The fence set is derived here, deterministically, from the registry at this commit index:
    /// the event carries its count and digest only, and applying it fences every member
    /// atomically with cause `AuthorityQuarantine`. A `NonTightening` change fences nobody.
    ///
    /// Refused, typed and with nothing appended: a gate grant; an epoch that does not exceed the
    /// current epoch of the kind ([`ExpertAdmissionError::AuthorityEpochNotIncreasing`]); a
    /// `Revoke` that leaves the revoked identity current
    /// ([`ExpertAdmissionError::AuthorityIdentityUnchanged`]). Commit and replay both re-derive
    /// the set and refuse a different count or digest
    /// ([`ExpertAdmissionError::FenceSetMismatch`]).
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check, or the runtime's own errors when
    /// execution is fenced or the append fails.
    pub fn change_expert_authority(
        &mut self,
        grant: &ExpertFenceGrant,
        change: ExpertAuthorityChange,
    ) -> Result<ExpertAuthorityOutcome, RuntimeError> {
        let expert = RuntimeError::Expert;
        if grant.authority() != ExpertFenceAuthority::GlobalRevocation {
            return Err(expert(ExpertAdmissionError::FenceAuthorityMismatch {
                action: ExpertFenceAction::Fence,
                required: ExpertFenceAuthority::GlobalRevocation,
            }));
        }
        let set = self
            .experts
            .authority_fence_set(change.kind, change.epoch, change.class)
            .map_err(expert)?;
        let event = LedgerEvent::ExpertAuthorityChanged {
            kind: change.kind,
            id_digest: change.id_digest,
            epoch: change.epoch,
            class: change.class,
            fence_count: set.count,
            fence_digest: set.digest,
        };
        // The same validation replay applies, including re-deriving the set.
        let planned = self.plan_authority_change(&event)?;
        self.revalidate_grant(grant.kind(), grant.binding())?;
        let index = self.append_prepared(event, None)?;
        Ok(ExpertAuthorityOutcome {
            index,
            change,
            fence_digest: planned.digest,
            fenced: planned.members,
        })
    }

    /// Replay-time and commit-time check of an `ExpertAuthorityChanged` event against the history
    /// as of now.
    pub(crate) fn validate_authority_changed_event(
        &self,
        event: &LedgerEvent,
    ) -> Result<(), RuntimeError> {
        self.plan_authority_change(event).map(|_| ())
    }

    fn plan_authority_change(&self, event: &LedgerEvent) -> Result<FenceSet, RuntimeError> {
        let LedgerEvent::ExpertAuthorityChanged {
            kind,
            id_digest,
            epoch,
            class,
            fence_count,
            fence_digest,
        } = event
        else {
            return Err(RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(
                "not an authority change".to_owned(),
            )));
        };
        let expert = RuntimeError::Expert;
        ptr_ledger::check_encodable(event)
            .map_err(|_| expert(ExpertAdmissionError::Unencodable))?;
        if let Some(current) = self.experts.authority(*kind) {
            if *epoch <= current.epoch() {
                return Err(expert(ExpertAdmissionError::AuthorityEpochNotIncreasing {
                    kind: *kind,
                    current: current.epoch(),
                    found: *epoch,
                }));
            }
            if *class == ExpertAuthorityClass::Revoke && *current.id_digest() == *id_digest {
                return Err(expert(ExpertAdmissionError::AuthorityIdentityUnchanged {
                    kind: *kind,
                }));
            }
        }
        let set = self
            .experts
            .authority_fence_set(*kind, *epoch, *class)
            .map_err(expert)?;
        if set.count != *fence_count || set.digest != *fence_digest {
            return Err(expert(ExpertAdmissionError::FenceSetMismatch {
                derived_count: set.count,
                event_count: *fence_count,
                digest_matches: set.digest == *fence_digest,
            }));
        }
        Ok(set)
    }

    /// Apply a committed `ExpertAuthorityChanged` event: the new standing, the fences of every
    /// member of the derived set in one step, the materialized `expert:` keys of the fenced
    /// experts and the KV catalog entries that rest on them.
    pub(crate) fn apply_authority_changed_event(
        &mut self,
        event: &LedgerEvent,
    ) -> Result<(), RuntimeError> {
        let LedgerEvent::ExpertAuthorityChanged {
            kind,
            id_digest,
            epoch,
            class,
            ..
        } = event
        else {
            return Ok(());
        };
        // Everything fallible happens before the first mutation.
        let set = self.plan_authority_change(event)?;
        self.invalidate_kv_expert_fences(&fence_set_epochs(&set.members));
        let change = ExpertAuthorityChange::new(*kind, *id_digest, *epoch, *class);
        self.experts.apply_authority_change(&change, &set);
        for member in &set.members {
            self.state.values.insert(
                ptr_state::expert_key(&member.project.0, &member.pod.0),
                ptr_state::expert_entry(member.generation.0, ExpertState::Fenced, member.epoch()),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with(gate: Option<u64>) -> AuthorityState {
        AuthorityState::from_parts(
            gate.map(|epoch| ExpertAuthorityStanding::new(epoch, [1; 32])),
            None,
            BTreeMap::new(),
        )
    }

    #[test]
    fn a_default_state_enforces_nothing_and_records_no_admission() {
        let mut state = AuthorityState::default();
        assert!(state.is_default());
        let key = ("p".to_owned(), "a".to_owned());
        state.record_admission(&key, true);
        assert!(
            state.is_default(),
            "epoch 0 is the default and is not stored"
        );
        assert_eq!(state.admission(&key), ExpertAdmissionEpochs::default());
    }

    #[test]
    fn an_admission_takes_the_current_standing_and_a_legacy_one_takes_none() {
        let mut state = state_with(Some(4));
        let key = ("p".to_owned(), "a".to_owned());
        state.record_admission(&key, true);
        assert_eq!(
            state.admission(&key).epoch(ExpertAuthorityKind::GateConfig),
            4
        );
        assert_eq!(
            state.admission(&key).epoch(ExpertAuthorityKind::Evaluator),
            0
        );
        state.record_admission(&key, false);
        assert!(state.is_default() || state.standing(ExpertAuthorityKind::GateConfig).is_some());
        assert_eq!(state.admission(&key), ExpertAdmissionEpochs::default());
    }

    #[test]
    fn an_empty_registry_checks_the_claim_only_against_a_standing() {
        let claim = EvidenceClaim {
            gate_epoch: 9,
            gate_config_digest: [2; 32],
            evaluator: EvaluatorClaim::Named(None),
        };
        let registry = ExpertRegistryProjection::default();
        assert!(registry.check_authority(&claim).is_ok());
        let registry = ExpertRegistryProjection::default().with_authority(state_with(Some(5)));
        assert_eq!(
            registry.check_authority(&claim),
            Err(ExpertAdmissionError::StalePolicyEvidence {
                kind: ExpertAuthorityKind::GateConfig,
                reason: StaleEvidenceReason::EpochMismatch {
                    current: 5,
                    evidence: 9
                },
            })
        );
        let mut current = claim;
        current.gate_epoch = 5;
        assert_eq!(
            registry.check_authority(&current),
            Err(ExpertAdmissionError::StalePolicyEvidence {
                kind: ExpertAuthorityKind::GateConfig,
                reason: StaleEvidenceReason::IdentityMismatch,
            })
        );
        current.gate_config_digest = [1; 32];
        assert!(registry.check_authority(&current).is_ok());
    }

    #[test]
    fn the_fence_set_digest_binds_every_field() {
        let member = |pod: &str| ExpertFenceSetMember {
            project: ProjectId::from("p"),
            pod: PodId::from(pod),
            generation: ptr_types::Generation(1),
            expected_state_version: 2,
            action: ExpertFenceAction::Fence,
            resume: ExpertFenceResume::Serving,
        };
        let members = [member("a"), member("b")];
        let base = expert_authority_fence_set_digest(
            ExpertAuthorityKind::GateConfig,
            3,
            ExpertAuthorityClass::Revoke,
            &members,
        );
        let variants = [
            expert_authority_fence_set_digest(
                ExpertAuthorityKind::Evaluator,
                3,
                ExpertAuthorityClass::Revoke,
                &members,
            ),
            expert_authority_fence_set_digest(
                ExpertAuthorityKind::GateConfig,
                4,
                ExpertAuthorityClass::Revoke,
                &members,
            ),
            expert_authority_fence_set_digest(
                ExpertAuthorityKind::GateConfig,
                3,
                ExpertAuthorityClass::TighteningRotation,
                &members,
            ),
            expert_authority_fence_set_digest(
                ExpertAuthorityKind::GateConfig,
                3,
                ExpertAuthorityClass::Revoke,
                &members[..1],
            ),
        ];
        for variant in variants {
            assert_ne!(variant, base);
        }
    }
}
