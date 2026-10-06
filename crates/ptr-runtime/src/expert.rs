//! Admission of expert structural changes (birth, merge, split, freeze, reactivate, prune,
//! retire) into the authoritative ledger.
//!
//! An [`ExpertProposal`] is inert data. It becomes authority only through
//! [`PtrRuntime::admit_expert_proposal`], which checks, in this order and before anything is
//! appended:
//!
//! 1. the proposal's project and proposer are the ones the host authenticated
//!    ([`ExpertAdmissionGrant`]), never the caller's claim;
//! 2. the gate report is bound to exactly this proposal and this operation;
//! 3. the report was produced under the gate configuration the grant requires;
//! 4. the report passed;
//! 5. a compare-and-set against [`ExpertRegistryProjection`] for every operand (generation,
//!    manifest digest, state, state version, level, base model) and for every result (pod and
//!    routing slot absent, except an atomic takeover from an operand of the same operation).
//!
//! The record and the event's ids are then derived from the validated proposal, never taken from
//! the caller, and the event is appended through the runtime's internal path. The public
//! [`PtrRuntime::commit`] keeps refusing the event.
//!
//! The record committed is a version 2 [`ExpertCommitRecord`]: besides the report digest it embeds
//! the report's canonical bytes and the gate configuration digest the grant required for its
//! gate epoch.
//!
//! Replay (`replay`, `open_durable`) runs the same record checks without a grant: it decodes the
//! record, re-derives the op kind, the proposal id, the operation id, the evidence digest and the
//! project, refuses any disagreement with the event, and repeats the compare-and-set against the
//! projection as of that point of the history. It also re-verifies the evidence itself: the
//! embedded report must decode canonically, its re-derived digest must equal the record's
//! evidence digest, it must have passed, its subject must be this record's proposal and
//! operation, it must have been produced under the gate configuration the record says was
//! required, and no two records of one gate epoch may disagree on that configuration. A tampered,
//! reordered or truncated history therefore fails to open instead of producing a different
//! expert registry, and a record whose digest was merely made self-consistent is refused.
//!
//! # Legacy records
//!
//! A version 1 record (written before evidence was embedded) carries only a digest. Rewriting
//! history is not possible and refusing it would make existing ledgers unopenable, so replay
//! accepts it, applies its compare-and-set and marks every expert it touched
//! [`ExpertEvidenceStatus::UnverifiedLegacy`] in [`ExpertRegistryProjection::evidence_status`]
//! until a later verified record rewrites the expert. Admission never writes version 1, so a
//! legacy record cannot be admitted again, and it takes part in no epoch consistency check.
//!
//! # What this does not check
//!
//! The report is trusted to be what the host's evaluator produced (see `ptr-lineage`
//! `BoundGateReport`): neither admission nor replay re-runs the gate, and the digest is an
//! unkeyed hash, so replay shows the committed evidence is the evidence that was admitted, not
//! that the measurements are true. Nothing checks that attested training inputs are still live
//! or revalidates the proposer's session or the gate epoch at commit. The grant is a snapshot the
//! host builds from its own current authority. A compacted snapshot carries neither the legacy
//! marks nor the per-epoch configuration digests, so both restart empty from one.
//!
//! # Post-operation state
//!
//! [`ptr_pods::StructuralOp::outcomes`] is the single definition of the state each operand and
//! result is left in; this module stores it, `ptr-state` projects it.

use crate::{PtrRuntime, RuntimeError};
use ptr_ledger::{ExpertOpKind, LedgerEvent};
use ptr_lineage::BoundGateReport;
use ptr_pods::{
    BaseModelRef, ExpertCommitRecord, ExpertLevel, ExpertProposal, ExpertState, ManifestDigest,
    OpKind, ProposerRef, RetiredReason,
};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};
use std::collections::{BTreeMap, BTreeSet};

/// What the host has authenticated for one expert admission.
///
/// Only the host constructs a grant, from its own authority: the project and policy pod the
/// session's principal is granted, the gate epoch it currently stands at, and the digest of the
/// gate configuration it requires evidence to come from. Nothing in a proposal or a report can
/// supply or widen one of these, and the runtime does not remember a grant between calls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertAdmissionGrant {
    project: ProjectId,
    proposer: ProposerRef,
    gate_epoch: u64,
    required_gate_config_digest: [u8; 32],
}

impl ExpertAdmissionGrant {
    /// A grant for `proposer` (a policy pod and its generation, which carries its own project)
    /// in `project`. The proposer is named whole, so a principal authenticated as one policy
    /// pod cannot admit a proposal naming another.
    pub fn new(
        project: ProjectId,
        proposer: ProposerRef,
        gate_epoch: u64,
        required_gate_config_digest: [u8; 32],
    ) -> Self {
        Self {
            project,
            proposer,
            gate_epoch,
            required_gate_config_digest,
        }
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn proposer(&self) -> &ProposerRef {
        &self.proposer
    }

    pub fn gate_epoch(&self) -> u64 {
        self.gate_epoch
    }

    pub fn required_gate_config_digest(&self) -> &[u8; 32] {
        &self.required_gate_config_digest
    }
}

/// The compare-and-set field that disagreed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertStaleField {
    Generation,
    ManifestDigest,
    State,
    StateVersion,
    Level,
    BaseModel,
}

/// The event field that disagreed with the record it carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertEventField {
    Project,
    OpKind,
    ProposalId,
    OperationId,
    EvidenceDigest,
}

/// Why an expert structural change was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpertAdmissionError {
    /// The proposal names a project other than the one the host authenticated.
    ProjectMismatch,
    /// The proposal names a proposer other than the one the host authenticated.
    ProposerMismatch,
    /// The gate report is bound to another proposal.
    EvidenceProposalMismatch,
    /// The gate report is bound to another operation.
    EvidenceOperationMismatch,
    /// The gate report was produced under a gate configuration the grant (at admission) or the
    /// record (on replay) does not require.
    GateConfigMismatch,
    /// The embedded gate report does not decode canonically.
    EvidenceUndecodable(String),
    /// The embedded gate report's digest is not the record's evidence digest.
    EvidenceDigestMismatch,
    /// Two records of one gate epoch name different required gate configurations.
    EpochGateConfigConflict { epoch: u64 },
    /// The gate report did not pass.
    GateFailed,
    /// An operand is not in the registry.
    UnknownOperand { pod: String },
    /// An operand differs from the authoritative entry (compare-and-set failed).
    StaleOperand {
        pod: String,
        field: ExpertStaleField,
    },
    /// A result's pod id is already registered.
    ResultPodOccupied { pod: String },
    /// A result's routing slot is held by an expert the operation does not replace.
    ResultSlotOccupied { layer: u16, index: u16 },
    /// The record does not decode, or the evidence digest cannot be bound to it.
    InvalidRecord(String),
    /// The event cannot be framed by the ledger (size or count limits).
    Unencodable,
    /// The event disagrees with the record it carries.
    EventMismatch { field: ExpertEventField },
}

/// How the evidence behind an expert's latest structural record was checked on replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertEvidenceStatus {
    /// The record embeds its gate report and replay re-verified it.
    Verified,
    /// The record is a version 1 record with only a digest; nothing about its evidence could be
    /// re-verified.
    UnverifiedLegacy,
}

/// The authoritative state of one expert, as projected from admitted records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertEntry {
    pub state: ExpertState,
    pub state_version: u64,
    pub generation: Generation,
    pub manifest_digest: ManifestDigest,
    pub level: ExpertLevel,
    pub base_model: BaseModelRef,
}

/// Experts by `(project, pod)`, rebuilt from the ledger by applying admitted records in order.
///
/// It is a projection: it changes only when an admitted record is applied, which the runtime does
/// for every committed [`LedgerEvent::ExpertStructuralCommitted`], live and on replay.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExpertRegistryProjection {
    entries: BTreeMap<(String, String), ExpertEntry>,
    /// Experts whose latest record is unverified legacy; every other entry is verified.
    legacy: BTreeSet<(String, String)>,
    /// The gate configuration digest the first verified record of each gate epoch required.
    epoch_gate_configs: BTreeMap<u64, [u8; 32]>,
}

type Plan = Vec<((String, String), ExpertEntry)>;

/// What applying one record changes: the entries, and how its evidence was checked.
struct ExpertPlan {
    entries: Plan,
    verified: Option<VerifiedEvidence>,
}

/// The facts a verified record contributes to the projection.
struct VerifiedEvidence {
    epoch: u64,
    gate_config_digest: [u8; 32],
}

impl ExpertRegistryProjection {
    pub fn get(&self, project: &ProjectId, pod: &PodId) -> Option<&ExpertEntry> {
        self.entries.get(&(project.0.clone(), pod.0.clone()))
    }

    /// A registry holding exactly `entries`, with the evidence marks a compacted snapshot
    /// carries. The snapshot reader has already checked order, identities and that every mark
    /// names an entry, and cross-checks the result against the materialized `expert:` keys.
    pub(crate) fn from_parts(
        entries: Vec<((String, String), ExpertEntry)>,
        legacy: BTreeSet<(String, String)>,
        epoch_gate_configs: BTreeMap<u64, [u8; 32]>,
    ) -> Self {
        Self {
            entries: entries.into_iter().collect(),
            legacy,
            epoch_gate_configs,
        }
    }

    /// A registry restored from a snapshot that does not say how its evidence was checked
    /// (PTRCS005): every expert is unverified legacy and no epoch configuration is known.
    pub(crate) fn from_entries_all_legacy(entries: Vec<((String, String), ExpertEntry)>) -> Self {
        let legacy = entries.iter().map(|(key, _)| key.clone()).collect();
        Self::from_parts(entries, legacy, BTreeMap::new())
    }

    pub(crate) fn legacy_keys(&self) -> impl Iterator<Item = (&str, &str)> {
        self.legacy.iter().map(|(p, d)| (p.as_str(), d.as_str()))
    }

    pub(crate) fn epoch_configs(&self) -> impl Iterator<Item = (u64, [u8; 32])> + '_ {
        self.epoch_gate_configs.iter().map(|(e, d)| (*e, *d))
    }

    /// How the evidence behind `pod`'s latest structural record was checked, or `None` when the
    /// registry holds no such expert. An expert restored from a compacted snapshot reads as
    /// verified: the snapshot does not carry the legacy marks.
    pub fn evidence_status(
        &self,
        project: &ProjectId,
        pod: &PodId,
    ) -> Option<ExpertEvidenceStatus> {
        let key = (project.0.clone(), pod.0.clone());
        if !self.entries.contains_key(&key) {
            return None;
        }
        Some(if self.legacy.contains(&key) {
            ExpertEvidenceStatus::UnverifiedLegacy
        } else {
            ExpertEvidenceStatus::Verified
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every entry in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, &ExpertEntry)> {
        self.entries
            .iter()
            .map(|((project, pod), entry)| (project.as_str(), pod.as_str(), entry))
    }

    /// Compare-and-set `record` against this projection; the entries it would write.
    fn plan(&self, record: &ExpertCommitRecord) -> Result<Plan, ExpertAdmissionError> {
        let proposal = record.proposal();
        let project = proposal.project();
        let op = proposal.op();
        let key = |pod: &PodId| (project.0.clone(), pod.0.clone());
        let sources = op.sources();
        for source in &sources {
            let entry = self.entries.get(&key(source.pod())).ok_or_else(|| {
                ExpertAdmissionError::UnknownOperand {
                    pod: source.pod().0.clone(),
                }
            })?;
            let stale = |field| ExpertAdmissionError::StaleOperand {
                pod: source.pod().0.clone(),
                field,
            };
            if entry.generation != source.generation() {
                return Err(stale(ExpertStaleField::Generation));
            }
            if entry.manifest_digest != *source.manifest_digest() {
                return Err(stale(ExpertStaleField::ManifestDigest));
            }
            if entry.state != source.state() {
                return Err(stale(ExpertStaleField::State));
            }
            if entry.state_version != source.state_version() {
                return Err(stale(ExpertStaleField::StateVersion));
            }
            if entry.level != source.level() {
                return Err(stale(ExpertStaleField::Level));
            }
            if entry.base_model != *source.base_model() {
                return Err(stale(ExpertStaleField::BaseModel));
            }
        }
        let takeover = match op.kind() {
            OpKind::Reactivate => sources.first().map(|target| target.pod().clone()),
            _ => None,
        };
        for result in op.results() {
            if takeover.as_ref() != Some(result.pod())
                && self.entries.contains_key(&key(result.pod()))
            {
                return Err(ExpertAdmissionError::ResultPodOccupied {
                    pod: result.pod().0.clone(),
                });
            }
            if let ExpertLevel::Token { layer, index } = result.level() {
                let occupied = self
                    .entries
                    .iter()
                    .any(|((entry_project, entry_pod), entry)| {
                        entry_project == &project.0
                            && entry.level == result.level()
                            && occupies_slot(entry.state)
                            && !sources.iter().any(|source| source.pod().0 == *entry_pod)
                    });
                if occupied {
                    return Err(ExpertAdmissionError::ResultSlotOccupied { layer, index });
                }
            }
        }
        Ok(op
            .outcomes()
            .into_iter()
            .map(|outcome| {
                (
                    key(&outcome.pod),
                    ExpertEntry {
                        state: outcome.state,
                        state_version: outcome.state_version,
                        generation: outcome.generation,
                        manifest_digest: outcome.manifest_digest,
                        level: outcome.level,
                        base_model: outcome.base_model,
                    },
                )
            })
            .collect())
    }

    fn apply(&mut self, plan: ExpertPlan) {
        for (key, _) in &plan.entries {
            if plan.verified.is_some() {
                self.legacy.remove(key);
            } else {
                self.legacy.insert(key.clone());
            }
        }
        if let Some(verified) = plan.verified {
            self.epoch_gate_configs
                .entry(verified.epoch)
                .or_insert(verified.gate_config_digest);
        }
        self.entries.extend(plan.entries);
    }
}

/// Whether an expert in `state` still holds its routing slot. A withdrawn or retired expert has
/// released it; every other state, a fenced one included, keeps it.
fn occupies_slot(state: ExpertState) -> bool {
    !matches!(
        state,
        ExpertState::Superseded
            | ExpertState::Retired(RetiredReason::Pruned)
            | ExpertState::Retired(RetiredReason::Superseded)
            | ExpertState::Retired(RetiredReason::GateFailed)
            | ExpertState::Retired(RetiredReason::InputRevoked)
    )
}

/// The ledger code for a proposal-level operation kind.
pub fn ledger_op_kind(kind: OpKind) -> ExpertOpKind {
    match kind {
        OpKind::Birth => ExpertOpKind::Birth,
        OpKind::Merge => ExpertOpKind::Merge,
        OpKind::Split => ExpertOpKind::Split,
        OpKind::Freeze => ExpertOpKind::Freeze,
        OpKind::Reactivate => ExpertOpKind::Reactivate,
        OpKind::Prune => ExpertOpKind::Prune,
        OpKind::Retire => ExpertOpKind::Retire,
    }
}

/// The event for `record`, with every id derived from it.
fn event_for(record: &ExpertCommitRecord) -> LedgerEvent {
    let proposal = record.proposal();
    LedgerEvent::ExpertStructuralCommitted {
        project: proposal.project().clone(),
        op_kind: ledger_op_kind(proposal.op().kind()),
        proposal_id: *proposal.proposal_id().as_bytes(),
        operation_id: *proposal.operation_id().as_bytes(),
        evidence_digest: *record.evidence_digest(),
        record: record.encode_canonical(),
    }
}

fn check_evidence(
    proposal: &ExpertProposal,
    report: &BoundGateReport,
    grant: &ExpertAdmissionGrant,
) -> Result<(), ExpertAdmissionError> {
    if proposal.project() != grant.project() {
        return Err(ExpertAdmissionError::ProjectMismatch);
    }
    if proposal.proposer() != grant.proposer() {
        return Err(ExpertAdmissionError::ProposerMismatch);
    }
    let subject = report.subject();
    if subject.proposal_id != *proposal.proposal_id().as_bytes() {
        return Err(ExpertAdmissionError::EvidenceProposalMismatch);
    }
    if subject.operation_id != *proposal.operation_id().as_bytes() {
        return Err(ExpertAdmissionError::EvidenceOperationMismatch);
    }
    if report.gate_config_digest() != *grant.required_gate_config_digest() {
        return Err(ExpertAdmissionError::GateConfigMismatch);
    }
    if !report.passed() {
        return Err(ExpertAdmissionError::GateFailed);
    }
    Ok(())
}

impl PtrRuntime {
    /// The expert registry as projected from admitted structural records.
    pub fn expert_registry(&self) -> &ExpertRegistryProjection {
        &self.experts
    }

    /// Admit `proposal` on `report` under the host's `grant` and commit it.
    ///
    /// See the module documentation for the checks and their order. On success the record is
    /// committed and the registry updated; on any error nothing is appended and the registry is
    /// unchanged. This is the only path that writes
    /// [`LedgerEvent::ExpertStructuralCommitted`]; [`PtrRuntime::commit`] refuses it.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check, or the runtime's own errors when
    /// execution is fenced or the append fails.
    pub fn admit_expert_proposal(
        &mut self,
        proposal: &ExpertProposal,
        report: &BoundGateReport,
        grant: &ExpertAdmissionGrant,
    ) -> Result<CommitIndex, RuntimeError> {
        check_evidence(proposal, report, grant).map_err(RuntimeError::Expert)?;
        let record = ExpertCommitRecord::new_with_evidence(
            proposal.clone(),
            report.digest(),
            grant.gate_epoch(),
            *grant.required_gate_config_digest(),
            report.encode_canonical(),
        )
        .map_err(|error| {
            RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(error.to_string()))
        })?;
        let event = event_for(&record);
        self.validate_expert_event(&event)?;
        self.append_prepared(event, None)
    }

    /// Replay-time and admission-time check of a structural event: the record decodes, every id
    /// the event carries is re-derived from it, the ledger can frame it, and the compare-and-set
    /// holds against the registry as of now.
    pub(crate) fn validate_expert_event(&self, event: &LedgerEvent) -> Result<(), RuntimeError> {
        self.plan_expert_event(event).map(|_| ())
    }

    fn plan_expert_event(&self, event: &LedgerEvent) -> Result<ExpertPlan, RuntimeError> {
        let LedgerEvent::ExpertStructuralCommitted {
            project,
            op_kind,
            proposal_id,
            operation_id,
            evidence_digest,
            record,
        } = event
        else {
            return Err(RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(
                "not a structural event".to_owned(),
            )));
        };
        let expert = RuntimeError::Expert;
        let decoded = ExpertCommitRecord::decode_canonical(record)
            .map_err(|error| expert(ExpertAdmissionError::InvalidRecord(error.to_string())))?;
        let proposal = decoded.proposal();
        let mismatch = |field| expert(ExpertAdmissionError::EventMismatch { field });
        if project != proposal.project() {
            return Err(mismatch(ExpertEventField::Project));
        }
        if *op_kind != ledger_op_kind(proposal.op().kind()) {
            return Err(mismatch(ExpertEventField::OpKind));
        }
        if proposal_id != proposal.proposal_id().as_bytes() {
            return Err(mismatch(ExpertEventField::ProposalId));
        }
        if operation_id != proposal.operation_id().as_bytes() {
            return Err(mismatch(ExpertEventField::OperationId));
        }
        if evidence_digest != decoded.evidence_digest() {
            return Err(mismatch(ExpertEventField::EvidenceDigest));
        }
        ptr_ledger::check_encodable(event)
            .map_err(|_| expert(ExpertAdmissionError::Unencodable))?;
        let verified = self.verify_expert_evidence(&decoded).map_err(expert)?;
        let entries = self.experts.plan(&decoded).map_err(expert)?;
        Ok(ExpertPlan { entries, verified })
    }

    /// Re-verify the evidence a record embeds. `None` for a legacy record, which embeds none.
    fn verify_expert_evidence(
        &self,
        record: &ExpertCommitRecord,
    ) -> Result<Option<VerifiedEvidence>, ExpertAdmissionError> {
        let Some(evidence) = record.evidence() else {
            return Ok(None);
        };
        let report = BoundGateReport::decode_canonical(evidence.report())
            .map_err(|error| ExpertAdmissionError::EvidenceUndecodable(error.to_string()))?;
        if report.digest() != *record.evidence_digest() {
            return Err(ExpertAdmissionError::EvidenceDigestMismatch);
        }
        if !report.passed() {
            return Err(ExpertAdmissionError::GateFailed);
        }
        let proposal = record.proposal();
        let subject = report.subject();
        if subject.proposal_id != *proposal.proposal_id().as_bytes() {
            return Err(ExpertAdmissionError::EvidenceProposalMismatch);
        }
        if subject.operation_id != *proposal.operation_id().as_bytes() {
            return Err(ExpertAdmissionError::EvidenceOperationMismatch);
        }
        if report.gate_config_digest() != *evidence.required_gate_config_digest() {
            return Err(ExpertAdmissionError::GateConfigMismatch);
        }
        let epoch = record.gate_epoch();
        if let Some(known) = self.experts.epoch_gate_configs.get(&epoch) {
            if known != evidence.required_gate_config_digest() {
                return Err(ExpertAdmissionError::EpochGateConfigConflict { epoch });
            }
        }
        Ok(Some(VerifiedEvidence {
            epoch,
            gate_config_digest: *evidence.required_gate_config_digest(),
        }))
    }

    /// Apply a committed structural event to the registry. The event passed validation before it
    /// was appended, so the compare-and-set holding again is an invariant; a failure surfaces as
    /// an error instead of a silently diverging registry.
    pub(crate) fn apply_expert_event(&mut self, event: &LedgerEvent) -> Result<(), RuntimeError> {
        let plan = self.plan_expert_event(event)?;
        self.experts.apply(plan);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_proposal_kind_has_its_own_ledger_code() {
        let kinds = [
            OpKind::Birth,
            OpKind::Merge,
            OpKind::Split,
            OpKind::Freeze,
            OpKind::Reactivate,
            OpKind::Prune,
            OpKind::Retire,
        ];
        let codes: Vec<u8> = kinds.iter().map(|k| ledger_op_kind(*k).code()).collect();
        assert_eq!(codes, vec![0, 1, 2, 3, 4, 5, 6]);
    }
}
