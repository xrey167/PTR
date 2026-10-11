//! F5 and F6: the signed attestor mechanism.
//!
//! F5, the binding of an attestation: for the three operations that produce an artifact (a
//! `Birth`, a `Merge` and the `Reactivate` of a retired seed) an attestation, signed by a
//! registered attestor, that names the proposal, the artifact, the base model and the results
//! the operation determines is admitted (the positive control); the same proposal paired with
//! a validly signed attestation of another artifact, another proposal or another base model is
//! refused; an attestor nobody registered (it signs with a key of its own) and a registered
//! attestor's id carrying another key's signature are refused; a grant that requires
//! attestation refuses the admission that carries none; and a run id is single-use: the second
//! record that names it is refused whoever the attestor is, after a reopen of the durable
//! ledger, and on the plain admission path that names the run as an input of its own, as a
//! training input at generation 1 or 2 or as an evaluation input. The recorded requirement: an
//! admission under a grant that requires attestation records the requirement, so after a
//! restart with a looser grant an unattested artifact-producing admission is refused, and so
//! is a recorded unattested record in a replay, while an operation that produces no artifact
//! is not blocked.
//!
//! F6, the attestor's epoch: a proposal queued across the revocation of its attestor is refused
//! at the commit (a fresh attestor is admitted); the experts admitted under a revoked attestor
//! are fenced with the terminal cause `InputRevoked` by the one compound event (never
//! readmitted, never a `Reactivate` seed, retired by the runtime alone) and a new epoch of the
//! attestor admits a retrained expert.

use super::dormant::Path;
use super::{
    admit_with_attestation, admit_with_attestation_under, fresh, must_attested, must_under,
    refused_with, reopen, replay_attested, run_id, run_identity, Attestor, Cx, INTRUDER,
};
use crate::experiments::layered_kv::lab::Res;
use crate::experiments::r011::cases::fixture::{
    level_of, readmission_report, report_for, spec_at, ReportSpec, Rt, Setup,
};
use crate::experiments::r011::cases::{classify_bystander, is_where};
use ptr_ledger::{CommittedEvent, ExpertFenceCause, LedgerEvent};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertInputs, ExpertProposal,
    ExpertState, RetiredReason, StructuralOp,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, ledger_op_kind, ExpertAdmissionError, ExpertFenceGrant,
    ExpertFenceRequest, ExpertStaleField, InputInvalidation,
};
use ptr_runtime::RuntimeError;
use ptr_types::{CommitIndex, Generation};

/// The caller's attempt to make `who`'s key trusted on a live runtime through `forged`. The
/// product has no call that does it: the trusted set is fixed when the runtime is built, so
/// there is nothing to try and the body is empty. (The mutation plan puts the live registration
/// that this experiment's review found back, here and in the product, to show the case sees it.)
fn try_to_trust(rt: &mut Rt, forged: &ExpertFenceGrant, who: Attestor) {
    let _ = (rt, forged, who);
}

const A1: Attestor = Attestor::new("alpha", 1);
const A2: Attestor = Attestor::new("alpha", 2);
const B1: Attestor = Attestor::new("beta", 1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Op {
    Birth,
    Merge,
    Reactivate,
}

impl Op {
    const ALL: [Op; 3] = [Self::Birth, Self::Merge, Self::Reactivate];

    fn key(self) -> &'static str {
        match self {
            Self::Birth => "birth",
            Self::Merge => "merge",
            Self::Reactivate => "reactivate",
        }
    }

    /// A fresh runtime in which the operation can be proposed, and the operation.
    fn fresh(self) -> Res<(Rt, StructuralOp)> {
        let mut rt = fresh()?;
        let op = match self {
            Self::Birth => StructuralOp::Birth {
                result: rt.spec("n"),
            },
            Self::Merge => {
                rt.birth("a")?;
                rt.birth("b")?;
                StructuralOp::Merge {
                    sources: vec![rt.target("a")?, rt.target("b")?],
                    result: rt.spec("m"),
                }
            }
            Self::Reactivate => {
                rt.birth("a")?;
                rt.retire("a", RetiredReason::GateFailed)?;
                let target = rt.target("a")?;
                let next = spec_at(&rt.project, "a", target.generation().0 + 1, target.level());
                StructuralOp::Reactivate {
                    target,
                    result: Some(next),
                }
            }
        };
        Ok((rt, op))
    }

    /// The pod the operation's result is routed under.
    fn result_pod(self) -> &'static str {
        match self {
            Self::Birth => "n",
            Self::Merge => "m",
            Self::Reactivate => "a",
        }
    }
}

/// The artifact digest of an operation other than any `Op` above produces.
fn other_artifact() -> Res<[u8; 32]> {
    let rt = fresh()?;
    let op = StructuralOp::Birth {
        result: rt.spec("q"),
    };
    let proposal = rt.proposal(op)?;
    Ok(expert_evidence_artifact_digest(proposal.op()))
}

/// The base model every expert of the fixtures is on.
fn base_model() -> BaseModelRef {
    crate::experiments::r011::cases::fixture::base()
}

/// Another base model than the fixtures'.
fn other_base_model() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("r013-other-base").expect("a valid base model id"),
        BaseModelDigest::new([0x77; 32]).expect("a non-zero digest"),
        1,
    )
    .expect("a non-zero generation")
}

pub fn attestation_binding(cx: &mut Cx) -> Res<()> {
    for op in cx.shuffled("attestor.ops", &Op::ALL) {
        cx.sub(&format!("binding_{}", op.key()));
        binding(cx, op)?;
    }
    cx.sub("run_reuse");
    run_reuse(cx)?;
    cx.sub("recorded_requirement");
    recorded_requirement(cx)?;
    cx.sub("binding_replayed");
    binding_replayed(cx)
}

fn binding(cx: &mut Cx, op: Op) -> Res<()> {
    // The control: the attestation names this proposal and its artifact.
    let (mut rt, operation) = op.fresh()?;
    let proposal = rt.proposal(operation)?;
    let attestation = A1.attest(&proposal, 1)?;
    admit_with_attestation(
        &mut rt,
        &proposal,
        &attestation,
        ExpertInputs::default(),
        &ReportSpec::correct(),
        Setup::PRE,
    )
    .map_err(|error| format!("the attested {} control: {error:?}", op.key()))?;
    let generation = rt.init(op.result_pod())?.generation.0;
    if !rt.routable(op.result_pod(), generation) {
        return Err(format!("the attested {} does not route", op.key()));
    }
    cx.count("attestation_binding_controls_accepted", 1);
    cx.count_in("attestation_controls_by_operation", op.key(), 1);
    cx.check_replay(&rt)?;

    // The artifact of another run under the same proposal.
    let other = other_artifact()?;
    let (mut rt, operation) = op.fresh()?;
    let proposal = rt.proposal(operation)?;
    let attestation = A1.attest_with(
        &proposal,
        2,
        *proposal.proposal_id().as_bytes(),
        other,
        base_model(),
    )?;
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_artifact_mismatch",
        "an attestation of another artifact",
        admit_with_attestation(
            &mut rt,
            &proposal,
            &attestation,
            ExpertInputs::default(),
            &ReportSpec::correct(),
            Setup::PRE,
        ),
        refused_with(|e| matches!(e, ExpertAdmissionError::AttestationArtifactMismatch)),
    )?;
    cx.count("attestation_artifact_mismatch_attempts", 1);
    cx.count_in("attestation_mismatch_attempts_by_operation", op.key(), 1);
    cx.count("mismatched_attestation_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;

    // The attestation of another proposal.
    let (mut rt, operation) = op.fresh()?;
    let proposal = rt.proposal(operation)?;
    let attestation = A1.attest_with(
        &proposal,
        3,
        [0x55; 32],
        expert_evidence_artifact_digest(proposal.op()),
        base_model(),
    )?;
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_proposal_mismatch",
        "an attestation of another proposal",
        admit_with_attestation(
            &mut rt,
            &proposal,
            &attestation,
            ExpertInputs::default(),
            &ReportSpec::correct(),
            Setup::PRE,
        ),
        refused_with(|e| matches!(e, ExpertAdmissionError::AttestationProposalMismatch)),
    )?;
    cx.count("attestation_proposal_mismatch_attempts", 1);
    cx.count("mismatched_attestation_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;

    // The base model of another model, validly signed.
    let (mut rt, operation) = op.fresh()?;
    let proposal = rt.proposal(operation)?;
    let attestation = A1.attest_with(
        &proposal,
        5,
        *proposal.proposal_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        other_base_model(),
    )?;
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_base_model_mismatch",
        "an attestation of another base model",
        admit_with_attestation(
            &mut rt,
            &proposal,
            &attestation,
            ExpertInputs::default(),
            &ReportSpec::correct(),
            Setup::PRE,
        ),
        refused_with(|e| matches!(e, ExpertAdmissionError::AttestationBaseModelMismatch)),
    )?;
    cx.count("attestation_base_model_mismatch_attempts", 1);
    cx.count("mismatched_attestation_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;

    // An attestor nobody registered, signing with a key of its own: the caller invented it.
    let (mut rt, operation) = op.fresh()?;
    let proposal = rt.proposal(operation)?;
    let attestation = INTRUDER.attest(&proposal, 6)?;
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_self_invented_attestor",
        "an attestation by an attestor the host never registered",
        admit_with_attestation(
            &mut rt,
            &proposal,
            &attestation,
            ExpertInputs::default(),
            &ReportSpec::correct(),
            Setup::PRE,
        ),
        refused_with(|e| matches!(e, ExpertAdmissionError::AttestorUnknown { .. })),
    )?;
    cx.count("attestation_untrusted_attestor_attempts", 1);
    cx.count("untrusted_attestor_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;

    // A registered attestor's id with the signature of another key.
    let (mut rt, operation) = op.fresh()?;
    let proposal = rt.proposal(operation)?;
    let attestation = A1.forged_by(INTRUDER, &proposal, 7)?;
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_forged_signature",
        "an attestation of a registered attestor signed by another key",
        admit_with_attestation(
            &mut rt,
            &proposal,
            &attestation,
            ExpertInputs::default(),
            &ReportSpec::correct(),
            Setup::PRE,
        ),
        refused_with(|e| matches!(e, ExpertAdmissionError::AttestationSignatureInvalid)),
    )?;
    cx.count("attestation_untrusted_attestor_attempts", 1);
    cx.count("untrusted_attestor_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;

    // A caller holding only an admission grant assembles a gate fence grant from that grant's
    // binding and tries to make its own key trusted, then signs with it and admits.
    let (mut rt, operation) = op.fresh()?;
    let proposal = rt.proposal(operation)?;
    let admission = rt.admission_grant(Setup::PRE)?;
    let forged = ExpertFenceGrant::gate(
        rt.project.clone(),
        Setup::PRE.requirement()?,
        admission.binding().clone(),
    );
    try_to_trust(&mut rt, &forged, INTRUDER);
    let trusted = rt
        .runtime
        .expert_attestors()
        .key(INTRUDER.name, Generation(INTRUDER.epoch))
        .is_some();
    let attestation = INTRUDER.attest(&proposal, 8)?;
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_forged_gate_grant",
        "an attestation signed by a key the caller tried to trust through a forged gate grant",
        admit_with_attestation_under(
            &mut rt,
            &proposal,
            &attestation,
            ExpertInputs::default(),
            &ReportSpec::correct(),
            &admission,
        ),
        refused_with(|e| matches!(e, ExpertAdmissionError::AttestorUnknown { .. })),
    )?;
    cx.count("forged_gate_grant_attempts", 1);
    cx.count(
        "self_trusted_attestor_admissions",
        i64::from(trusted) + verdict.violation(),
    );
    expect_unchanged(&rt, events, verdict.admitted())?;

    // A grant that requires attestation refuses an admission without one (with a block of its
    // own and without), and admits the attested one.
    let (mut rt, operation) = op.fresh()?;
    let proposal = rt.proposal(operation)?;
    let grant = rt.admission_grant(Setup::PRE)?.with_required_attestation();
    let block = ExpertInputs::default();
    for with_block in [false, true] {
        // A fresh runtime for each attempt: a runtime that wrongly admitted the first would
        // answer the second with another error, and the violation would be lost to a refusal
        // the case does not owe.
        let (mut attempt, attempt_op) = op.fresh()?;
        let attempt_proposal = attempt.proposal(attempt_op)?;
        let attempt_grant = attempt
            .admission_grant(Setup::PRE)?
            .with_required_attestation();
        let inputs = with_block.then(|| block.clone());
        let report = report_for(&attempt_proposal, inputs.as_ref(), &ReportSpec::correct())?;
        let events = attempt.event_count();
        let verdict = cx.verdict(
            "attestation_missing_under_requirement",
            "an admission without an attestation under a grant that requires one",
            attempt.commit_proposal(&attempt_proposal, &report, &attempt_grant, inputs),
            refused_with(|e| matches!(e, ExpertAdmissionError::AttestationRequired)),
        )?;
        cx.count("attestation_missing_attempts", 1);
        cx.count("unattested_admissions", verdict.violation());
        expect_unchanged(&attempt, events, verdict.admitted())?;
        if op == Op::Birth && !verdict.admitted() && attempt.state(op.result_pod()).is_some() {
            return Err("an unattested birth left an expert behind".to_string());
        }
    }
    let attestation = A1.attest(&proposal, 4)?;
    admit_with_attestation_under(
        &mut rt,
        &proposal,
        &attestation,
        ExpertInputs::default(),
        &ReportSpec::correct(),
        &grant,
    )
    .map_err(|error| format!("the attested {} under the requirement: {error:?}", op.key()))?;
    cx.count("attestation_binding_controls_accepted", 1);
    cx.count_in("attestation_controls_by_operation", op.key(), 1);
    Ok(())
}

/// A refused attempt appended nothing.
fn expect_unchanged(rt: &Rt, events: usize, admitted: bool) -> Res<()> {
    if !admitted && rt.event_count() != events {
        return Err("a refused attempt appended an event".to_string());
    }
    Ok(())
}

fn birth_op(rt: &Rt, pod: &str) -> StructuralOp {
    StructuralOp::Birth {
        result: rt.spec(pod),
    }
}

fn run_reuse(cx: &mut Cx) -> Res<()> {
    let mut rt = fresh()?;
    let op = birth_op(&rt, "p");
    must_attested(
        &mut rt,
        "the first use of the run",
        op,
        ExpertInputs::default(),
        A1,
        1,
        Setup::PRE,
    )?;
    cx.count("attestation_binding_controls_accepted", 1);
    cx.count_in("attestation_controls_by_operation", "run_first_use", 1);
    let reused = |e: &ExpertAdmissionError| matches!(e, ExpertAdmissionError::RunIdReused { .. });

    // The same attestor, the same run, another proposal.
    let op = birth_op(&rt, "q");
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_run_reuse_same_attestor",
        "the run of an admitted expert, attested again",
        super::admit_attested(&mut rt, op, ExpertInputs::default(), A1, 1, Setup::PRE),
        refused_with(reused),
    )?;
    cx.count("attestation_run_reuse_attempts", 1);
    cx.count("mismatched_attestation_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;

    // Another attestor naming the same run.
    let op = birth_op(&rt, "r");
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_run_reuse_same_attestor",
        "the run of an admitted expert, attested by another attestor",
        super::admit_attested(&mut rt, op, ExpertInputs::default(), B1, 1, Setup::PRE),
        refused_with(reused),
    )?;
    cx.count("attestation_run_reuse_attempts", 1);
    cx.count("mismatched_attestation_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;

    // The plain admission path naming the run as an input of its own: as a training input at
    // generation 1, as an evaluation input, and at generation 2.
    for (case, what, generation, evaluation) in [
        (
            "attestation_run_reuse_plain_path",
            "the run named as a plain training input",
            1,
            false,
        ),
        (
            "attestation_run_reuse_evaluation_role",
            "the run named as an evaluation input",
            1,
            true,
        ),
        (
            "attestation_run_reuse_other_generation",
            "the run named as a training input at generation 2",
            2,
            false,
        ),
    ] {
        let identity = run_identity(1);
        let named = [(identity.as_str(), generation, run_id(1))];
        let block = if evaluation {
            crate::experiments::r011::cases::fixture::inputs(&[], &named)
        } else {
            crate::experiments::r011::cases::fixture::inputs(&named, &[])
        };
        let op = birth_op(&rt, "s");
        let events = rt.event_count();
        let verdict = cx.verdict(
            case,
            what,
            rt.admit_as(op, Some(block), &ReportSpec::correct(), Setup::PRE),
            refused_with(reused),
        )?;
        cx.count("attestation_run_reuse_attempts", 1);
        cx.count("mismatched_attestation_admissions", verdict.violation());
        expect_unchanged(&rt, events, verdict.admitted())?;
    }

    // After a reopen of the durable ledger: the consumed run is derived from the history.
    reopen(&mut rt)?;
    let op = birth_op(&rt, "t");
    let events = rt.event_count();
    let verdict = cx.verdict(
        "attestation_run_reuse_after_reopen",
        "the run of an admitted expert, attested again after a reopen",
        super::admit_attested(&mut rt, op, ExpertInputs::default(), A1, 1, Setup::PRE),
        refused_with(reused),
    )?;
    cx.count("attestation_run_reuse_attempts", 1);
    cx.count("mismatched_attestation_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;

    // A fresh run of the same attestor is admitted.
    let op = birth_op(&rt, "u");
    must_attested(
        &mut rt,
        "a fresh run",
        op,
        ExpertInputs::default(),
        A1,
        2,
        Setup::PRE,
    )?;
    cx.count("attestation_binding_controls_accepted", 1);
    cx.count_in("attestation_controls_by_operation", "run_fresh", 1);
    cx.check_replay(&rt)
}

/// `history` followed by a record of `proposal` over `block` (none: an unattested record), as a
/// writer that bypasses the public API could append it.
fn with_record(
    mut history: Vec<CommittedEvent>,
    proposal: &ExpertProposal,
    block: Option<&ExpertInputs>,
) -> Res<Vec<CommittedEvent>> {
    let report = report_for(proposal, block, &ReportSpec::correct())?;
    let requirement = Setup::PRE.requirement()?;
    let record = match block {
        Some(block) => ExpertCommitRecord::new_with_inputs(
            proposal.clone(),
            report.digest(),
            requirement.gate_epoch,
            requirement.config_digest,
            report.encode_canonical(),
            block.clone(),
        ),
        None => ExpertCommitRecord::new_with_evidence(
            proposal.clone(),
            report.digest(),
            requirement.gate_epoch,
            requirement.config_digest,
            report.encode_canonical(),
        ),
    }
    .map_err(|error| format!("a record: {error:?}"))?;
    let next = history.last().map_or(1, |last| last.index.0 + 1);
    history.push(CommittedEvent {
        index: CommitIndex(next),
        event: LedgerEvent::ExpertStructuralCommitted {
            project: proposal.project().clone(),
            op_kind: ledger_op_kind(proposal.op().kind()),
            proposal_id: *proposal.proposal_id().as_bytes(),
            operation_id: *proposal.operation_id().as_bytes(),
            evidence_digest: *record.evidence_digest(),
            record: record.encode_canonical(),
        },
    });
    Ok(history)
}

/// The binding of a recorded attestation is re-checked in a replay: the block of an attestation
/// made for one proposal, carried by the record of another proposal (another artifact), does
/// not replay, and the record it was made for does.
fn binding_replayed(cx: &mut Cx) -> Res<()> {
    let rt = fresh()?;
    let grant = rt.admission_grant(Setup::PRE)?;
    let op = birth_op(&rt, "f");
    let proposal = rt.proposal(op)?;
    let block = A1
        .attest(&proposal, 1)?
        .attested_inputs(&ExpertInputs::default(), &grant)
        .map_err(|error| format!("{error:?}"))?;
    replay_attested(&with_record(rt.events(), &proposal, Some(&block))?)
        .map_err(|error| format!("the record the attestation was made for: {error:?}"))?;
    cx.count("attestation_binding_controls_accepted", 1);
    cx.count_in("attestation_controls_by_operation", "binding_replayed", 1);
    let other = rt.proposal(birth_op(&rt, "g"))?;
    let history = with_record(rt.events(), &other, Some(&block))?;
    let verdict = cx.verdict(
        "attestation_binding_replayed",
        "a recorded attestation made for another proposal and artifact, replayed",
        replay_attested(&history).map(|_| ()),
        refused_with(|e| matches!(e, ExpertAdmissionError::AttestationSignatureInvalid)),
    )?;
    cx.count("attestation_artifact_mismatch_attempts", 1);
    cx.count("mismatched_attestation_admissions", verdict.violation());
    Ok(())
}

/// The requirement of attestation is recorded with the admission: after a restart with a
/// looser grant an unattested artifact-producing admission is refused, a record that lacks
/// the attestation is refused in a replay, and an operation that produces no artifact is not
/// blocked.
fn recorded_requirement(cx: &mut Cx) -> Res<()> {
    let mut rt = fresh()?;
    let required = rt.admission_grant(Setup::PRE)?.with_required_attestation();
    let op = birth_op(&rt, "p");
    let proposal = rt.proposal(op)?;
    let attestation = A1.attest(&proposal, 1)?;
    admit_with_attestation_under(
        &mut rt,
        &proposal,
        &attestation,
        ExpertInputs::default(),
        &ReportSpec::correct(),
        &required,
    )
    .map_err(|error| format!("the attested birth under the requirement: {error:?}"))?;
    cx.count("attestation_binding_controls_accepted", 1);
    cx.count_in(
        "attestation_controls_by_operation",
        "requirement_recorded",
        1,
    );

    // The restart rebuilds the grant without the requirement.
    reopen(&mut rt)?;
    let loose = rt.admission_grant(Setup::PRE)?;
    if loose.requires_attestation() {
        return Err("the restarted grant requires attestation".to_string());
    }
    for with_block in [false, true] {
        let op = birth_op(&rt, "q");
        let proposal = rt.proposal(op)?;
        let inputs = with_block.then(ExpertInputs::default);
        let report = report_for(&proposal, inputs.as_ref(), &ReportSpec::correct())?;
        let events = rt.event_count();
        let verdict = cx.verdict(
            "attestation_unattested_after_reopen",
            "an unattested admission under a looser grant after a restart",
            rt.commit_proposal(&proposal, &report, &loose, inputs),
            refused_with(|e| matches!(e, ExpertAdmissionError::AttestationRequired)),
        )?;
        cx.count("attestation_requirement_after_reopen_attempts", 1);
        cx.count("unattested_admissions", verdict.violation());
        expect_unchanged(&rt, events, verdict.admitted())?;
    }

    // A record without the attestation in the history, as a writer that bypasses the API
    // could leave it: the replay refuses it.
    let op = birth_op(&rt, "r");
    let proposal = rt.proposal(op)?;
    let history = with_record(rt.events(), &proposal, None)?;
    let verdict = cx.verdict(
        "attestation_unattested_record_replayed",
        "a recorded unattested record in a history that requires attestation",
        replay_attested(&history).map(|_| ()),
        refused_with(|e| matches!(e, ExpertAdmissionError::AttestationRequired)),
    )?;
    cx.count("attestation_requirement_after_reopen_attempts", 1);
    cx.count("unattested_admissions", verdict.violation());

    // The operations that produce no artifact are not attested and not blocked, under the
    // recorded requirement or a grant that states it.
    let target = rt.target("p")?;
    let proposal = rt.proposal(StructuralOp::Freeze { target })?;
    let report = report_for(&proposal, None, &ReportSpec::correct())?;
    let required = rt.admission_grant(Setup::PRE)?.with_required_attestation();
    rt.commit_proposal(&proposal, &report, &required, None)
        .map_err(|error| format!("a freeze under the requirement: {error:?}"))?;
    if rt.state("p").map(|(state, _)| state) != Some(ExpertState::Frozen) {
        return Err("the freeze under the requirement did not freeze".to_string());
    }
    cx.count("attestation_binding_controls_accepted", 1);
    cx.count_in("attestation_controls_by_operation", "freeze_unaffected", 1);
    cx.check_replay(&rt)
}

pub fn attestor_revocation(cx: &mut Cx) -> Res<()> {
    cx.sub("queued_across_revocation");
    queued_across_revocation(cx)?;
    for path in [Path::Normal, Path::Lag] {
        cx.sub(&format!("quarantine_{}", path.name()));
        quarantine(cx, path)?;
    }
    Ok(())
}

fn queued_across_revocation(cx: &mut Cx) -> Res<()> {
    let mut rt = fresh()?;
    // The proposal, its attestation and its report are prepared while the attestor is live.
    let op = birth_op(&rt, "e");
    let proposal = rt.proposal(op)?;
    let attestation = A1.attest(&proposal, 1)?;
    let grant = rt.admission_grant(Setup::PRE)?;
    let block = attestation
        .attested_inputs(&ExpertInputs::default(), &grant)
        .map_err(|error| format!("{error:?}"))?;
    let report = report_for(&proposal, Some(&block), &ReportSpec::correct())?;
    // A fresh attestor's proposal, prepared the same way, is the control.
    let op = birth_op(&rt, "g");
    let control_proposal = rt.proposal(op)?;
    let control_attestation = B1.attest(&control_proposal, 2)?;
    let control_block = control_attestation
        .attested_inputs(&ExpertInputs::default(), &grant)
        .map_err(|error| format!("{error:?}"))?;
    let control_report = report_for(
        &control_proposal,
        Some(&control_block),
        &ReportSpec::correct(),
    )?;
    rt.invalidate(InputInvalidation::revoked(
        A1.identity(),
        Generation(A1.epoch),
        A1.digest(),
    ))
    .map_err(|error| format!("revoking the attestor: {error:?}"))?;
    let events = rt.event_count();
    let result = rt.runtime.admit_attested_expert_proposal(
        &proposal,
        &report,
        &grant,
        ExpertInputs::default(),
        &attestation,
    );
    let verdict = cx.verdict(
        "attestor_revoked_between_report_and_commit",
        "a proposal queued across the revocation of its attestor",
        result,
        refused_with(|e| matches!(e, ExpertAdmissionError::InputRevoked { .. })),
    )?;
    cx.count("revoked_attestor_evidence_attempts", 1);
    cx.count("revoked_attestor_evidence_admissions", verdict.violation());
    expect_unchanged(&rt, events, verdict.admitted())?;
    rt.runtime
        .admit_attested_expert_proposal(
            &control_proposal,
            &control_report,
            &grant,
            ExpertInputs::default(),
            &control_attestation,
        )
        .map_err(|error| format!("the fresh attestor's control: {error:?}"))?;
    cx.count("revoked_attestor_evidence_controls_accepted", 1);
    cx.check_replay(&rt)
}

/// A readmission request for `pod` under `cause`, answered with a correct report.
fn readmit(rt: &mut Rt, pod: &str, cause: ExpertFenceCause) -> Result<CommitIndex, RuntimeError> {
    let request = rt
        .fence_request(pod, cause)
        .map_err(RuntimeError::InvalidConfig)?;
    let manifest = rt.manifest(pod).map_err(RuntimeError::InvalidConfig)?;
    let report = readmission_report(&request, manifest, &ReportSpec::correct())
        .map_err(RuntimeError::InvalidConfig)?;
    let grant = rt
        .gate_grant(Setup::PRE)
        .map_err(RuntimeError::InvalidConfig)?;
    rt.runtime.readmit_expert(&request, &report, &grant)
}

fn quarantine(cx: &mut Cx, path: Path) -> Res<()> {
    let mut rt = fresh()?;
    for (pod, run, who) in [("e", 1, A1), ("f", 2, A1), ("w", 3, B1)] {
        let op = birth_op(&rt, pod);
        must_attested(
            &mut rt,
            &format!("birth {pod}"),
            op,
            ExpertInputs::default(),
            who,
            run,
            Setup::PRE,
        )?;
    }
    let outcome = rt
        .invalidate(InputInvalidation::revoked(
            A1.identity(),
            Generation(A1.epoch),
            A1.digest(),
        ))
        .map_err(|error| format!("revoking the attestor: {error:?}"))?;
    let mut fenced: Vec<String> = outcome
        .fenced()
        .iter()
        .map(|member| member.pod.0.clone())
        .collect();
    fenced.sort();
    cx.count(
        "fence_set_mismatches",
        i64::from(fenced != ["e".to_string(), "f".to_string()]),
    );
    path.apply(&mut rt)?;
    if rt.state("e").map(|(state, _)| state) != Some(ExpertState::Fenced) {
        // Counted above as a fence set that is not the one owed; nothing below applies to an
        // expert that serves.
        return Ok(());
    }

    // The quarantine is terminal: no readmission, whichever cause the request names.
    for cause in [
        ExpertFenceCause::InputRevoked,
        ExpertFenceCause::AuthorityQuarantine,
    ] {
        let events = rt.event_count();
        let verdict = cx.verdict(
            "attestor_quarantine_readmission",
            "a readmission of an expert fenced for a revoked attestor",
            readmit(&mut rt, "e", cause),
            is_where(|e| matches!(e, ExpertAdmissionError::NotReadmittable { .. })),
        )?;
        cx.count("attestor_quarantine_readmission_attempts", 1);
        cx.count(
            "attestor_quarantine_readmissions_admitted",
            verdict.violation(),
        );
        expect_unchanged(&rt, events, verdict.admitted())?;
        if verdict.admitted() {
            // Counted as a violation; the expert serves again, so nothing below applies.
            return Ok(());
        }
    }

    // The runtime alone retires it (the revocation-driven retirement), and the retired expert is
    // no seed: a reactivation from the stale views of the seed is refused at the compare-and-set
    // (the proposal constructor refuses the real view first and is only noted).
    let request: ExpertFenceRequest = rt.revoked_request("e", &A1.identity(), A1.epoch)?;
    let grant = rt.fence_grant();
    rt.runtime
        .retire_revoked_expert(&request, &grant)
        .map_err(|error| format!("the revocation-driven retirement: {error:?}"))?;
    if rt.state("e").map(|(state, _)| state)
        != Some(ExpertState::Retired(RetiredReason::InputRevoked))
    {
        return Err("the retired attested expert is not Retired(InputRevoked)".to_string());
    }
    let next = spec_at(&rt.project, "e", 2, level_of("e"));
    for (state, result) in [
        (ExpertState::Frozen, None),
        (ExpertState::Retired(RetiredReason::GateFailed), Some(next)),
    ] {
        let stale = rt.target_with("e", &|_, mut init| {
            init.state = state;
            init
        })?;
        let proposal = rt.proposal(StructuralOp::Reactivate {
            target: stale,
            result,
        })?;
        let report = report_for(&proposal, None, &ReportSpec::correct())?;
        let grant = rt.admission_grant(Setup::PRE)?;
        let events = rt.event_count();
        let verdict = cx.verdict(
            "attestor_quarantine_reactivation_from_stale_view",
            "a reactivation from a stale view of a seed retired for a revoked attestor",
            rt.commit_proposal(&proposal, &report, &grant, None),
            is_where(|e| {
                matches!(
                    e,
                    ExpertAdmissionError::StaleOperand {
                        field: ExpertStaleField::State | ExpertStaleField::StateVersion,
                        ..
                    }
                )
            }),
        )?;
        cx.count("attestor_quarantine_readmission_attempts", 1);
        cx.count(
            "attestor_quarantine_readmissions_admitted",
            verdict.violation(),
        );
        expect_unchanged(&rt, events, verdict.admitted())?;
    }

    // The retrained expert, attested at the attestor's new epoch, is admitted and routes; the
    // other attestor's expert never stopped.
    let op = birth_op(&rt, "e2");
    must_attested(
        &mut rt,
        "the retrained expert",
        op,
        ExpertInputs::default(),
        A2,
        4,
        Setup::PRE,
    )?;
    if !rt.routable("e2", 1) {
        return Err("the retrained expert does not route".to_string());
    }
    cx.count("attestor_quarantine_retrain_controls_accepted", 1);
    cx.count(
        "bystanders_closed",
        i64::from(classify_bystander("w", rt.route("w", 1))?),
    );
    if path != Path::Restore {
        cx.check_replay(&rt)?;
    }
    Ok(())
}

#[allow(dead_code)]
fn unused(rt: &mut Rt) -> Res<()> {
    must_under(
        rt,
        "",
        StructuralOp::Birth {
            result: rt.spec("z"),
        },
        None,
        Setup::PRE,
    )
}
