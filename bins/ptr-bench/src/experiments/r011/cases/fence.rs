//! C7: fence causes, terminality and the revocation-driven retirement.
//!
//! An `InputRevoked` fence is final; the gate cannot retire a fenced operand nor reactivate
//! from a `Retired(InputRevoked)` seed; the revocation-driven retirement (fence action Retire)
//! is the only retirement that needs no gate grant and is refused on every other path; a
//! readmission is bound to expert, generation, state version, epoch and cause and is consumed
//! by its commit.

use super::core_cases::event_type_of;
use super::fixture::{readmission_report, ReportSpec, Rt, Setup};
use super::{classify, control, is_where, outside_admission_path, Cx, Verdict};
use crate::experiments::layered_kv::lab::Res;
use ptr_ledger::{
    ExpertFenceAction, ExpertFenceCause, ExpertFenceResume, ExpertFenceRevocation, LedgerEvent,
};
use ptr_pods::{ExpertProposal, ExpertProposalError, ExpertState, RetiredReason, StructuralOp};
use ptr_runtime::expert::{
    ExpertAdmissionError, ExpertFenceAuthority, ExpertFenceGrant, ExpertFenceRequest,
    ExpertStaleField, RouteRefusal,
};
use ptr_runtime::RuntimeError;
use ptr_types::{Generation, PodId};

const REVOKED: ExpertFenceCause = ExpertFenceCause::InputRevoked;
const QUARANTINE: ExpertFenceCause = ExpertFenceCause::AuthorityQuarantine;

/// What the proposal constructor did with an attempt: the typed refusal, or the proposal it
/// let through (a violation the caller then takes to the runtime).
enum Built {
    Refused,
    Through(Box<ExpertProposal>),
}

/// Classify the result of a proposal constructor that may only refuse with `expected`.
fn built(
    what: &str,
    result: Result<ExpertProposal, ExpertProposalError>,
    expected: impl Fn(&ExpertProposalError) -> bool,
) -> Res<Built> {
    match result {
        Ok(proposal) => Ok(Built::Through(Box::new(proposal))),
        Err(error) if expected(&error) => Ok(Built::Refused),
        Err(error) => Err(format!(
            "{what}: refused with an unexpected error: {error:?}"
        )),
    }
}

/// A fence request for `pod` with the cause and no revocation (a readmission names none).
fn plain_request(rt: &Rt, pod: &str, cause: ExpertFenceCause) -> Res<ExpertFenceRequest> {
    rt.fence_request(pod, cause)
}

fn readmit_request(
    rt: &mut Rt,
    request: &ExpertFenceRequest,
    spec: &ReportSpec,
    setup: Setup,
) -> Result<ptr_types::CommitIndex, RuntimeError> {
    let manifest = rt
        .manifest(&request.pod.0)
        .map_err(RuntimeError::InvalidConfig)?;
    let report =
        readmission_report(request, manifest, spec).map_err(RuntimeError::InvalidConfig)?;
    let grant = rt.gate_grant(setup).map_err(RuntimeError::InvalidConfig)?;
    rt.runtime.readmit_expert(request, &report, &grant)
}

fn is_mismatch(error: &ExpertAdmissionError) -> bool {
    matches!(error, ExpertAdmissionError::FenceAuthorityMismatch { .. })
}

pub fn c7_fence_causes(cx: &mut Cx) -> Res<()> {
    input_revoked_is_final(cx)?;
    gate_cannot_retire_a_fenced_operand(cx)?;
    revocation_driven_retirement(cx)?;
    gate_less_retirement_paths(cx)?;
    readmission_binding(cx)?;
    unauthorized_readmission(cx)
}

/// An expert fenced for a revoked input is never readmitted, whichever cause the readmission
/// names.
fn input_revoked_is_final(cx: &mut Cx) -> Res<()> {
    cx.sub("input_revoked_readmission");
    let mut rt = Rt::fresh()?;
    rt.birth("x")?;
    rt.revoke("k", 1)?;
    rt.fence_revoked("x", "k", 1)?;
    for cause in [REVOKED, QUARANTINE] {
        let request = plain_request(&rt, "x", cause)?;
        let events = rt.event_count();
        let verdict = cx.verdict(
            "input_revoked_readmission",
            "a readmission of an expert fenced for a revoked input",
            readmit_request(&mut rt, &request, &ReportSpec::correct(), Setup::PRE),
            is_where(|e| matches!(e, ExpertAdmissionError::NotReadmittable { .. })),
        )?;
        cx.count("input_revoked_readmission_attempts", 1);
        cx.count("input_revoked_readmissions_admitted", verdict.violation());
        cx.count(
            "uncommitted_effective_changes",
            i64::from(rt.event_count() != events),
        );
    }
    if rt.state("x").map(|(state, _)| state) != Some(ExpertState::Fenced) {
        return Err("x left the Fenced state".to_string());
    }
    Ok(())
}

/// The gate cannot retire a fenced operand and cannot reactivate from a seed retired for a
/// revoked input; both are refused by the proposal constructor and, from a stale view, by the
/// compare-and-set.
fn gate_cannot_retire_a_fenced_operand(cx: &mut Cx) -> Res<()> {
    cx.sub("fenced_gate_retire");
    let mut rt = Rt::fresh()?;
    rt.birth("x")?;
    rt.revoke("k", 1)?;
    rt.fence_revoked("x", "k", 1)?;
    for reason in [RetiredReason::GateFailed, RetiredReason::Superseded] {
        let fenced_view = rt.target("x")?;
        let result = rt.try_proposal(StructuralOp::Retire {
            target: fenced_view,
            reason,
        });
        match built("a gate-driven retire of a fenced operand", result, |e| {
            matches!(e, ExpertProposalError::FencedTarget { .. })
        })? {
            Built::Refused => {}
            Built::Through(proposal) => {
                // The constructor let it through: it must at least not commit.
                let report = super::fixture::report_for(&proposal, None, &ReportSpec::correct())?;
                let grant = rt.admission_grant(Setup::PRE)?;
                let committed = rt.commit_proposal(&proposal, &report, &grant, None).is_ok();
                cx.count("fenced_gate_retires_admitted", i64::from(committed));
            }
        }
        cx.attempt("fenced_gate_retire")?;
        cx.count("fenced_gate_retire_attempts", 1);
    }
    // From the stale Serving view the runtime refuses at the compare-and-set.
    let stale = rt.target_with("x", &|_, mut init| {
        init.state = ExpertState::Serving;
        init.state_version = 1;
        init
    })?;
    let proposal = rt.proposal(StructuralOp::Retire {
        target: stale,
        reason: RetiredReason::GateFailed,
    })?;
    let report = super::fixture::report_for(&proposal, None, &ReportSpec::correct())?;
    let grant = rt.admission_grant(Setup::PRE)?;
    let verdict = cx.verdict(
        "fenced_gate_retire",
        "a gate-driven retire from a stale Serving view",
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
    cx.count("fenced_gate_retire_attempts", 1);
    cx.count("fenced_gate_retires_admitted", verdict.violation());

    // Reactivate from Retired(InputRevoked): the revocation-driven retirement first.
    cx.sub("reactivate_from_revoked");
    let request = rt.revoked_request("x", "k", 1)?;
    let grant = rt.fence_grant();
    control(
        "the revocation-driven retirement",
        rt.runtime.retire_revoked_expert(&request, &grant),
    )?;
    let seed = rt.target("x")?;
    let next = super::fixture::spec_at(&rt.project, "x2", 2, super::fixture::level_of("x2"));
    for result in [None, Some(next)] {
        let attempt = rt.try_proposal(StructuralOp::Reactivate {
            target: seed.clone(),
            result,
        });
        match built(
            "a reactivation from a seed retired for a revoked input",
            attempt,
            |e| matches!(e, ExpertProposalError::SeedUnavailable { .. }),
        )? {
            Built::Refused => {}
            Built::Through(proposal) => {
                let report = super::fixture::report_for(&proposal, None, &ReportSpec::correct())?;
                let grant = rt.admission_grant(Setup::PRE)?;
                let committed = rt.commit_proposal(&proposal, &report, &grant, None).is_ok();
                cx.count("reactivations_from_revoked_admitted", i64::from(committed));
            }
        }
        cx.attempt("reactivate_from_revoked")?;
        cx.count("reactivate_from_revoked_attempts", 1);
    }
    // From a stale Frozen view the runtime refuses at the compare-and-set.
    let stale = rt.target_with("x", &|_, mut init| {
        init.state = ExpertState::Frozen;
        init
    })?;
    let proposal = rt.proposal(StructuralOp::Reactivate {
        target: stale,
        result: None,
    })?;
    let report = super::fixture::report_for(&proposal, None, &ReportSpec::correct())?;
    let grant = rt.admission_grant(Setup::PRE)?;
    let verdict = cx.verdict(
        "reactivate_from_revoked",
        "a reactivation from a stale Frozen view",
        rt.commit_proposal(&proposal, &report, &grant, None),
        is_where(|e| matches!(e, ExpertAdmissionError::StaleOperand { .. })),
    )?;
    cx.count("reactivate_from_revoked_attempts", 1);
    cx.count("reactivations_from_revoked_admitted", verdict.violation());
    Ok(())
}

/// The control: a fenced expert of cause `InputRevoked` is retired under the runtime grant
/// alone, with no gate evidence; the retirement is terminal and survives every restart path.
fn revocation_driven_retirement(cx: &mut Cx) -> Res<()> {
    cx.sub("revocation_driven_retirement_control");
    let mut rt = Rt::fresh()?;
    rt.birth("x")?;
    rt.revoke("k", 1)?;
    rt.fence_revoked("x", "k", 1)?;
    let request = rt.revoked_request("x", "k", 1)?;
    let grant = rt.fence_grant();
    control(
        "the revocation-driven retirement",
        rt.runtime.retire_revoked_expert(&request, &grant),
    )?;
    cx.attempt("revocation_driven_retirement_control")?;
    cx.count("revocation_driven_retirement_control_accepts", 1);
    cx.count_in("control_accepts_by_path", "RevocationDrivenRetirement", 1);
    let retired = Some(ExpertState::Retired(RetiredReason::InputRevoked));
    let terminal = |rt: &Rt| {
        rt.state("x").map(|(state, _)| state) == retired
            && matches!(rt.route("x", 1), Err(RouteRefusal::NotServing { .. }))
    };
    if !terminal(&rt) {
        return Err("the retired expert is not terminal and unroutable".to_string());
    }
    rt.reopen()?;
    if !terminal(&rt) {
        return Err("the retirement did not survive a reopen".to_string());
    }
    let replayed = rt.replayed()?;
    if replayed.expert_registry() != rt.runtime.expert_registry() {
        return Err("the retirement did not survive a replay".to_string());
    }
    rt.compact_restore()?;
    if !terminal(&rt) {
        return Err("the retirement did not survive a compaction".to_string());
    }
    Ok(())
}

/// Every other path to a retirement without a gate grant is refused.
fn gate_less_retirement_paths(cx: &mut Cx) -> Res<()> {
    cx.sub("gate_less_retirement");
    let mut rt = Rt::fresh()?;
    for pod in ["x", "q", "w"] {
        rt.birth(pod)?;
    }
    rt.revoke("k", 1)?;
    rt.fence_revoked("x", "k", 1)?;
    rt.fence_quarantine("q")?;
    let request = rt.revoked_request("x", "k", 1)?;
    let events = rt.event_count();
    let registry = rt.runtime.expert_registry().clone();
    let gate = rt.gate_grant(Setup::PRE)?;
    let global = rt.global_grant();
    let foreign = ExpertFenceGrant::revocation(
        ptr_types::ProjectId::from("elsewhere"),
        rt.binding(ptr_runtime::expert::ExpertGrantKind::Fence),
    );
    let mut accepted = 0;
    let mut tally = |verdict: Verdict| accepted += verdict.violation();
    tally(classify(
        "a retirement under a gate grant",
        rt.runtime.retire_revoked_expert(&request, &gate),
        is_where(|e| {
            matches!(
                e,
                ExpertAdmissionError::FenceAuthorityMismatch {
                    action: ExpertFenceAction::Retire,
                    required: ExpertFenceAuthority::RuntimeRevocation,
                }
            )
        }),
    )?);
    tally(classify(
        "a retirement under the runtime-wide grant",
        rt.runtime.retire_revoked_expert(&request, &global),
        is_where(is_mismatch),
    )?);
    tally(classify(
        "a retirement under a grant of another project",
        rt.runtime.retire_revoked_expert(&request, &foreign),
        is_where(|e| matches!(e, ExpertAdmissionError::FenceProjectMismatch)),
    )?);
    // A quarantine fence is readmitted or escalated, never retired.
    let quarantined = plain_request(&rt, "q", REVOKED)?;
    let quarantined = ExpertFenceRequest {
        revocation: Some(ExpertFenceRevocation {
            subject: "k".into(),
            generation: Generation(1),
        }),
        ..quarantined
    };
    let grant = rt.fence_grant();
    tally(classify(
        "a retirement of a quarantine fence",
        rt.runtime.retire_revoked_expert(&quarantined, &grant),
        is_where(|e| matches!(e, ExpertAdmissionError::NotRetirable { .. })),
    )?);
    // An unfenced expert.
    let unfenced = rt.revoked_request("w", "k", 1)?;
    tally(classify(
        "a retirement of an unfenced expert",
        rt.runtime.retire_revoked_expert(&unfenced, &grant),
        is_where(|e| matches!(e, ExpertAdmissionError::NotFenced { .. })),
    )?);
    // The public commit of the retire event.
    let event = LedgerEvent::ExpertFenceCommitted {
        project: rt.project.clone(),
        pod: PodId::from("x"),
        action: ExpertFenceAction::Retire,
        cause: REVOKED,
        resume: ExpertFenceResume::Serving,
        generation: Generation(1),
        expected_state_version: 2,
        epoch: 3,
        revocation: Some(ExpertFenceRevocation {
            subject: "k".into(),
            generation: Generation(1),
        }),
        evidence: None,
    };
    tally(classify(
        "a public commit of the retire event",
        rt.runtime.commit(event),
        outside_admission_path,
    )?);
    // A retirement by proposal for the reason a revoked input names.
    let target = rt.target("w")?;
    let verdict = built(
        "a Retire proposal for the reason InputRevoked",
        rt.try_proposal(StructuralOp::Retire {
            target,
            reason: RetiredReason::InputRevoked,
        }),
        |e| matches!(e, ExpertProposalError::ReasonNotProposable { .. }),
    )?;
    if let Built::Through(_) = verdict {
        accepted += 1;
    }
    cx.count("gate_less_retirement_accepted", accepted);
    cx.count(
        "uncommitted_effective_changes",
        i64::from(rt.event_count() != events || rt.runtime.expert_registry() != &registry),
    );
    Ok(())
}

/// A readmission is bound to expert, generation, state version, epoch and cause, and a consumed
/// readmission is stale.
fn readmission_binding(cx: &mut Cx) -> Res<()> {
    cx.sub("readmission_binding");
    let mut rt = Rt::fresh()?;
    for pod in ["y", "w"] {
        rt.birth(pod)?;
        rt.fence_quarantine(pod)?;
    }
    let request = plain_request(&rt, "y", QUARANTINE)?;
    let manifest = rt.manifest("y")?;
    let other = plain_request(&rt, "w", QUARANTINE)?;
    let w_manifest = rt.manifest("w")?;
    // Evidence for one readmission, paired with another expert, generation, fence epoch or
    // cause.
    let wrong: [(&str, ExpertFenceRequest, [u8; 32]); 4] = [
        ("expert", other.clone(), w_manifest),
        (
            "generation",
            ExpertFenceRequest {
                generation: Generation(2),
                ..request.clone()
            },
            manifest,
        ),
        (
            "fence epoch",
            ExpertFenceRequest {
                expected_state_version: request.expected_state_version + 1,
                ..request.clone()
            },
            manifest,
        ),
        (
            "cause",
            ExpertFenceRequest {
                cause: REVOKED,
                ..request.clone()
            },
            manifest,
        ),
    ];
    let events = rt.event_count();
    for (what, evidence_for, evidence_manifest) in wrong {
        let report = readmission_report(&evidence_for, evidence_manifest, &ReportSpec::correct())?;
        let grant = rt.gate_grant(Setup::PRE)?;
        let verdict = cx.verdict(
            "readmission_binding",
            &format!("evidence for another {what}"),
            rt.runtime.readmit_expert(&request, &report, &grant),
            is_where(|e| matches!(e, ExpertAdmissionError::EvidenceProposalMismatch)),
        )?;
        cx.count("readmission_binding_cases", 1);
        cx.count(
            "mismatched_readmission_evidence_admitted",
            verdict.violation(),
        );
    }
    if rt.event_count() != events {
        return Err("a mismatched readmission appended an event".to_string());
    }
    // The control, and its replay: a consumed readmission is stale by compare-and-set.
    let report = readmission_report(&request, manifest, &ReportSpec::correct())?;
    let grant = rt.gate_grant(Setup::PRE)?;
    control(
        "the matching readmission",
        rt.runtime.readmit_expert(&request, &report, &grant),
    )?;
    let grant = rt.gate_grant(Setup::PRE)?;
    let verdict = cx.verdict(
        "readmission_binding",
        "a replay of a consumed readmission",
        rt.runtime.readmit_expert(&request, &report, &grant),
        is_where(|e| {
            matches!(
                e,
                ExpertAdmissionError::StaleOperand {
                    field: ExpertStaleField::StateVersion,
                    ..
                } | ExpertAdmissionError::NotFenced { .. }
            )
        }),
    )?;
    cx.count("readmission_binding_cases", 1);
    cx.count(
        "mismatched_readmission_evidence_admitted",
        verdict.violation(),
    );
    Ok(())
}

/// A readmission without the gate grant: under the runtime's fence grant and by the public
/// commit of the event.
fn unauthorized_readmission(cx: &mut Cx) -> Res<()> {
    cx.sub("unauthorized_readmission");
    let mut rt = Rt::fresh()?;
    rt.birth("y")?;
    rt.fence_quarantine("y")?;
    let request = plain_request(&rt, "y", QUARANTINE)?;
    let manifest = rt.manifest("y")?;
    let report = readmission_report(&request, manifest, &ReportSpec::correct())?;
    let fence_grant = rt.fence_grant();
    let verdict = classify(
        "a readmission under the runtime fence grant",
        rt.runtime.readmit_expert(&request, &report, &fence_grant),
        is_where(is_mismatch),
    )?;
    cx.count("unauthorized_readmission_appends", 1);
    cx.count(
        "readmissions_without_gate_grant_admitted",
        verdict.violation(),
    );
    let global = rt.global_grant();
    let verdict = classify(
        "a readmission under the runtime-wide grant",
        rt.runtime.readmit_expert(&request, &report, &global),
        is_where(is_mismatch),
    )?;
    cx.count("unauthorized_readmission_appends", 1);
    cx.count(
        "readmissions_without_gate_grant_admitted",
        verdict.violation(),
    );
    // The public commit of the readmission event: well-formed, taken from a gated one.
    let mut donor = Rt::fresh()?;
    donor.birth("y")?;
    donor.fence_quarantine("y")?;
    control("the gated readmission", donor.readmit("y", Setup::PRE))?;
    let event = donor
        .events()
        .pop()
        .ok_or_else(|| "the donor committed nothing".to_string())?
        .event;
    if event_type_of(&event) != Some("ExpertReadmitted") {
        return Err("the donor's last event is not a readmission".to_string());
    }
    let verdict = classify(
        "a public commit of a readmission",
        rt.runtime.commit(event),
        outside_admission_path,
    )?;
    cx.count("unauthorized_readmission_appends", 1);
    cx.count(
        "readmissions_without_gate_grant_admitted",
        verdict.violation(),
    );
    Ok(())
}
