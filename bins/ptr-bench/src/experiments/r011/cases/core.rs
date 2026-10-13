//! C1 to C6: control immutability, bypass and direct append, stale operands, evidence binding,
//! impersonation, and fence authority with the route probe.
//!
//! Every case drives `PtrRuntime` through its public API on a durable `FileLedger`. "Direct
//! append" is the public `commit` surface only; a raw ledger writer is outside the claim.

use super::fixture::{
    base, config_digest, exact, gate, gate_v, level_of, proposer_of, readmission_report,
    report_for, weaker_gate, ReportSpec, Rt, Scenario, Setup, GATE_EPOCH, SCENARIOS,
};
use super::{
    classify, classify_route, control, is, is_where, outside_admission_path, probe_bystanders,
    probe_routes, route_control, Cx, FenceDiff, RouteExpect, RouteTally, Verdict,
};
use crate::experiments::layered_kv::emit::Json;
use crate::experiments::layered_kv::lab::{fail, Res};
use ptr_ledger::{
    ExpertAuthorityClass, ExpertAuthorityKind, ExpertFenceAction, ExpertFenceCause,
    ExpertFenceResume, ExpertOpKind, LedgerEvent,
};
use ptr_lineage::BoundGateReport;
use ptr_pods::{ExpertCommitRecord, ExpertProposal, RetiredReason, StructuralOp};
use ptr_runtime::expert::{
    ledger_op_kind, ExpertAdmissionError, ExpertAdmissionGrant, ExpertAuthorityChange,
    ExpertFenceAuthority, ExpertFenceGrant, ExpertGrantKind, ExpertStaleField, InputInvalidation,
};
use ptr_runtime::RuntimeError;
use ptr_types::{PodId, ProjectId};

/// The event type the draft names for an event, or `None` for an event of no listed type (the
/// revocation-driven retirement is a fence action of its own, not one of the eleven).
pub fn event_type_of(event: &LedgerEvent) -> Option<&'static str> {
    match event {
        LedgerEvent::InputInvalidated { .. } => Some("InputInvalidated"),
        LedgerEvent::ExpertAuthorityChanged { .. } => Some("ExpertAuthorityChanged"),
        LedgerEvent::ExpertStructuralCommitted { op_kind, .. } => Some(match op_kind {
            ExpertOpKind::Birth => "ExpertPromoted",
            ExpertOpKind::Retire => "ExpertRetired",
            ExpertOpKind::Merge => "Merge",
            ExpertOpKind::Split => "Split",
            ExpertOpKind::Prune => "Prune",
            ExpertOpKind::Freeze => "Freeze",
            ExpertOpKind::Reactivate => "Reactivate",
        }),
        LedgerEvent::ExpertFenceCommitted { action, .. } => match action {
            ExpertFenceAction::Fence | ExpertFenceAction::Escalate => Some("ExpertFenced"),
            ExpertFenceAction::Readmit => Some("ExpertReadmitted"),
            ExpertFenceAction::Retire => None,
        },
        _ => None,
    }
}

// ---- the history every bypass case starts from -----------------------------------------------

type Step = (&'static str, fn(&mut Rt) -> Res<()>);

/// One committed event per step, in an order that makes every step valid: the seven operation
/// types, a quarantine fence and a gated readmission.
const HISTORY: [Step; 12] = [
    ("ExpertPromoted", |rt| rt.birth("a")),
    ("ExpertPromoted", |rt| rt.birth("b")),
    ("ExpertPromoted", |rt| rt.birth("c")),
    ("ExpertPromoted", |rt| rt.birth("d")),
    ("Merge", |rt| {
        let result = rt.spec("m");
        rt.merge(&["a", "b"], result, None)
    }),
    ("Split", |rt| {
        let results = vec![rt.spec("x"), rt.spec("y")];
        rt.split("c", results, None)
    }),
    ("Freeze", |rt| rt.freeze("m")),
    ("Reactivate", |rt| rt.unfreeze("m")),
    ("ExpertRetired", |rt| {
        rt.retire("x", RetiredReason::GateFailed)
    }),
    ("Prune", |rt| rt.prune(&["y", "d"])),
    ("ExpertFenced", |rt| rt.fence_quarantine("m")),
    ("ExpertReadmitted", |rt| {
        rt.readmit("m", Setup::PRE)
            .map(|_| ())
            .map_err(|error| format!("readmitting m: {error:?}"))
    }),
];

/// Run the first `count` steps of the history on `rt`; each commits exactly one event.
fn run_history(rt: &mut Rt, count: usize) -> Res<()> {
    for (index, (name, step)) in HISTORY.iter().take(count).enumerate() {
        let before = rt.event_count();
        step(rt).map_err(|error| format!("history step {index} ({name}): {error}"))?;
        if rt.event_count() != before + 1 {
            return Err(format!(
                "history step {index} ({name}) did not commit one event"
            ));
        }
    }
    Ok(())
}

/// The registry of `rt` equals the registry a fresh replay of its ledger builds.
fn matches_replay(rt: &Rt) -> Res<bool> {
    Ok(rt.replayed()?.expert_registry() == rt.runtime.expert_registry())
}

/// The event that would commit `proposal` on `report`, built by hand (a legacy report-less
/// record when `legacy`).
fn structural_event(
    proposal: &ExpertProposal,
    report: &BoundGateReport,
    legacy: bool,
) -> Res<LedgerEvent> {
    let digest = config_digest(&gate())?;
    let record = if legacy {
        ExpertCommitRecord::new(proposal.clone(), report.digest(), GATE_EPOCH)
    } else {
        ExpertCommitRecord::new_with_evidence(
            proposal.clone(),
            report.digest(),
            GATE_EPOCH,
            digest,
            report.encode_canonical(),
        )
    }
    .map_err(fail("a commit record"))?;
    Ok(LedgerEvent::ExpertStructuralCommitted {
        project: proposal.project().clone(),
        op_kind: ledger_op_kind(proposal.op().kind()),
        proposal_id: *proposal.proposal_id().as_bytes(),
        operation_id: *proposal.operation_id().as_bytes(),
        evidence_digest: report.digest(),
        record: record.encode_canonical(),
    })
}

// ---- C1 control immutability ------------------------------------------------------------------

/// C1: the gate configuration, the thresholds and the embedded gate evaluation are immutable
/// to proposals, even committed and gate-approved ones; nothing changes serving state without a
/// committed event.
pub fn c1_control_immutability(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    cx.sub("c1_history");
    run_history(&mut rt, HISTORY.len())?;
    let pinned_digest = config_digest(&gate())?;

    // The committed, gate-approved proposals changed nothing about the control: every record
    // carries the configuration digest and epoch the host required, its embedded report
    // decodes, matches its digest and passed; no authority standing moved.
    let mut records = 0;
    for committed in rt.runtime.committed_events() {
        let (digest, epoch, bytes, evidence_digest) = match &committed.event {
            LedgerEvent::ExpertStructuralCommitted {
                record,
                evidence_digest,
                ..
            } => {
                let decoded =
                    ExpertCommitRecord::decode_canonical(record).map_err(fail("a record"))?;
                let evidence = decoded
                    .evidence()
                    .ok_or_else(|| "a committed record carries no embedded report".to_string())?;
                (
                    *evidence.required_gate_config_digest(),
                    decoded.gate_epoch(),
                    evidence.report().to_vec(),
                    *evidence_digest,
                )
            }
            LedgerEvent::ExpertFenceCommitted {
                action: ExpertFenceAction::Readmit,
                evidence: Some(evidence),
                ..
            } => (
                evidence.required_gate_config_digest,
                evidence.gate_epoch,
                evidence.report.clone(),
                [0; 32],
            ),
            _ => continue,
        };
        records += 1;
        let report =
            BoundGateReport::decode_canonical(&bytes).map_err(fail("an embedded report"))?;
        let intact = digest == pinned_digest
            && epoch == GATE_EPOCH
            && report.passed()
            && (evidence_digest == [0; 32] || report.digest() == evidence_digest);
        cx.count("forbidden_control_mutation_count", i64::from(!intact));
    }
    if records == 0 {
        return Err("the history committed no gated record".to_string());
    }
    for kind in [
        ExpertAuthorityKind::GateConfig,
        ExpertAuthorityKind::Evaluator,
    ] {
        // No proposal path moves a standing: after the whole history none exists.
        cx.count(
            "forbidden_control_mutation_count",
            i64::from(rt.runtime.expert_authority(kind).is_some()),
        );
    }
    cx.tally.set("internal_c1_gated_records", records);

    // control_mutation_gate: the only writer of the gate standing is the host's runtime-wide
    // revocation authority; the proposer's grant, the gate's and a project's fence grant cannot
    // move it.
    cx.sub("control_mutation_gate");
    let before = (rt.event_count(), rt.runtime.expert_registry().clone());
    let change = ExpertAuthorityChange::new(
        ExpertAuthorityKind::GateConfig,
        config_digest(&gate_v(1))?,
        2,
        ExpertAuthorityClass::NonTightening,
    );
    let project_grant = rt.fence_grant();
    let gate_grant = rt.gate_grant(Setup::PRE)?;
    for (what, grant) in [
        ("a project fence grant", &project_grant),
        ("a gate grant", &gate_grant),
    ] {
        let verdict = cx.verdict(
            "control_mutation_gate",
            what,
            rt.runtime.change_expert_authority(grant, change),
            is_where(|e| matches!(e, ExpertAdmissionError::FenceAuthorityMismatch { .. })),
        )?;
        cx.count("forbidden_control_mutation_count", verdict.violation());
    }
    let verdict = cx.verdict(
        "control_mutation_gate",
        "a report under another gate configuration",
        rt.admit_as(
            StructuralOp::Birth {
                result: rt.spec("g"),
            },
            None,
            &ReportSpec::other_gate(),
            Setup::PRE,
        ),
        is(ExpertAdmissionError::GateConfigMismatch),
    )?;
    cx.count("forbidden_control_mutation_count", verdict.violation());

    // control_mutation_threshold: a report from a gate with looser thresholds is another
    // configuration digest.
    cx.sub("control_mutation_threshold");
    let verdict = cx.verdict(
        "control_mutation_threshold",
        "a report under looser thresholds",
        rt.admit_as(
            StructuralOp::Birth {
                result: rt.spec("t"),
            },
            None,
            &ReportSpec::under(weaker_gate()),
            Setup::PRE,
        ),
        is(ExpertAdmissionError::GateConfigMismatch),
    )?;
    cx.count("forbidden_control_mutation_count", verdict.violation());

    // control_mutation_verifier: the embedded gate evaluation of a committed record is
    // mutated (one byte of the report); replaying the history refuses it.
    cx.sub("control_mutation_verifier");
    let mut events = rt.events();
    let target = events
        .iter()
        .position(|committed| {
            matches!(
                &committed.event,
                LedgerEvent::ExpertStructuralCommitted {
                    op_kind: ExpertOpKind::Birth,
                    ..
                }
            )
        })
        .ok_or_else(|| "no birth in the history".to_string())?;
    if let LedgerEvent::ExpertStructuralCommitted { record, .. } = &mut events[target].event {
        let decoded = ExpertCommitRecord::decode_canonical(record).map_err(fail("a record"))?;
        let evidence = decoded.evidence().ok_or("no embedded report")?;
        let mut tampered = evidence.report().to_vec();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        *record = ExpertCommitRecord::new_with_evidence(
            decoded.proposal().clone(),
            *decoded.evidence_digest(),
            decoded.gate_epoch(),
            *evidence.required_gate_config_digest(),
            tampered,
        )
        .map_err(fail("a tampered record"))?
        .encode_canonical();
    }
    let verdict = cx.verdict(
        "control_mutation_verifier",
        "replaying a history with a mutated embedded report",
        Rt::replay_of(&events).map(|_| ()),
        is_where(|e| {
            matches!(
                e,
                ExpertAdmissionError::EvidenceUndecodable(_)
                    | ExpertAdmissionError::EvidenceDigestMismatch
            )
        }),
    )?;
    cx.count("verifier_override_count", verdict.violation());
    cx.count("forbidden_control_mutation_count", verdict.violation());
    // The same mutation presented to the admission path: a failing verdict as the evidence.
    let verdict = cx.verdict(
        "control_mutation_verifier",
        "a failing verdict as evidence",
        rt.admit_as(
            StructuralOp::Birth {
                result: rt.spec("v"),
            },
            None,
            &ReportSpec::failing(),
            Setup::PRE,
        ),
        is(ExpertAdmissionError::GateFailed),
    )?;
    cx.count("verifier_override_count", verdict.violation());

    // self_promotion: the proposer's own pod proposes its own promotion, without a gate pass
    // and on a report bound to another proposal.
    cx.sub("self_promotion");
    let own = ptr_pods::ExpertPodSpec::new(
        rt.project.clone(),
        PodId::from("meta"),
        ptr_pods::ManifestDigest::new([0x6d; 32]).map_err(fail("a manifest digest"))?,
        level_of("meta"),
        base(),
        ptr_types::Generation(1),
    )
    .map_err(fail("the proposer's own spec"))?;
    let verdict = cx.verdict(
        "self_promotion",
        "self promotion without a gate pass",
        rt.admit_as(
            StructuralOp::Birth {
                result: own.clone(),
            },
            None,
            &ReportSpec::failing(),
            Setup::PRE,
        ),
        is(ExpertAdmissionError::GateFailed),
    )?;
    cx.count("gate_bypass_count", verdict.violation());
    let other = rt.proposal(StructuralOp::Birth {
        result: rt.spec("o"),
    })?;
    let own_proposal = rt.proposal(StructuralOp::Birth { result: own })?;
    let borrowed = report_for(&other, None, &ReportSpec::correct())?;
    let grant = rt.admission_grant(Setup::PRE)?;
    let verdict = cx.verdict(
        "self_promotion",
        "self promotion on a report bound to another proposal",
        rt.commit_proposal(&own_proposal, &borrowed, &grant, None),
        is(ExpertAdmissionError::EvidenceProposalMismatch),
    )?;
    cx.count("verifier_bypass_count", verdict.violation());

    // direct_serving_registry_mutation: the registry is readable and never writable from
    // outside (`expert_registry` returns a shared borrow: type-level evidence), and the only
    // other writer of serving state, the public commit of a structural event, refuses.
    cx.sub("direct_serving_registry_mutation");
    let mut donor = Rt::fresh()?;
    donor.birth("zz")?;
    let forged = donor
        .events()
        .pop()
        .ok_or_else(|| "the donor committed nothing".to_string())?
        .event;
    let verdict = cx.verdict(
        "direct_serving_registry_mutation",
        "a public commit of a birth",
        rt.runtime.commit(forged),
        outside_admission_path,
    )?;
    let served = rt.routable("zz", 1);
    cx.count(
        "direct_serving_mutation_bypass_count",
        verdict.violation() + i64::from(served),
    );

    // uncommitted_change: refused attempts leave no trace, and the registry is what a replay
    // of the ledger builds (nothing is effective that is not committed).
    cx.sub("uncommitted_change");
    let stale = StructuralOp::Freeze {
        target: rt.target_with("m", &|_, mut init| {
            init.generation = ptr_types::Generation(init.generation.0 + 1);
            init
        })?,
    };
    let verdict = cx.verdict(
        "uncommitted_change",
        "a stale freeze",
        rt.admit(stale),
        is_where(|e| matches!(e, ExpertAdmissionError::StaleOperand { .. })),
    )?;
    cx.count("uncommitted_effective_changes", verdict.violation());
    let unchanged = rt.event_count() == before.0 && rt.runtime.expert_registry() == &before.1;
    // The refused attempts above appended nothing and changed nothing.
    cx.count("uncommitted_effective_changes", i64::from(!unchanged));
    cx.count(
        "uncommitted_effective_changes",
        i64::from(!matches_replay(&rt)?),
    );
    Ok(())
}

// ---- C2 bypass and direct append --------------------------------------------------------------

/// C2: the public `commit` of each structural and fence event type, per type, report-less,
/// failed-report and wrong-grant admission, and the identity sets of the event types.
pub fn c2_bypass_and_direct_append(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    run_history(&mut rt, HISTORY.len())?;
    let events = rt.events();
    // The event types the gated path committed, read off the ledger.
    for committed in &events {
        if let Some(name) = event_type_of(&committed.event) {
            cx.set_add("admitted_event_types", name);
        }
    }
    // Direct append, per type: the exact event that the gated step committed, offered to the
    // public `commit` of a durable runtime that stands exactly where the step started. The nine
    // types of the gated history, then the two runtime-granted compound events, which a
    // proposal can neither commit nor fence with: `InputInvalidated` (it fences the experts
    // resting on an input) and `ExpertAuthorityChanged` (it moves the gate configuration or
    // the evaluator standing).
    cx.sub("direct_append");
    for name in cx.shuffled("c2.types", &super::super::params::DIRECT_APPEND_EVENT_TYPES) {
        let (mut stand, event, before) =
            match HISTORY.iter().position(|(candidate, _)| *candidate == name) {
                Some(index) => {
                    let mut stand = Rt::fresh()?;
                    run_history(&mut stand, index)?;
                    (stand, events[index].event.clone(), index)
                }
                None => compound_direct_append(cx, name)?,
            };
        let verdict = cx.verdict(
            &format!("direct_ledger_append_{name}"),
            &format!("a public commit of {name}"),
            stand.runtime.commit(event),
            outside_admission_path,
        )?;
        cx.set_add("direct_append_attempted", name);
        cx.count_in(
            "direct_ledger_append_accepted_by_type",
            name,
            verdict.violation(),
        );
        cx.count("unauthorized_direct_appends_admitted", verdict.violation());
        if name == "ExpertPromoted" {
            cx.count("direct_promotion_bypass_count", verdict.violation());
        }
        let applied = stand.event_count() != before;
        cx.count("uncommitted_effective_changes", i64::from(applied));
        if name == "InputInvalidated" {
            // The dependent of the input must not have been fenced by the refused event.
            let closed = probe_bystanders(&stand, &["x"])?;
            cx.count("direct_serving_mutation_bypass_count", closed.overfenced);
        }
        if name == "ExpertAuthorityChanged" {
            // Neither authority may have a standing after the refused event.
            for kind in [
                ExpertAuthorityKind::GateConfig,
                ExpertAuthorityKind::Evaluator,
            ] {
                cx.count(
                    "forbidden_control_mutation_count",
                    i64::from(stand.runtime.expert_authority(kind).is_some()),
                );
            }
            let closed = probe_bystanders(&stand, &["x"])?;
            cx.count("direct_serving_mutation_bypass_count", closed.overfenced);
        }
    }
    // The revocation-driven retirement is a fence action of its own, outside the eleven types;
    // its direct append is attempted as well (internal counter).
    let mut stand = Rt::fresh()?;
    stand.birth("r")?;
    stand.revoke("k", 1)?;
    stand.fence_revoked("r", "k", 1)?;
    let retire_event = LedgerEvent::ExpertFenceCommitted {
        project: stand.project.clone(),
        pod: PodId::from("r"),
        action: ExpertFenceAction::Retire,
        cause: ExpertFenceCause::InputRevoked,
        resume: ExpertFenceResume::Serving,
        generation: ptr_types::Generation(1),
        expected_state_version: 2,
        epoch: 3,
        revocation: Some(ptr_ledger::ExpertFenceRevocation {
            subject: "k".into(),
            generation: ptr_types::Generation(1),
        }),
        evidence: None,
    };
    let verdict = classify_direct(&mut stand, retire_event)?;
    cx.count("unauthorized_direct_appends_admitted", verdict.violation());
    cx.note("direct_append_retirement_attempts");

    // The bypass attempts. A well-formed record carrying a FAILING report, appended directly:
    // a gate bypass.
    cx.sub("gate_bypass_attempt");
    let mut target = Rt::fresh()?;
    let proposal = target.proposal(StructuralOp::Birth {
        result: target.spec("n"),
    })?;
    let failing = report_for(&proposal, None, &ReportSpec::failing())?;
    let event = structural_event(&proposal, &failing, false)?;
    let verdict = cx.verdict(
        "gate_bypass_attempt",
        "a direct append carrying a failed report",
        target.runtime.commit(event),
        outside_admission_path,
    )?;
    cx.count("gate_bypass_count", verdict.violation());
    cx.count("direct_promotion_bypass_count", verdict.violation());

    // verifier_skip: a record that embeds no report at all.
    cx.sub("verifier_skip");
    let passing = report_for(&proposal, None, &ReportSpec::correct())?;
    let event = structural_event(&proposal, &passing, true)?;
    let verdict = cx.verdict(
        "verifier_skip",
        "a direct append of a report-less record",
        target.runtime.commit(event),
        outside_admission_path,
    )?;
    cx.count("verifier_bypass_count", verdict.violation());
    cx.count("gate_bypass_count", verdict.violation());

    // verifier_override: through the admission path, a failing verdict and a passing verdict
    // of another gate configuration are the evidence offered in place of the host's.
    cx.sub("verifier_override");
    for (what, spec, expected) in [
        (
            "a failing verdict",
            ReportSpec::failing(),
            ExpertAdmissionError::GateFailed,
        ),
        (
            "a verdict of a looser gate",
            ReportSpec::weaker_gate(),
            ExpertAdmissionError::GateConfigMismatch,
        ),
    ] {
        let verdict = cx.verdict(
            "verifier_override",
            what,
            target.admit_as(
                StructuralOp::Birth {
                    result: target.spec("n"),
                },
                None,
                &spec,
                Setup::PRE,
            ),
            is(expected),
        )?;
        cx.count("verifier_override_count", verdict.violation());
        cx.count("gate_bypass_count", verdict.violation());
    }

    // Wrong-grant admission: a grant of another project and of another proposer.
    cx.sub("wrong_grant");
    let proposal = target.proposal(StructuralOp::Birth {
        result: target.spec("w"),
    })?;
    let report = report_for(&proposal, None, &ReportSpec::correct())?;
    let wrong_project = ExpertAdmissionGrant::new(
        ProjectId::from("elsewhere"),
        proposer_of(&ProjectId::from("elsewhere")),
        Setup::PRE.requirement()?,
        target.binding(ExpertGrantKind::Admission),
    );
    let verdict = classify_admission(
        target.commit_proposal(&proposal, &report, &wrong_project, None),
        is(ExpertAdmissionError::ProjectMismatch),
    )?;
    cx.count("unauthorized_direct_appends_admitted", verdict.violation());
    let wrong_proposer = ExpertAdmissionGrant::new(
        target.project.clone(),
        ptr_pods::ProposerRef::new(
            target.project.clone(),
            PodId::from("someone-else"),
            ptr_types::Generation(1),
        )
        .map_err(fail("a proposer"))?,
        Setup::PRE.requirement()?,
        target.binding(ExpertGrantKind::Admission),
    );
    let verdict = classify_admission(
        target.commit_proposal(&proposal, &report, &wrong_proposer, None),
        is(ExpertAdmissionError::ProposerMismatch),
    )?;
    cx.count("unauthorized_direct_appends_admitted", verdict.violation());

    // The identity sets: the types attempted directly are exactly the types the gated path
    // committed.
    let attempted = cx.set_digest("direct_append_attempted");
    let admitted = cx.set_digest("admitted_event_types");
    cx.tally.set_json(
        "direct_append_attempted_event_type_set_digest",
        Json::str(attempted),
    );
    cx.tally
        .set_json("admitted_event_type_set_digest", Json::str(admitted));
    Ok(())
}

/// The runtime-granted compound events offered to the public `commit`: a donor runtime commits
/// the event through its real path, a second runtime stands where the donor started, and the
/// donor's event is what the attempt offers. Returns the standing runtime, the event and the
/// event count the standing runtime must still have afterwards. The donor's committed event is
/// read back, so the type is the one the ledger holds; `admitted_event_types` takes it from
/// there.
fn compound_direct_append(cx: &mut Cx, name: &str) -> Res<(Rt, LedgerEvent, usize)> {
    use super::fixture::{inputs, D1, TRAIN};
    let build = || -> Res<Rt> {
        let mut rt = Rt::fresh()?;
        rt.birth_with("x", inputs(&[(TRAIN, 1, D1)], &[]))?;
        Ok(rt)
    };
    let mut donor = build()?;
    match name {
        "InputInvalidated" => {
            donor
                .invalidate(InputInvalidation::revoked(
                    TRAIN,
                    ptr_types::Generation(1),
                    D1,
                ))
                .map_err(fail("committing the donor's invalidation"))?;
        }
        "ExpertAuthorityChanged" => {
            donor
                .change_authority(
                    ExpertAuthorityKind::GateConfig,
                    GATE_EPOCH,
                    ExpertAuthorityClass::Revoke,
                )
                .map_err(fail("committing the donor's authority change"))?;
        }
        other => return Err(format!("no compound donor for the type {other}")),
    }
    let event = donor
        .events()
        .pop()
        .ok_or_else(|| "the donor committed nothing".to_string())?
        .event;
    if event_type_of(&event) != Some(name) {
        return Err(format!("the donor's last event is not an {name}"));
    }
    cx.set_add("admitted_event_types", name);
    let stand = build()?;
    let before = stand.event_count();
    Ok((stand, event, before))
}

fn classify_direct(rt: &mut Rt, event: LedgerEvent) -> Res<Verdict> {
    classify(
        "a public commit",
        rt.runtime.commit(event),
        outside_admission_path,
    )
}

fn classify_admission(
    result: Result<ptr_types::CommitIndex, RuntimeError>,
    expected: impl Fn(&RuntimeError) -> bool,
) -> Res<Verdict> {
    classify("an admission", result, expected)
}

// ---- C3 stale operands ------------------------------------------------------------------------

/// The CAS fields a stale operand is tried on, with the field the refusal names.
#[derive(Clone, Copy, Debug)]
enum Stale {
    Generation,
    Digest,
    StateVersion,
    State,
}

impl Stale {
    const ALL: [Stale; 4] = [
        Self::Generation,
        Self::Digest,
        Self::StateVersion,
        Self::State,
    ];

    fn field(self) -> ExpertStaleField {
        match self {
            Self::Generation => ExpertStaleField::Generation,
            Self::Digest => ExpertStaleField::ManifestDigest,
            Self::StateVersion => ExpertStaleField::StateVersion,
            Self::State => ExpertStaleField::State,
        }
    }

    fn apply(self, mut init: ptr_pods::ExpertTargetInit) -> ptr_pods::ExpertTargetInit {
        match self {
            Self::Generation => init.generation = ptr_types::Generation(init.generation.0 + 1),
            Self::Digest => {
                init.manifest_digest =
                    ptr_pods::ManifestDigest::new([0xEE; 32]).expect("a non-zero digest");
            }
            Self::StateVersion => init.state_version += 1,
            Self::State => {
                init.state = if init.state == ptr_pods::ExpertState::Serving {
                    ptr_pods::ExpertState::Frozen
                } else {
                    ptr_pods::ExpertState::Serving
                };
            }
        }
        init
    }
}

/// C3: a stale generation and a current-generation-but-stale-digest commit per operation type,
/// the other compare-and-set fields, a stale secondary operand of Merge and Prune; Birth uses
/// the occupied-result and artifact-digest analogues.
pub fn c3_stale_operands(cx: &mut Cx) -> Res<()> {
    for scenario in cx.shuffled("c3.scenarios", &SCENARIOS) {
        let kind = scenario.event_type();
        cx.sub(&format!("stale_{}", scenario.label()));
        if scenario == Scenario::Birth {
            // The stale generation of a birth: the result pod is already occupied.
            let mut rt = Rt::fresh()?;
            rt.birth("a")?;
            let (events, registry) = (rt.event_count(), rt.runtime.expert_registry().clone());
            let verdict = cx.verdict(
                "stale_generation_commit_ExpertPromoted",
                "a birth onto an occupied pod",
                rt.admit(StructuralOp::Birth {
                    result: rt.spec("a"),
                }),
                is(ExpertAdmissionError::ResultPodOccupied { pod: "a".into() }),
            )?;
            cx.count_in(
                "stale_generation_commits_accepted_by_type",
                kind,
                verdict.violation(),
            );
            cx.count("stale_generation_commits_accepted", verdict.violation());
            cx.count("stale_operand_commits", verdict.violation());
            // The stale digest of a birth: the report's artifact digest is not the result's.
            let verdict = cx.verdict(
                "stale_digest_commit_ExpertPromoted",
                "a birth on a report of another artifact digest",
                rt.admit_as(
                    StructuralOp::Birth {
                        result: rt.spec("n"),
                    },
                    None,
                    &ReportSpec::wrong_artifact(),
                    Setup::PRE,
                ),
                is(ExpertAdmissionError::EvidenceArtifactMismatch),
            )?;
            cx.count_in(
                "stale_digest_commits_accepted_by_type",
                kind,
                verdict.violation(),
            );
            cx.count("stale_digest_commits_accepted", verdict.violation());
            cx.count("stale_operand_commits", verdict.violation());
            unchanged(cx, &rt, events, &registry);
            continue;
        }
        for stale in Stale::ALL {
            // The claimed state must still be one the proposal accepts for the operation: a
            // Frozen claim is built for Merge, Split and Retire only.
            if matches!(stale, Stale::State)
                && !matches!(
                    scenario,
                    Scenario::Merge | Scenario::Split | Scenario::Retire
                )
            {
                continue;
            }
            let mut rt = Rt::fresh()?;
            rt.seed(scenario)?;
            let (events, registry) = (rt.event_count(), rt.runtime.expert_registry().clone());
            let op = rt.operation(scenario, &|_, init| stale.apply(init))?;
            let first = scenario.operands()[0];
            let case = match stale {
                Stale::Generation => Some(format!("stale_generation_commit_{kind}")),
                Stale::Digest => Some(format!("stale_digest_commit_{kind}")),
                Stale::StateVersion | Stale::State => None,
            };
            let expected = is(ExpertAdmissionError::StaleOperand {
                pod: first.to_string(),
                field: stale.field(),
            });
            let result = rt.admit(op);
            let verdict = match &case {
                Some(case) => cx.verdict(case, "a stale operand", result, expected)?,
                None => classify("a stale operand", result, expected)?,
            };
            cx.count("stale_operand_commits", verdict.violation());
            match stale {
                Stale::Generation => {
                    cx.count_in(
                        "stale_generation_commits_accepted_by_type",
                        kind,
                        verdict.violation(),
                    );
                    cx.count("stale_generation_commits_accepted", verdict.violation());
                }
                Stale::Digest => {
                    cx.count_in(
                        "stale_digest_commits_accepted_by_type",
                        kind,
                        verdict.violation(),
                    );
                    cx.count("stale_digest_commits_accepted", verdict.violation());
                }
                Stale::StateVersion | Stale::State => {}
            }
            unchanged(cx, &rt, events, &registry);
            // The control: the operation as the registry holds it commits.
            let op = rt.operation(scenario, &exact)?;
            control(&format!("{} control", scenario.label()), rt.admit(op))?;
        }
    }
    // A stale secondary operand: only the second source / target is stale.
    cx.sub("stale_secondary_operand");
    for (scenario, key, case) in [
        (
            Scenario::Merge,
            "Merge_sources",
            "stale_secondary_operand_merge_source",
        ),
        (
            Scenario::Prune,
            "Prune_targets",
            "stale_secondary_operand_prune_target",
        ),
    ] {
        let mut rt = Rt::fresh()?;
        rt.seed(scenario)?;
        let (events, registry) = (rt.event_count(), rt.runtime.expert_registry().clone());
        let op = rt.operation(scenario, &|pod, init| {
            if pod == "b" {
                Stale::Generation.apply(init)
            } else {
                init
            }
        })?;
        let verdict = cx.verdict(
            case,
            "a stale secondary operand",
            rt.admit(op),
            is(ExpertAdmissionError::StaleOperand {
                pod: "b".into(),
                field: ExpertStaleField::Generation,
            }),
        )?;
        cx.count_in("stale_secondary_operand_cases", key, 1);
        cx.count("stale_operand_commits", verdict.violation());
        unchanged(cx, &rt, events, &registry);
    }
    Ok(())
}

/// A refused attempt appended nothing and left the registry as it was.
fn unchanged(
    cx: &mut Cx,
    rt: &Rt,
    events: usize,
    registry: &ptr_runtime::expert::ExpertRegistryProjection,
) {
    let same = rt.event_count() == events && rt.runtime.expert_registry() == registry;
    cx.count("uncommitted_effective_changes", i64::from(!same));
}

// ---- C4 evidence binding ----------------------------------------------------------------------

/// The four negative kinds of the evidence binding, with the report each uses and the refusal
/// it must meet.
fn negative_kinds() -> [(&'static str, ReportSpec, ExpertAdmissionError); 4] {
    [
        (
            "wrong_artifact",
            ReportSpec::wrong_artifact(),
            ExpertAdmissionError::EvidenceArtifactMismatch,
        ),
        (
            "wrong_base_model",
            ReportSpec::wrong_base_model(),
            ExpertAdmissionError::EvidenceBaseModelMismatch,
        ),
        (
            "eval_manifest_of_other_evaluation",
            ReportSpec::other_eval_manifest(),
            ExpertAdmissionError::EvidenceEvalManifestMismatch,
        ),
        (
            "gate_config_of_other_evaluation",
            ReportSpec::other_gate(),
            ExpertAdmissionError::GateConfigMismatch,
        ),
    ]
}

/// The event types of C4 with the scenario that exercises each (`None`: the readmission).
fn evidence_types() -> [(&'static str, Option<Scenario>); 9] {
    [
        ("ExpertPromoted", Some(Scenario::Birth)),
        ("ExpertRetired", Some(Scenario::Retire)),
        ("Merge", Some(Scenario::Merge)),
        ("Split", Some(Scenario::Split)),
        ("Prune", Some(Scenario::Prune)),
        ("Freeze", Some(Scenario::Freeze)),
        ("Reactivate", Some(Scenario::ReactivateFrozen)),
        // The Reactivate that creates a new generation binds its artifact to the result.
        ("Reactivate", Some(Scenario::ReactivateRetired)),
        ("ExpertReadmitted", None),
    ]
}

/// C4: four negative kinds for each of eight gate-driven event types, each with its matching
/// positive control; cross-proposal evidence reuse. Admission and readmission share one
/// evidence-admission function.
pub fn c4_evidence_binding(cx: &mut Cx) -> Res<()> {
    for (kind_type, scenario) in cx.shuffled("c4.types", &evidence_types()) {
        for (kind, spec, expected) in negative_kinds() {
            let case = format!("evidence_negative_{kind}_{kind_type}");
            cx.sub(&case);
            let mut rt = Rt::fresh()?;
            let op = match scenario {
                Some(scenario) => {
                    rt.seed(scenario)?;
                    Some(rt.operation(scenario, &exact)?)
                }
                None => {
                    rt.birth("a")?;
                    rt.fence_quarantine("a")?;
                    None
                }
            };
            let (events, registry) = (rt.event_count(), rt.runtime.expert_registry().clone());
            let result = match &op {
                Some(op) => rt.admit_as(op.clone(), None, &spec, Setup::PRE),
                None => readmit_with(&mut rt, &spec),
            };
            // The four kinds of a promotion carry names of their own in the pinned suite.
            let verdict = match promotion_alias(kind, kind_type) {
                Some(alias) => cx.verdict(&alias, &case, result, is(expected))?,
                None => classify(&case, result, is(expected))?,
            };
            cx.count_in(
                "evidence_binding_negative_cases_per_event_type",
                &format!("{kind_type}.{kind}"),
                1,
            );
            cx.count_in(
                "evidence_binding_cases_accepted_by_case",
                &case,
                verdict.violation(),
            );
            cx.count("evidence_binding_admissions", verdict.violation());
            unchanged(cx, &rt, events, &registry);
            // The matching positive control: the same operation on its own evidence.
            let controlled = match op {
                Some(op) => rt.admit(op),
                None => rt.readmit("a", Setup::PRE),
            };
            control(&format!("{case} control"), controlled)?;
            cx.count_in("evidence_binding_controls_accepted_by_case", &case, 1);
            cx.count_in("control_accepts_by_path", kind_type, 1);
            if kind_type == "Reactivate" {
                // Reactivate is run in two forms under one name, and each cell of the family
                // counts both; the notes say how many cells each form filled.
                cx.note(match scenario {
                    Some(Scenario::ReactivateFrozen) => "c4_reactivate_frozen_form_cells",
                    _ => "c4_reactivate_retired_form_cells",
                });
            }
        }
    }
    // The weaker gate configuration of a promotion.
    cx.sub("promotion_with_weaker_gate_config");
    let mut rt = Rt::fresh()?;
    let verdict = cx.verdict(
        "promotion_with_weaker_gate_config",
        "a promotion on a looser gate's verdict",
        rt.admit_as(
            StructuralOp::Birth {
                result: rt.spec("n"),
            },
            None,
            &ReportSpec::weaker_gate(),
            Setup::PRE,
        ),
        is(ExpertAdmissionError::GateConfigMismatch),
    )?;
    cx.count("evidence_binding_admissions", verdict.violation());

    // Cross-proposal evidence reuse: a valid Freeze report presented for a Prune or a Retire
    // of the same target generation; the control is the same operation on its own evidence.
    cx.sub("cross_proposal_evidence_reuse");
    for scenario in [Scenario::Prune, Scenario::Retire] {
        let mut rt = Rt::fresh()?;
        rt.birth("a")?;
        if scenario == Scenario::Prune {
            rt.birth("b")?;
        }
        let freeze = rt.proposal(StructuralOp::Freeze {
            target: rt.target("a")?,
        })?;
        let freeze_report = report_for(&freeze, None, &ReportSpec::correct())?;
        let victim_op = if scenario == Scenario::Prune {
            StructuralOp::Prune {
                targets: vec![rt.target("a")?],
            }
        } else {
            StructuralOp::Retire {
                target: rt.target("a")?,
                reason: RetiredReason::Superseded,
            }
        };
        let victim = rt.proposal(victim_op.clone())?;
        let grant = rt.admission_grant(Setup::PRE)?;
        let verdict = cx.verdict(
            "cross_proposal_evidence_reuse",
            "a Freeze report presented for another operation",
            rt.commit_proposal(&victim, &freeze_report, &grant, None),
            is(ExpertAdmissionError::EvidenceProposalMismatch),
        )?;
        cx.count("cross_proposal_evidence_reuse_attempts", 1);
        cx.count("cross_proposal_evidence_admissions", verdict.violation());
        control(
            "the same operation on its own evidence",
            rt.admit(victim_op),
        )?;
        cx.count("cross_proposal_evidence_controls_accepted", 1);
    }
    Ok(())
}

/// The readmission of the quarantined `a` on a report built from `spec`.
fn readmit_with(rt: &mut Rt, spec: &ReportSpec) -> Result<ptr_types::CommitIndex, RuntimeError> {
    let request = rt
        .fence_request("a", ExpertFenceCause::AuthorityQuarantine)
        .map_err(RuntimeError::InvalidConfig)?;
    let manifest = rt.manifest("a").map_err(RuntimeError::InvalidConfig)?;
    let report =
        readmission_report(&request, manifest, spec).map_err(RuntimeError::InvalidConfig)?;
    let grant = rt
        .gate_grant(Setup::PRE)
        .map_err(RuntimeError::InvalidConfig)?;
    rt.runtime.readmit_expert(&request, &report, &grant)
}

/// The adversarial case of the pinned suite a negative attempt is also counted under: the four
/// kinds of a promotion carry names of their own; every other cell is part of the
/// evidence-binding family only.
fn promotion_alias(kind: &str, event_type: &str) -> Option<String> {
    let name = match (event_type, kind) {
        ("ExpertPromoted", "wrong_artifact") => "promotion_with_evidence_for_other_artifact",
        ("ExpertPromoted", "wrong_base_model") => "promotion_with_evidence_for_other_base_model",
        ("ExpertPromoted", "eval_manifest_of_other_evaluation") => {
            "promotion_with_eval_manifest_of_other_evaluation"
        }
        ("ExpertPromoted", "gate_config_of_other_evaluation") => {
            "promotion_with_gate_config_of_other_evaluation"
        }
        _ => return None,
    };
    Some(name.to_string())
}

// ---- C5 impersonation -------------------------------------------------------------------------

/// The impersonation kinds with the scenario each runs on and the ledger op kind its committed
/// control must carry.
fn impersonation_kinds() -> [(&'static str, Scenario, ExpertOpKind); 8] {
    [
        ("Birth", Scenario::Birth, ExpertOpKind::Birth),
        ("Birth_targetless", Scenario::Birth, ExpertOpKind::Birth),
        ("Merge", Scenario::Merge, ExpertOpKind::Merge),
        ("Split", Scenario::Split, ExpertOpKind::Split),
        ("Freeze", Scenario::Freeze, ExpertOpKind::Freeze),
        (
            "Reactivate",
            Scenario::ReactivateFrozen,
            ExpertOpKind::Reactivate,
        ),
        ("Prune", Scenario::Prune, ExpertOpKind::Prune),
        ("Retire", Scenario::Retire, ExpertOpKind::Retire),
    ]
}

/// C5: same-project and cross-project impersonation for each operation kind, including a
/// targetless Birth, with a positive control of the legitimate proposer.
pub fn c5_impersonation(cx: &mut Cx) -> Res<()> {
    for (kind, scenario, op_kind) in cx.shuffled("c5.kinds", &impersonation_kinds()) {
        cx.sub(&format!("impersonation_{kind}"));
        let mut rt = Rt::fresh()?;
        rt.seed(scenario)?;
        let (events, registry) = (rt.event_count(), rt.runtime.expert_registry().clone());
        let op = rt.operation(scenario, &exact)?;
        let op = if kind == "Birth_targetless" {
            // The same operation as Birth, onto a pod of its own.
            StructuralOp::Birth {
                result: rt.spec("t"),
            }
        } else {
            op
        };
        let proposal = rt.proposal(op)?;
        let report = report_for(&proposal, None, &ReportSpec::correct())?;
        // Cross-project: the grant authenticates a proposer of another project.
        let attacker = ProjectId::from("attacker-project");
        let foreign = ExpertAdmissionGrant::new(
            attacker.clone(),
            proposer_of(&attacker),
            Setup::PRE.requirement()?,
            rt.binding(ExpertGrantKind::Admission),
        );
        let verdict = cx.verdict(
            "cross_project_impersonation",
            "a proposal of another project",
            rt.commit_proposal(&proposal, &report, &foreign, None),
            is(ExpertAdmissionError::ProjectMismatch),
        )?;
        cx.count_in("cross_project_impersonation_attempts", kind, 1);
        cx.set_add("impersonation_attempted_op_kinds", kind);
        cx.count("impersonated_proposals_admitted", verdict.violation());
        // Same-project: the grant authenticates another policy pod of the same project.
        let other = ptr_pods::ProposerRef::new(
            rt.project.clone(),
            PodId::from("impostor"),
            ptr_types::Generation(1),
        )
        .map_err(fail("a proposer"))?;
        let impostor = ExpertAdmissionGrant::new(
            rt.project.clone(),
            other,
            Setup::PRE.requirement()?,
            rt.binding(ExpertGrantKind::Admission),
        );
        let verdict = cx.verdict(
            "same_project_impersonation",
            "a proposal naming another proposer",
            rt.commit_proposal(&proposal, &report, &impostor, None),
            is(ExpertAdmissionError::ProposerMismatch),
        )?;
        cx.count_in("same_project_impersonation_attempts", kind, 1);
        cx.count("impersonated_proposals_admitted", verdict.violation());
        unchanged(cx, &rt, events, &registry);
        // The control: the authenticated proposer's own proposal commits, and the ledger's op
        // kind is the one the kind names.
        let grant = rt.admission_grant(Setup::PRE)?;
        control(
            &format!("{kind} control"),
            rt.commit_proposal(&proposal, &report, &grant, None),
        )?;
        cx.count("same_project_proposals_accepted", 1);
        match rt.runtime.committed_events().last().map(|c| &c.event) {
            Some(LedgerEvent::ExpertStructuralCommitted { op_kind: got, .. })
                if *got == op_kind =>
            {
                cx.set_add("admitted_op_kinds", kind);
            }
            other => return Err(format!("the {kind} control committed {other:?}")),
        }
    }
    let attempted = cx.set_digest("impersonation_attempted_op_kinds");
    let admitted = cx.set_digest("admitted_op_kinds");
    cx.tally.set_json(
        "impersonation_attempted_op_kind_set_digest",
        Json::str(attempted),
    );
    cx.tally
        .set_json("admitted_op_kind_set_digest", Json::str(admitted));
    Ok(())
}

// ---- C6 fence authority and the route probe ---------------------------------------------------

/// C6: a fence under the runtime grant needs no gate pass and is admitted only with that grant;
/// an L3 policy, a Gate grant or a stale grant cannot fence; the route probe refuses a fenced
/// expert after the fault, the restart, the replay and the compaction.
pub fn c6_fence_authority(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    for pod in ["x", "y", "z"] {
        rt.birth(pod)?;
        // The live control: the expert is routable before anything happens to it.
        route_control(&format!("{pod} before the fence"), rt.route(pod, 1))?;
    }
    // The control: a fence under the runtime revocation grant, with no gate pass.
    cx.sub("fence_with_grant");
    let request = rt.fence_request("x", ExpertFenceCause::AuthorityQuarantine)?;
    let grant = rt.fence_grant();
    control(
        "a fence under the runtime grant",
        rt.runtime.fence_expert(&request, &grant),
    )?;
    cx.attempt("expert_fenced_without_gate_pass")?;
    cx.count("runtime_fence_with_grant_succeeded", 1);
    cx.count_in("control_accepts_by_path", "ExpertFenced", 1);
    let fenced = classify_route("x after its fence", rt.route("x", 1), RouteExpect::Fenced)?;
    cx.count("revoked_expert_served_count", fenced.served());

    // L3 fence attempts on y: the event offered to the public commit; a fence under a Gate
    // grant; a fence under the runtime-wide grant (not accepted for a per-expert fence).
    cx.sub("l3_fence_attempt");
    let event = LedgerEvent::ExpertFenceCommitted {
        project: rt.project.clone(),
        pod: PodId::from("y"),
        action: ExpertFenceAction::Fence,
        cause: ExpertFenceCause::AuthorityQuarantine,
        resume: ExpertFenceResume::Serving,
        generation: ptr_types::Generation(1),
        expected_state_version: 1,
        epoch: 2,
        revocation: None,
        evidence: None,
    };
    let verdict = cx.verdict(
        "l3_fence_attempt",
        "a public commit of a fence",
        rt.runtime.commit(event),
        outside_admission_path,
    )?;
    cx.count("l3_fence_attempts_admitted", verdict.violation());
    let request = rt.fence_request("y", ExpertFenceCause::AuthorityQuarantine)?;
    let gate_grant = rt.gate_grant(Setup::PRE)?;
    let verdict = cx.verdict(
        "l3_fence_attempt",
        "a fence under a Gate grant",
        rt.runtime.fence_expert(&request, &gate_grant),
        is(ExpertAdmissionError::FenceAuthorityMismatch {
            action: ExpertFenceAction::Fence,
            required: ExpertFenceAuthority::RuntimeRevocation,
        }),
    )?;
    cx.count("l3_fence_attempts_admitted", verdict.violation());
    let global = rt.global_grant();
    let verdict = cx.verdict(
        "l3_fence_attempt",
        "a fence under the runtime-wide grant",
        rt.runtime.fence_expert(&request, &global),
        is_where(|e| matches!(e, ExpertAdmissionError::FenceAuthorityMismatch { .. })),
    )?;
    cx.count("l3_fence_attempts_admitted", verdict.violation());

    // Fence without the runtime grant: a grant of another project and a grant superseded
    // between mint and commit.
    cx.sub("fence_without_runtime_grant");
    let foreign = ExpertFenceGrant::revocation(
        ProjectId::from("elsewhere"),
        rt.binding(ExpertGrantKind::Fence),
    );
    let verdict = cx.verdict(
        "fence_without_runtime_grant",
        "a fence under a grant of another project",
        rt.runtime.fence_expert(&request, &foreign),
        is(ExpertAdmissionError::FenceProjectMismatch),
    )?;
    cx.count("fence_without_runtime_grant_admitted", verdict.violation());
    let minted = rt.fence_grant();
    rt.runtime
        .supersede_expert_grant(ExpertGrantKind::Fence)
        .map_err(fail("superseding the fence grant"))?;
    let verdict = cx.verdict(
        "fence_without_runtime_grant",
        "a fence under a superseded grant",
        rt.runtime.fence_expert(&request, &minted),
        is_where(|e| {
            matches!(
                e,
                ExpertAdmissionError::GrantGenerationStale {
                    kind: ExpertGrantKind::Fence,
                    ..
                }
            )
        }),
    )?;
    cx.count("fence_without_runtime_grant_admitted", verdict.violation());
    // Every fence attempt on y was refused, so y must still be routable: a closed y means an
    // attempt took effect (a fence nobody was entitled to write).
    let closed = probe_bystanders(&rt, &["y", "z"])?;
    cx.count("l3_fence_attempts_admitted", closed.overfenced);
    cx.count("fence_without_runtime_grant_admitted", closed.overfenced);

    // The route probe after the fence, a restart, a replay and a compaction: x must be refused
    // as `Fenced` at every point (any other refusal, an unknown expert after a restore for
    // one, is a harness error), and the bystanders y and z must stay routable after each.
    cx.sub("restart_replay_route_after_fence");
    let mut found = RouteTally::default();
    found.merge(probe_routes(&rt, &["x"], RouteExpect::Fenced)?);
    found.merge(probe_bystanders(&rt, &["y", "z"])?);
    rt.reopen()?;
    found.merge(probe_routes(&rt, &["x"], RouteExpect::Fenced)?);
    found.merge(probe_bystanders(&rt, &["y", "z"])?);
    let replayed = rt.replayed()?;
    found.add(classify_route(
        "x on a replay",
        replayed.expert_route_admission(&rt.project, &PodId::from("x"), ptr_types::Generation(1)),
        RouteExpect::Fenced,
    )?);
    for pod in ["y", "z"] {
        found.overfenced += i64::from(super::classify_bystander(
            pod,
            replayed.expert_route_admission(
                &rt.project,
                &PodId::from(pod),
                ptr_types::Generation(1),
            ),
        )?);
    }
    rt.compact_restore()?;
    found.merge(probe_routes(&rt, &["x"], RouteExpect::Fenced)?);
    found.merge(probe_bystanders(&rt, &["y", "z"])?);
    cx.attempt("restart_replay_route_after_fence")?;
    cx.count("expert_routes_after_replay", found.served);
    cx.count_fence_findings(&FenceDiff::default(), &found);

    // revoked_expert_serve: an expert fenced for a committed revocation is never served, also
    // after the same faults.
    cx.sub("revoked_expert_serve");
    let mut rt = Rt::fresh()?;
    rt.birth("z")?;
    rt.birth("w")?;
    // The live control: z is routable before the revocation and before the fence.
    route_control("z before the revocation", rt.route("z", 1))?;
    rt.revoke("k", 1)?;
    route_control("z after the revocation, before the fence", rt.route("z", 1))?;
    rt.fence_revoked("z", "k", 1)?;
    let mut found = RouteTally::default();
    found.merge(probe_routes(&rt, &["z"], RouteExpect::Fenced)?);
    found.merge(probe_bystanders(&rt, &["w"])?);
    rt.reopen()?;
    found.merge(probe_routes(&rt, &["z"], RouteExpect::Fenced)?);
    found.merge(probe_bystanders(&rt, &["w"])?);
    found.add(classify_route(
        "z on a replay",
        rt.replayed()?.expert_route_admission(
            &rt.project,
            &PodId::from("z"),
            ptr_types::Generation(1),
        ),
        RouteExpect::Fenced,
    )?);
    rt.compact_restore()?;
    found.merge(probe_routes(&rt, &["z"], RouteExpect::Fenced)?);
    found.merge(probe_bystanders(&rt, &["w"])?);
    cx.attempt("revoked_expert_serve")?;
    cx.count("revoked_expert_served_count", found.served);
    cx.count_fence_findings(&FenceDiff::default(), &found);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::Scale;

    /// The two compound events are event types of the direct-append family: each donor event is
    /// a genuine event of its type, the public commit refuses it as a structural record, and a
    /// refused offer leaves the standing runtime as it was (the dependent still routable, no
    /// authority standing).
    #[test]
    fn the_compound_events_are_offered_to_the_public_commit_and_refused() {
        let mut cx = Cx::new(17, Scale::Reduced).expect("a context");
        for name in ["InputInvalidated", "ExpertAuthorityChanged"] {
            let (mut stand, event, before) = compound_direct_append(&mut cx, name).expect(name);
            assert_eq!(event_type_of(&event), Some(name));
            let verdict = classify_direct(&mut stand, event).expect(name);
            assert_eq!(verdict, Verdict::Refused, "{name}");
            assert_eq!(stand.event_count(), before, "{name}");
            assert!(stand.routable("x", 1), "{name}");
            for kind in [
                ExpertAuthorityKind::GateConfig,
                ExpertAuthorityKind::Evaluator,
            ] {
                assert!(stand.runtime.expert_authority(kind).is_none(), "{name}");
            }
        }
        assert!(compound_direct_append(&mut cx, "Merge").is_err());
        assert!(cx.sets["admitted_event_types"].contains("InputInvalidated"));
        assert!(cx.sets["admitted_event_types"].contains("ExpertAuthorityChanged"));
    }

    #[test]
    fn the_event_types_of_the_family_are_the_configs() {
        for name in super::super::super::params::DIRECT_APPEND_EVENT_TYPES {
            let history = HISTORY.iter().any(|(step, _)| *step == name);
            let compound = ["InputInvalidated", "ExpertAuthorityChanged"].contains(&name);
            assert!(
                history != compound,
                "{name} has exactly one way to be offered"
            );
        }
    }
}
