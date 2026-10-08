//! C12: the authority epoch for the gate configuration and the evaluator identity.
//!
//! Evidence must come from the current epoch of each kind; a revocation or a tightening
//! rotation fences, in the same ledger event and with cause `AuthorityQuarantine`, every expert
//! admitted under an earlier epoch; a non-tightening rotation fences nobody; a quarantined
//! expert is readmitted only through the gated path on evidence under the current standing.
//! The same events run on the normal, lag (dropped and reopened) and restore (compacted
//! snapshot) paths of the matrix.

use super::fixture::{ReportSpec, Rt, Setup};
use super::{classify, control, Cx};
use crate::experiments::layered_kv::lab::{fail, Res};
use ptr_ledger::{ExpertAuthorityClass, ExpertAuthorityKind, ExpertFenceCause};
use ptr_pods::{ExpertState, StructuralOp};
use ptr_runtime::expert::{ExpertAdmissionError, StaleEvidenceReason};
use ptr_runtime::RuntimeError;
use std::collections::BTreeSet;

/// The gate standing of the regime the cases start from: gate epoch 2 (configuration version
/// 2) and evaluator epoch 1.
const GATE_START: u64 = 2;
const EVALUATOR_START: u64 = 1;

/// The setup of the evidence the regime's own experts were admitted under.
fn regime() -> Setup {
    Setup {
        epoch: GATE_START,
        version: GATE_START as u8,
        evaluator: Some(EVALUATOR_START as u8),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    GateConfig,
    Evaluator,
}

impl Kind {
    const ALL: [Kind; 2] = [Self::GateConfig, Self::Evaluator];

    fn key(self) -> &'static str {
        match self {
            Self::GateConfig => "gate_config",
            Self::Evaluator => "evaluator",
        }
    }

    fn authority(self) -> ExpertAuthorityKind {
        match self {
            Self::GateConfig => ExpertAuthorityKind::GateConfig,
            Self::Evaluator => ExpertAuthorityKind::Evaluator,
        }
    }

    /// The epoch an event moves the kind to (the regime stands one below).
    fn next_epoch(self) -> u64 {
        match self {
            Self::GateConfig => GATE_START + 1,
            Self::Evaluator => EVALUATOR_START + 1,
        }
    }

    /// The setup of evidence under the standing after the event.
    fn current(self) -> Setup {
        match self {
            Self::GateConfig => Setup {
                epoch: GATE_START + 1,
                version: (GATE_START + 1) as u8,
                evaluator: Some(EVALUATOR_START as u8),
            },
            Self::Evaluator => Setup {
                evaluator: Some((EVALUATOR_START + 1) as u8),
                ..regime()
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Class {
    Revoke,
    Tightening,
    NonTightening,
}

impl Class {
    fn authority(self) -> ExpertAuthorityClass {
        match self {
            Self::Revoke => ExpertAuthorityClass::Revoke,
            Self::Tightening => ExpertAuthorityClass::TighteningRotation,
            Self::NonTightening => ExpertAuthorityClass::NonTightening,
        }
    }

    /// The type as the matrix cells name it.
    fn cell(self) -> &'static str {
        match self {
            Self::Revoke => "revocation",
            Self::Tightening => "tightening_rotation",
            Self::NonTightening => "non_tightening_rotation",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Path {
    Normal,
    Lag,
    Restore,
}

impl Path {
    const ALL: [Path; 3] = [Self::Normal, Self::Lag, Self::Restore];

    fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Lag => "lag",
            Self::Restore => "restore",
        }
    }
}

/// A runtime at the regime: both kinds have a standing (set by non-tightening rotations that
/// fence nothing) and `a`, `b` were admitted under it.
fn regime_rt() -> Res<Rt> {
    let mut rt = Rt::fresh()?;
    for (kind, epoch) in [
        (ExpertAuthorityKind::GateConfig, GATE_START),
        (ExpertAuthorityKind::Evaluator, EVALUATOR_START),
    ] {
        let outcome = rt
            .change_authority(kind, epoch, ExpertAuthorityClass::NonTightening)
            .map_err(fail("starting the regime"))?;
        if outcome.fence_count() != 0 {
            return Err("starting the regime fenced experts".to_string());
        }
    }
    for pod in ["a", "b"] {
        let op = StructuralOp::Birth {
            result: rt.spec(pod),
        };
        rt.admit_as(op, None, &ReportSpec::correct_under(regime()), regime())
            .map_err(|error| format!("admitting {pod} under the regime: {error:?}"))?;
    }
    Ok(rt)
}

fn birth_under(
    rt: &mut Rt,
    pod: &str,
    setup: Setup,
) -> Result<ptr_types::CommitIndex, RuntimeError> {
    let op = StructuralOp::Birth {
        result: rt.spec(pod),
    };
    rt.admit_as(op, None, &ReportSpec::correct_under(setup), setup)
}

fn routable_among(rt: &Rt, pods: &[&str]) -> i64 {
    pods.iter().filter(|pod| rt.routable(pod, 1)).count() as i64
}

fn is_stale(kind: Kind) -> impl Fn(&RuntimeError) -> bool {
    move |error| {
        matches!(
            error,
            RuntimeError::Expert(ExpertAdmissionError::StalePolicyEvidence { kind: found, .. })
                if *found == kind.authority()
        )
    }
}

pub fn c12_authority_epoch(cx: &mut Cx) -> Res<()> {
    stale_policy_evidence(cx)?;
    non_tightening_rotation(cx)?;
    readmission(cx)?;
    for kind in cx.shuffled("c12.kinds", &Kind::ALL) {
        for class in [Class::Revoke, Class::Tightening] {
            for path in Path::ALL {
                cx.sub(&format!("{}_{}_{}", kind.key(), path.name(), class.cell()));
                cell(cx, kind, class, path)?;
            }
        }
    }
    Ok(())
}

/// A proposal queued across an epoch update and committed with evidence of the old epoch.
fn stale_policy_evidence(cx: &mut Cx) -> Res<()> {
    cx.sub("stale_policy_evidence");
    for kind in Kind::ALL {
        let mut rt = regime_rt()?;
        // The proposal is queued: the evidence and the grant are minted under the regime.
        let proposal = rt.proposal(StructuralOp::Birth {
            result: rt.spec("n"),
        })?;
        let report =
            super::fixture::report_for(&proposal, None, &ReportSpec::correct_under(regime()))?;
        let grant = rt.admission_grant(regime())?;
        // The epoch of the kind moves (a rotation that fences nothing keeps the registry as it
        // is, so only the evidence is stale).
        rt.change_authority(
            kind.authority(),
            kind.next_epoch(),
            ExpertAuthorityClass::NonTightening,
        )
        .map_err(fail("rotating the authority"))?;
        let verdict = cx.verdict(
            "stale_policy_evidence",
            &format!("a commit with {} evidence of the old epoch", kind.key()),
            rt.commit_proposal(&proposal, &report, &grant, None),
            is_stale(kind),
        )?;
        cx.count("stale_policy_evidence_attempts", 1);
        cx.count("stale_policy_evidence_admissions", verdict.violation());
        // The control: the same operation under evidence of the current epoch.
        control(
            &format!("{} evidence of the current epoch", kind.key()),
            birth_under(&mut rt, "n", kind.current()),
        )?;
        cx.count("stale_policy_evidence_controls_accepted", 1);
    }
    Ok(())
}

/// A rotation that does not tighten leaves the promoted experts routable.
fn non_tightening_rotation(cx: &mut Cx) -> Res<()> {
    cx.sub("non_tightening_rotation_control");
    for kind in Kind::ALL {
        let mut rt = regime_rt()?;
        if routable_among(&rt, &["a", "b"]) != 2 {
            return Err("the regime's experts are not routable".to_string());
        }
        let outcome = rt
            .change_authority(
                kind.authority(),
                kind.next_epoch(),
                Class::NonTightening.authority(),
            )
            .map_err(fail("rotating without tightening"))?;
        cx.attempt("non_tightening_rotation_control")?;
        let kept = routable_among(&rt, &["a", "b"]);
        if outcome.fence_count() == 0 && kept == 2 {
            cx.count("non_tightening_rotation_controls", 1);
        } else {
            // The control failed: a non-tightening rotation fenced promoted experts.
            return Err(format!(
                "a non-tightening rotation of {} fenced {} experts and left {kept} routable",
                kind.key(),
                outcome.fence_count()
            ));
        }
    }
    Ok(())
}

/// A quarantined expert is readmitted only on evidence under the current standing.
fn readmission(cx: &mut Cx) -> Res<()> {
    cx.sub("authority_quarantine_readmission");
    for kind in Kind::ALL {
        let mut rt = regime_rt()?;
        rt.change_authority(
            kind.authority(),
            kind.next_epoch(),
            Class::Tightening.authority(),
        )
        .map_err(fail("tightening the authority"))?;
        for pod in ["a", "b"] {
            if rt
                .runtime
                .expert_registry()
                .fence_cause(&rt.project, &ptr_types::PodId::from(pod))
                != Some(ExpertFenceCause::AuthorityQuarantine)
            {
                return Err(format!("{pod} was not quarantined by the tightening"));
            }
        }
        // Evidence of the old epoch.
        let verdict = cx.verdict(
            "authority_quarantine_readmission",
            &format!("a readmission on {} evidence of the old epoch", kind.key()),
            rt.readmit("a", regime()),
            is_stale(kind),
        )?;
        cx.count(
            "readmitted_without_current_epoch_evidence",
            verdict.violation(),
        );
        // Evidence of the current epoch.
        control(
            &format!(
                "a readmission on {} evidence of the current epoch",
                kind.key()
            ),
            rt.readmit("a", kind.current()),
        )?;
        cx.count_in("authority_quarantine_readmission_cases", kind.key(), 1);
        cx.count("readmission_controls_accepted", 1);
        if !rt.routable("a", 1) || rt.routable("b", 1) {
            return Err("the readmission did not return exactly the readmitted expert".to_string());
        }
        let _ = StaleEvidenceReason::IdentityNotNamed;
    }
    Ok(())
}

/// One cell of the matrix: the event of `class` on `kind`, then the path, then the probes.
fn cell(cx: &mut Cx, kind: Kind, class: Class, path: Path) -> Res<()> {
    let name = format!("{}_{}_{}", kind.key(), path.name(), class.cell());
    let mut rt = regime_rt()?;
    let dependents = ["a", "b"];
    let outcome = rt
        .change_authority(kind.authority(), kind.next_epoch(), class.authority())
        .map_err(|error| format!("{name}: {error:?}"))?;
    if outcome.fence_count() as usize != dependents.len() {
        return Err(format!(
            "{name}: the event fenced {} experts, the topology derives {}",
            outcome.fence_count(),
            dependents.len()
        ));
    }
    // An expert admitted under the new standing is not touched by the event.
    control(
        &format!("{name}: an expert under the new standing"),
        birth_under(&mut rt, "c", kind.current()),
    )?;
    match path {
        Path::Normal => {}
        Path::Lag => rt.reopen()?,
        Path::Restore => rt.compact_restore()?,
    }
    let mut hits = routable_among(&rt, &dependents);
    if !rt.routable("c", 1) {
        return Err(format!(
            "{name}: the expert under the new standing was fenced"
        ));
    }
    // The standing survived the path: evidence of the old epoch is refused.
    let stale = classify(
        &format!("{name}: a commit on evidence of the old epoch"),
        birth_under(&mut rt, "d", regime()),
        is_stale(kind),
    )?;
    hits += stale.violation();
    let standing = rt
        .runtime
        .expert_authority(kind.authority())
        .map(|s| s.epoch());
    if standing != Some(kind.next_epoch()) {
        hits += 1;
    }
    cx.count_in("invalidation_matrix_cases", &name, 1);
    cx.count_in("invalidation_matrix_stale_hits", &name, hits);
    // The named metrics, counted on the normal path.
    if path == Path::Normal {
        match class {
            Class::Revoke => {
                cx.attempt("post_promotion_authority_revocation")?;
                cx.count_in("post_promotion_authority_revocation_cases", kind.key(), 1);
                cx.count(
                    "routable_after_authority_revocation",
                    routable_among(&rt, &dependents),
                );
            }
            Class::Tightening => {
                cx.attempt("tightening_rotation")?;
                cx.count_in("tightening_rotation_cases", kind.key(), 1);
                cx.count(
                    "routable_after_tightening_rotation",
                    routable_among(&rt, &dependents),
                );
            }
            Class::NonTightening => {}
        }
    }
    // The set the topology derives and the set the runtime fenced and refuses to route.
    let scenario = format!("authority_{name}");
    let mut fenced = BTreeSet::new();
    for pod in dependents {
        cx.set_add("derived_dependents", &format!("{scenario}/{pod}"));
        let fenced_now = matches!(rt.state(pod), Some((ExpertState::Fenced, _)))
            && matches!(
                rt.route(pod, 1),
                Err(ptr_runtime::expert::RouteRefusal::Fenced)
            );
        if fenced_now {
            fenced.insert(format!("{scenario}/{pod}"));
        }
    }
    for member in fenced {
        cx.set_add("fenced_and_probed", &member);
    }
    Ok(())
}
