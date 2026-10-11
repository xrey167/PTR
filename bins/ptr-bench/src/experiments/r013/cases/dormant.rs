//! F1: dormant seeds. An expert is retired (`Retired(GateFailed)`), so it is no longer
//! `Serving` or `Frozen` and no fence set contains it; then the authority or the input it
//! rested on is revoked; then a `Reactivate` of the retired seed is attempted.
//!
//! The kinds are the gate configuration and the evaluator (an authority revoked), an input (a
//! training input revoked) and an attestor (an attestor revoked, the attestor being an input
//! identity). A seed is *compromised* when what it rested on is revoked:
//!
//! * for an input or an attestor, any reactivation is owed a refusal, whatever block it names
//!   (revoked data is not laundered by retraining): the seed's own block and a block that omits
//!   the revoked input are both attempted;
//! * for an authority kind, a reactivation on the evidence of the revoked authority (the
//!   evidence the seed's retirement stood under) is owed a refusal as stale evidence. A
//!   reactivation on evidence under the standing that replaced it is a re-evaluation of the same
//!   weights under the current authority and is the positive control, as a readmission of a
//!   quarantined expert is (R011).
//!
//! Each kind runs on the normal path, after a reopen of the durable ledger (`lag`) and on a
//! runtime restored from a compacted snapshot (`restore`), and an unaffected retired seed
//! reactivates in every cell, so a refusal is not a blanket refusal.

use super::{
    admit_with_attestation, must_attested, must_under, refused_with, regime, regime_rt, Attestor,
    Cx, EVALUATOR_START, GATE_START,
};
use crate::experiments::layered_kv::lab::Res;
use crate::experiments::r011::cases::fixture::{
    inputs, spec_at, ReportSpec, Rt, Setup, D1, D3, OTHER, TRAIN,
};
use ptr_ledger::{ExpertAuthorityClass, ExpertAuthorityKind};
use ptr_pods::{ExpertInputs, RetiredReason, StructuralOp};
use ptr_runtime::expert::{
    ExpertAdmissionError, ExpertInputRole, InputInvalidation, StaleEvidenceReason,
};
use ptr_runtime::RuntimeError;
use ptr_types::Generation;

const SEED: &str = "s";
const UNAFFECTED: &str = "u";
const BYSTANDER: &str = "n";
const A: Attestor = Attestor::new("alpha", 1);
const B: Attestor = Attestor::new("beta", 1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    GateConfig,
    Evaluator,
    Input,
    Attestor,
}

impl Kind {
    pub const ALL: [Kind; 4] = [
        Self::GateConfig,
        Self::Evaluator,
        Self::Input,
        Self::Attestor,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Self::GateConfig => "gate_config",
            Self::Evaluator => "evaluator",
            Self::Input => "input",
            Self::Attestor => "attestor",
        }
    }

    fn authority(self) -> Option<ExpertAuthorityKind> {
        match self {
            Self::GateConfig => Some(ExpertAuthorityKind::GateConfig),
            Self::Evaluator => Some(ExpertAuthorityKind::Evaluator),
            Self::Input | Self::Attestor => None,
        }
    }

    /// The setup of evidence under the standing after the revocation of an authority kind.
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
            Self::Input | Self::Attestor => regime(),
        }
    }

    /// The refusal the revoked authority's evidence is owed.
    fn stale(self) -> impl Fn(&RuntimeError) -> bool {
        move |error| {
            let reason_ok = |kind: ExpertAuthorityKind, reason: &StaleEvidenceReason| match kind {
                ExpertAuthorityKind::GateConfig => {
                    matches!(reason, StaleEvidenceReason::EpochMismatch { .. })
                }
                ExpertAuthorityKind::Evaluator => {
                    matches!(reason, StaleEvidenceReason::IdentityMismatch)
                }
            };
            matches!(
                error,
                RuntimeError::Expert(ExpertAdmissionError::StalePolicyEvidence { kind, reason })
                    if Some(*kind) == self.authority() && reason_ok(*kind, reason)
            )
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Path {
    Normal,
    Lag,
    Restore,
}

impl Path {
    pub const ALL: [Path; 3] = [Self::Normal, Self::Lag, Self::Restore];

    pub fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Lag => "lag",
            Self::Restore => "restore",
        }
    }

    /// Move the runtime to the path's state.
    pub fn apply(self, rt: &mut Rt) -> Res<()> {
        match self {
            Self::Normal => Ok(()),
            Self::Lag => rt.reopen(),
            Self::Restore => rt.compact_restore(),
        }
    }
}

pub fn revoke_while_retired(cx: &mut Cx) -> Res<()> {
    for kind in cx.shuffled("dormant.kinds", &Kind::ALL) {
        for path in cx.shuffled(&format!("dormant.paths.{}", kind.key()), &Path::ALL) {
            cx.sub(&format!("{}_{}", kind.key(), path.name()));
            cell(cx, kind, path)?;
        }
    }
    Ok(())
}

/// The retired seeds of a cell and the experts that must stay routable.
fn build(kind: Kind) -> Res<Rt> {
    let mut rt = regime_rt()?;
    let setup = regime();
    let birth = |rt: &Rt, pod: &str| StructuralOp::Birth {
        result: rt.spec(pod),
    };
    match kind {
        Kind::GateConfig | Kind::Evaluator => {
            for pod in [SEED, UNAFFECTED] {
                let op = birth(&rt, pod);
                must_under(&mut rt, &format!("birth {pod}"), op, None, setup)?;
            }
        }
        Kind::Input => {
            let op = birth(&rt, SEED);
            must_under(
                &mut rt,
                "birth s",
                op,
                Some(inputs(&[(TRAIN, 1, D1)], &[])),
                setup,
            )?;
            let op = birth(&rt, UNAFFECTED);
            must_under(
                &mut rt,
                "birth u",
                op,
                Some(inputs(&[(OTHER, 1, D3)], &[])),
                setup,
            )?;
        }
        Kind::Attestor => {
            let op = birth(&rt, SEED);
            must_attested(&mut rt, "birth s", op, ExpertInputs::default(), A, 1, setup)?;
            let op = birth(&rt, UNAFFECTED);
            must_attested(&mut rt, "birth u", op, ExpertInputs::default(), B, 2, setup)?;
        }
    }
    for pod in [SEED, UNAFFECTED] {
        retire(&mut rt, pod, setup)?;
    }
    Ok(rt)
}

/// Retire `pod` through the gate (`Retired(GateFailed)`, the one reason a seed can reactivate
/// from) under `setup`.
fn retire(rt: &mut Rt, pod: &str, setup: Setup) -> Res<()> {
    let op = StructuralOp::Retire {
        target: rt.target(pod)?,
        reason: RetiredReason::GateFailed,
    };
    must_under(rt, &format!("retire {pod}"), op, None, setup)
}

/// The revocation of what the seeds rested on.
fn revoke(rt: &mut Rt, kind: Kind) -> Res<()> {
    match kind {
        Kind::GateConfig | Kind::Evaluator => {
            let (authority, epoch) = match kind {
                Kind::GateConfig => (ExpertAuthorityKind::GateConfig, GATE_START + 1),
                _ => (ExpertAuthorityKind::Evaluator, EVALUATOR_START + 1),
            };
            rt.change_authority(authority, epoch, ExpertAuthorityClass::Revoke)
                .map_err(|error| format!("revoking the {} authority: {error:?}", kind.key()))?;
        }
        Kind::Input => {
            rt.invalidate(InputInvalidation::revoked(TRAIN, Generation(1), D1))
                .map_err(|error| format!("revoking the training input: {error:?}"))?;
        }
        Kind::Attestor => {
            rt.invalidate(InputInvalidation::revoked(
                A.identity(),
                Generation(A.epoch),
                A.digest(),
            ))
            .map_err(|error| format!("revoking the attestor: {error:?}"))?;
        }
    }
    Ok(())
}

/// A reactivation of the retired `pod` as its next generation, in the form `kind` calls for.
enum Form {
    /// On the evidence of the regime the seed was retired under, resting on what it rested on.
    Original,
    /// Resting on other, live inputs only (the revoked one omitted).
    Laundered,
    /// On evidence under the current standing, or a seed that rested on nothing revoked.
    Current,
}

fn reactivate(
    rt: &mut Rt,
    kind: Kind,
    pod: &str,
    form: Form,
    run: u8,
) -> Result<ptr_types::CommitIndex, RuntimeError> {
    let target = rt.target(pod).map_err(RuntimeError::InvalidConfig)?;
    let next = spec_at(&rt.project, pod, target.generation().0 + 1, target.level());
    let op = StructuralOp::Reactivate {
        target,
        result: Some(next),
    };
    let setup = match (kind, &form) {
        (Kind::GateConfig | Kind::Evaluator, Form::Current) => kind.current(),
        _ => regime(),
    };
    let spec = ReportSpec::correct_under(setup);
    match kind {
        Kind::GateConfig | Kind::Evaluator => rt.admit_as(op, None, &spec, setup),
        Kind::Input => {
            let block = match form {
                Form::Original => inputs(&[(TRAIN, 1, D1)], &[]),
                Form::Laundered | Form::Current => inputs(&[(OTHER, 1, D3)], &[]),
            };
            rt.admit_as(op, Some(block), &spec, setup)
        }
        Kind::Attestor => {
            let who = match form {
                Form::Original => A,
                Form::Laundered | Form::Current => B,
            };
            let proposal = rt.proposal(op).map_err(RuntimeError::InvalidConfig)?;
            let attestation = who
                .attest(&proposal, run)
                .map_err(RuntimeError::InvalidConfig)?;
            admit_with_attestation(
                rt,
                &proposal,
                &attestation,
                ExpertInputs::default(),
                &spec,
                setup,
            )
        }
    }
}

fn cell(cx: &mut Cx, kind: Kind, path: Path) -> Res<()> {
    let mut rt = build(kind)?;
    revoke(&mut rt, kind)?;
    cx.count_in("revoke_while_retired_cases", kind.key(), 1);
    path.apply(&mut rt)?;
    cx.note(&format!("dormant_cells.{}.{}", kind.key(), path.name()));

    let routable_seed_before = rt.routable(SEED, rt.init(SEED)?.generation.0);
    if routable_seed_before {
        return Err("a retired seed is routable".to_string());
    }

    // The attempts the revoked seed is owed a refusal of.
    let mut admitted = 0;
    let attempts: Vec<(&str, Form)> = match kind {
        Kind::GateConfig => vec![(
            "reactivate_seed_retired_before_gate_config_revocation",
            Form::Original,
        )],
        Kind::Evaluator => vec![(
            "reactivate_seed_retired_before_evaluator_revocation",
            Form::Original,
        )],
        Kind::Input => vec![
            (
                "reactivate_seed_retired_before_input_revocation",
                Form::Original,
            ),
            (
                "reactivate_seed_retired_before_input_revocation_laundered",
                Form::Laundered,
            ),
        ],
        Kind::Attestor => vec![
            (
                "reactivate_seed_retired_before_attestor_revocation",
                Form::Original,
            ),
            (
                "reactivate_seed_retired_before_attestor_revocation_laundered",
                Form::Laundered,
            ),
        ],
    };
    for (index, (case, form)) in attempts.into_iter().enumerate() {
        let expected: Box<dyn Fn(&RuntimeError) -> bool> = match (kind, &form) {
            (Kind::GateConfig | Kind::Evaluator, _) => Box::new(kind.stale()),
            (_, Form::Laundered) => Box::new(refused_with(|e| {
                matches!(
                    e,
                    ExpertAdmissionError::InputRevoked {
                        role: ExpertInputRole::Inherited,
                        ..
                    }
                )
            })),
            (_, _) => Box::new(refused_with(|e| {
                matches!(e, ExpertAdmissionError::InputRevoked { .. })
            })),
        };
        let events = rt.event_count();
        let verdict = cx.verdict(
            case,
            case,
            reactivate(&mut rt, kind, SEED, form, 10 + index as u8),
            expected,
        )?;
        cx.count("reactivation_attempts", 1);
        admitted += verdict.violation();
        cx.count(
            "reactivations_from_compromised_seed_admitted",
            verdict.violation(),
        );
        if !verdict.admitted() && rt.event_count() != events {
            return Err("a refused reactivation appended an event".to_string());
        }
        if verdict.admitted() {
            // The seed serves again; a further attempt would meet another refusal, which is
            // not the one the case owes. The violation is counted.
            break;
        }
    }
    // A reactivation that got through is also measured at the route: the seed's new generation
    // must not serve.
    let generation = rt.init(SEED)?.generation.0;
    let served = rt.routable(SEED, generation);
    cx.count(
        "compromised_reactivations_routable",
        i64::from(served && admitted > 0),
    );

    // The control: the same refusal is not a blanket one.
    let events = rt.event_count();
    let control = reactivate(&mut rt, kind, UNAFFECTED, Form::Current, 20);
    control.map_err(|error| {
        format!(
            "the control reactivation of an unaffected seed ({} / {}): {error:?}",
            kind.key(),
            path.name()
        )
    })?;
    if rt.event_count() != events + 1 {
        return Err("the control reactivation did not append one event".to_string());
    }
    cx.count_in("reactivation_controls_accepted", kind.key(), 1);
    let generation = rt.init(UNAFFECTED)?.generation.0;
    if !rt.routable(UNAFFECTED, generation) {
        return Err("the reactivated unaffected seed does not route".to_string());
    }

    // A bystander admitted after the revocation, on the current standing, stays routable.
    let setup = kind.current();
    let op = StructuralOp::Birth {
        result: rt.spec(BYSTANDER),
    };
    match kind {
        Kind::Attestor => must_attested(
            &mut rt,
            "bystander",
            op,
            ExpertInputs::default(),
            B,
            30,
            setup,
        )?,
        Kind::Input => must_under(
            &mut rt,
            "bystander",
            op,
            Some(inputs(&[(OTHER, 1, D3)], &[])),
            setup,
        )?,
        _ => must_under(&mut rt, "bystander", op, None, setup)?,
    }
    cx.count("bystanders_closed", i64::from(!rt.routable(BYSTANDER, 1)));
    // A runtime restored from a compacted snapshot holds no history to replay.
    if path != Path::Restore {
        cx.check_replay(&rt)?;
    }
    Ok(())
}
