//! F2: transitive authority. F3: superseded sources.
//!
//! F2. An expert `a` is admitted under epoch E1 of an authority kind (the gate configuration or
//! the evaluator); the kind is rotated without tightening to E2 (nothing is fenced); a
//! descendant of `a` (a `Merge` result, a `Split` result of such a result, or the `Reactivate`
//! of `a` itself after its retirement) is admitted under E2; then the authority is revoked (E3).
//! Every serving expert whose evidence came from an earlier epoch is owed a fence, and no
//! descendant of `a` may route afterwards. The descendant's own epoch is E2, below E3: the case
//! asks that no path leave a descendant serving that rests on a revoked authority through its
//! source. Each cell runs on the normal, lag (reopen) and restore (compacted snapshot) paths.
//!
//! F3. A source is superseded (consumed by a `Merge` or a `Split`), a live descendant remains,
//! and then an input of the source, or the attestor behind it, is revoked. The descendant's own
//! block names neither (or names another attestor), so it rests on the revoked input only
//! through its source: the compound event must fence it through the source edge, the source
//! being no longer live. (The authority kinds of this family are F2's `Merge` and `Split`
//! forms: the source is consumed there too.)

use super::dormant::Path;
use super::{
    must_attested, must_under, regime, regime_rt, Attestor, Cx, EVALUATOR_START, GATE_START,
};
use crate::experiments::layered_kv::lab::Res;
use crate::experiments::r011::cases::fixture::{inputs, spec_at, Rt, Setup, D1, D3, OTHER, TRAIN};
use crate::experiments::r011::cases::{
    classify_bystander, classify_route, FenceDiff, RouteExpect, RouteTally,
};
use ptr_ledger::{ExpertAuthorityClass, ExpertAuthorityKind};
use ptr_pods::{ExpertInputs, RetiredReason, StructuralOp};
use ptr_runtime::expert::InputInvalidation;
use ptr_types::Generation;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthKind {
    GateConfig,
    Evaluator,
}

impl AuthKind {
    const ALL: [AuthKind; 2] = [Self::GateConfig, Self::Evaluator];

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

    fn start(self) -> u64 {
        match self {
            Self::GateConfig => GATE_START,
            Self::Evaluator => EVALUATOR_START,
        }
    }

    /// The setup of evidence under the standing `steps` epochs after the regime.
    fn setup(self, steps: u64) -> Setup {
        match self {
            Self::GateConfig => Setup {
                epoch: GATE_START + steps,
                version: (GATE_START + steps) as u8,
                evaluator: Some(EVALUATOR_START as u8),
            },
            Self::Evaluator => Setup {
                evaluator: Some((EVALUATOR_START + steps) as u8),
                ..regime()
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Form {
    Merge,
    Split,
    Reactivate,
}

impl Form {
    const ALL: [Form; 3] = [Self::Merge, Self::Split, Self::Reactivate];

    fn key(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Split => "split",
            Self::Reactivate => "reactivate",
        }
    }
}

pub fn transitive_authority(cx: &mut Cx) -> Res<()> {
    for kind in cx.shuffled("lineage.authority_kinds", &AuthKind::ALL) {
        for form in cx.shuffled(&format!("lineage.forms.{}", kind.key()), &Form::ALL) {
            for path in cx.shuffled(
                &format!("lineage.authority_paths.{}.{}", kind.key(), form.key()),
                &Path::ALL,
            ) {
                cx.sub(&format!("{}_{}_{}", kind.key(), form.key(), path.name()));
                authority_cell(cx, kind, form, path)?;
            }
        }
    }
    Ok(())
}

fn authority_cell(cx: &mut Cx, kind: AuthKind, form: Form, path: Path) -> Res<()> {
    let mut rt = regime_rt()?;
    let e1 = regime();
    let e2 = kind.setup(1);
    let e3 = kind.setup(2);
    for pod in ["a", "c"] {
        let op = StructuralOp::Birth {
            result: rt.spec(pod),
        };
        must_under(&mut rt, &format!("birth {pod}"), op, None, e1)?;
    }
    if form == Form::Reactivate {
        let op = StructuralOp::Retire {
            target: rt.target("a")?,
            reason: RetiredReason::GateFailed,
        };
        must_under(&mut rt, "retire a", op, None, e1)?;
    }
    let rotation = rt
        .change_authority(
            kind.authority(),
            kind.start() + 1,
            ExpertAuthorityClass::NonTightening,
        )
        .map_err(|error| format!("the non-tightening rotation: {error:?}"))?;
    if rotation.fence_count() != 0 {
        return Err("a non-tightening rotation fenced experts".to_string());
    }
    // The descendants of `a`, admitted under E2.
    let (descendants, owed): (Vec<&str>, Vec<&str>) = match form {
        Form::Merge => {
            let op = StructuralOp::Merge {
                sources: vec![rt.target("a")?, rt.target("c")?],
                result: rt.spec("b"),
            };
            must_under(&mut rt, "merge a c", op, None, e2)?;
            (vec!["b"], vec!["b"])
        }
        Form::Split => {
            let op = StructuralOp::Merge {
                sources: vec![rt.target("a")?, rt.target("c")?],
                result: rt.spec("b"),
            };
            must_under(&mut rt, "merge a c", op, None, e2)?;
            let op = StructuralOp::Split {
                source: rt.target("b")?,
                results: vec![rt.spec("x"), rt.spec("y")],
            };
            must_under(&mut rt, "split b", op, None, e2)?;
            (vec!["x", "y"], vec!["x", "y"])
        }
        Form::Reactivate => {
            let target = rt.target("a")?;
            let next = spec_at(&rt.project, "a", target.generation().0 + 1, target.level());
            let op = StructuralOp::Reactivate {
                target,
                result: Some(next),
            };
            must_under(&mut rt, "reactivate a", op, None, e2)?;
            (vec!["a"], vec!["a", "c"])
        }
    };
    for pod in &descendants {
        let generation = rt.init(pod)?.generation.0;
        if !rt.routable(pod, generation) {
            return Err(format!(
                "the descendant {pod} does not route before the revocation"
            ));
        }
    }
    // The revocation of the authority.
    let outcome = rt
        .change_authority(
            kind.authority(),
            kind.start() + 2,
            ExpertAuthorityClass::Revoke,
        )
        .map_err(|error| format!("the revocation: {error:?}"))?;
    cx.count_in("transitive_authority_revocation_cases", kind.key(), 1);
    if matches!(form, Form::Merge | Form::Split) {
        cx.count_in("superseded_source_revocation_cases", kind.key(), 1);
    }
    let fenced: Vec<String> = outcome
        .fenced()
        .iter()
        .map(|member| member.pod.0.clone())
        .collect();
    let diff = FenceDiff::of(owed.iter().copied(), fenced.iter().map(String::as_str));
    path.apply(&mut rt)?;
    let mut probes = RouteTally::default();
    for pod in &descendants {
        let generation = rt.init(pod)?.generation.0;
        probes.add(classify_route(
            pod,
            rt.route(pod, generation),
            RouteExpect::Fenced,
        )?);
    }
    let case = match kind {
        AuthKind::GateConfig => "route_descendant_after_authority_revocation_gate_config",
        AuthKind::Evaluator => "route_descendant_after_authority_revocation_evaluator",
    };
    cx.attempt(case)?;
    cx.count(
        "routable_descendants_after_source_authority_revocation",
        probes.served,
    );
    if matches!(form, Form::Merge | Form::Split) {
        cx.count(
            "descendants_routable_after_superseded_source_revocation",
            probes.served,
        );
        cx.attempt(match kind {
            AuthKind::GateConfig => {
                "route_descendant_of_superseded_source_after_gate_config_revocation"
            }
            AuthKind::Evaluator => {
                "route_descendant_of_superseded_source_after_evaluator_revocation"
            }
        })?;
    }
    cx.count("fence_set_mismatches", diff.mismatches() + probes.unfenced);
    // A bystander admitted after the revocation, on the new standing.
    let op = StructuralOp::Birth {
        result: rt.spec("n"),
    };
    must_under(&mut rt, "bystander", op, None, e3)?;
    cx.count(
        "bystanders_closed",
        i64::from(classify_bystander("n", rt.route("n", 1))?),
    );
    if path != Path::Restore {
        cx.check_replay(&rt)?;
    }
    Ok(())
}

// ---- F3 ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DepKind {
    Input,
    Attestor,
}

impl DepKind {
    const ALL: [DepKind; 2] = [Self::Input, Self::Attestor];

    fn key(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Attestor => "attestor",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Consume {
    Merge,
    Split,
}

impl Consume {
    const ALL: [Consume; 2] = [Self::Merge, Self::Split];

    fn key(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Split => "split",
        }
    }
}

const BAD: Attestor = Attestor::new("alpha", 1);
const GOOD: Attestor = Attestor::new("beta", 1);

pub fn superseded_sources(cx: &mut Cx) -> Res<()> {
    for kind in cx.shuffled("superseded.kinds", &DepKind::ALL) {
        for consume in cx.shuffled(&format!("superseded.forms.{}", kind.key()), &Consume::ALL) {
            for path in cx.shuffled(
                &format!("superseded.paths.{}.{}", kind.key(), consume.key()),
                &Path::ALL,
            ) {
                cx.sub(&format!("{}_{}_{}", kind.key(), consume.key(), path.name()));
                dependency_cell(cx, kind, consume, path)?;
            }
        }
    }
    Ok(())
}

/// `block` for a pod that rests on `OTHER` only, as an attestation extension or a plain block.
fn live_block() -> ExpertInputs {
    inputs(&[(OTHER, 1, D3)], &[])
}

fn dependency_cell(cx: &mut Cx, kind: DepKind, consume: Consume, path: Path) -> Res<()> {
    let mut rt = Rt::fresh()?;
    let setup = Setup::PRE;
    let birth = |rt: &Rt, pod: &str| StructuralOp::Birth {
        result: rt.spec(pod),
    };
    // The source `s` rests on the bad input / attestor; `t` and the bystander `w` do not.
    match kind {
        DepKind::Input => {
            let op = birth(&rt, "s");
            must_under(
                &mut rt,
                "birth s",
                op,
                Some(inputs(&[(TRAIN, 1, D1)], &[])),
                setup,
            )?;
            for pod in ["t", "w"] {
                let op = birth(&rt, pod);
                must_under(
                    &mut rt,
                    &format!("birth {pod}"),
                    op,
                    Some(live_block()),
                    setup,
                )?;
            }
        }
        DepKind::Attestor => {
            let op = birth(&rt, "s");
            must_attested(
                &mut rt,
                "birth s",
                op,
                ExpertInputs::default(),
                BAD,
                1,
                setup,
            )?;
            for (pod, run) in [("t", 2), ("w", 3)] {
                let op = birth(&rt, pod);
                must_attested(
                    &mut rt,
                    &format!("birth {pod}"),
                    op,
                    ExpertInputs::default(),
                    GOOD,
                    run,
                    setup,
                )?;
            }
        }
    }
    // The source is consumed; the descendants are admitted on live inputs / a live attestor.
    let (descendants, run): (Vec<&str>, u8) = match consume {
        Consume::Merge => (vec!["d"], 4),
        Consume::Split => (vec!["d", "e"], 4),
    };
    let op = match consume {
        Consume::Merge => StructuralOp::Merge {
            sources: vec![rt.target("s")?, rt.target("t")?],
            result: rt.spec("d"),
        },
        Consume::Split => StructuralOp::Split {
            source: rt.target("s")?,
            results: vec![rt.spec("d"), rt.spec("e")],
        },
    };
    match kind {
        DepKind::Input => must_under(&mut rt, "consume s", op, Some(live_block()), setup)?,
        DepKind::Attestor => must_attested(
            &mut rt,
            "consume s",
            op,
            ExpertInputs::default(),
            GOOD,
            run,
            setup,
        )?,
    }
    for pod in &descendants {
        if !rt.routable(pod, 1) {
            return Err(format!(
                "the descendant {pod} does not route before the revocation"
            ));
        }
    }
    let request = match kind {
        DepKind::Input => InputInvalidation::revoked(TRAIN, Generation(1), D1),
        DepKind::Attestor => {
            InputInvalidation::revoked(BAD.identity(), Generation(BAD.epoch), BAD.digest())
        }
    };
    let outcome = rt
        .invalidate(request)
        .map_err(|error| format!("the revocation: {error:?}"))?;
    cx.count_in("superseded_source_revocation_cases", kind.key(), 1);
    let fenced: Vec<String> = outcome
        .fenced()
        .iter()
        .map(|member| member.pod.0.clone())
        .collect();
    let diff = FenceDiff::of(
        descendants.iter().copied(),
        fenced.iter().map(String::as_str),
    );
    path.apply(&mut rt)?;
    let mut probes = RouteTally::default();
    for pod in &descendants {
        probes.add(classify_route(pod, rt.route(pod, 1), RouteExpect::Fenced)?);
    }
    cx.attempt(match kind {
        DepKind::Input => "route_descendant_of_superseded_source_after_input_revocation",
        DepKind::Attestor => "route_descendant_of_superseded_source_after_attestor_revocation",
    })?;
    cx.count(
        "descendants_routable_after_superseded_source_revocation",
        probes.served,
    );
    cx.count("fence_set_mismatches", diff.mismatches() + probes.unfenced);
    cx.count(
        "bystanders_closed",
        i64::from(classify_bystander("w", rt.route("w", 1))?),
    );
    if path != Path::Restore {
        cx.check_replay(&rt)?;
    }
    Ok(())
}
