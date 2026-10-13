//! C8: result collisions, takeover and planned succession.
//!
//! The result of a Birth, Merge, Split or retired-target Reactivate may take a pod or routing
//! slot only from an operand of the same operation; a second proposal naming a foreign pod or
//! slot is refused; a planned succession (the operation retiring its own sources) does not
//! fence its own result.

use super::fixture::{inputs, level_of, spec_at, Rt, D1, TRAIN};
use super::{control, is, is_where, Cx};
use crate::experiments::layered_kv::lab::Res;
use ptr_pods::{ExpertState, RetiredReason, StructuralOp};
use ptr_runtime::expert::{ExpertAdmissionError, InputInvalidation};
use ptr_types::Generation;

pub fn c8_results_and_succession(cx: &mut Cx) -> Res<()> {
    collisions(cx)?;
    takeovers(cx)?;
    planned_succession(cx)
}

/// The slot (layer, index) of a token-level pod of the harness.
fn slot(pod: &str) -> (u16, u16) {
    match level_of(pod) {
        ptr_pods::ExpertLevel::Token { layer, index } => (layer, index),
        ptr_pods::ExpertLevel::Segment => (0, 0),
    }
}

/// Two sequenced proposals naming the same result pod or the same routing slot: the first
/// commits, the second is refused, for each producing operation.
fn collisions(cx: &mut Cx) -> Res<()> {
    cx.sub("colliding_result_race");
    // Birth.
    let mut rt = Rt::fresh()?;
    let (first, pod, slot_taker) = (
        StructuralOp::Birth {
            result: rt.spec("a"),
        },
        StructuralOp::Birth {
            result: rt.spec("a"),
        },
        StructuralOp::Birth {
            result: spec_at(&rt.project, "a2", 1, level_of("a")),
        },
    );
    control("the first birth", rt.admit(first))?;
    let (layer, index) = slot("a");
    for (what, op, expected) in [
        (
            "a second birth of the same pod",
            pod,
            ExpertAdmissionError::ResultPodOccupied { pod: "a".into() },
        ),
        (
            "a second birth onto the same slot",
            slot_taker,
            ExpertAdmissionError::ResultSlotOccupied { layer, index },
        ),
    ] {
        let verdict = cx.verdict("colliding_result_race", what, rt.admit(op), is(expected))?;
        cx.count_in("colliding_result_race_cases", "Birth", 1);
        cx.count("duplicate_result_commits", verdict.violation());
    }

    // Merge: both proposals are built from the same view of the registry.
    let mut rt = Rt::fresh()?;
    for pod in ["a", "b", "c", "d", "e"] {
        rt.birth(pod)?;
    }
    let merge = |rt: &Rt, sources: [&str; 2], result: ptr_pods::ExpertPodSpec| {
        Ok::<_, String>(StructuralOp::Merge {
            sources: vec![rt.target(sources[0])?, rt.target(sources[1])?],
            result,
        })
    };
    let first = merge(&rt, ["a", "b"], rt.spec("m"))?;
    let same_pod = merge(&rt, ["c", "d"], rt.spec("m"))?;
    // The foreign slot: the slot of `e`, which no source of the second merge holds.
    let foreign_slot = merge(&rt, ["c", "d"], spec_at(&rt.project, "n", 1, level_of("e")))?;
    control("the first merge", rt.admit(first))?;
    let (layer, index) = slot("e");
    for (what, op, expected) in [
        (
            "a second merge onto the same result pod",
            same_pod,
            ExpertAdmissionError::ResultPodOccupied { pod: "m".into() },
        ),
        (
            "a merge onto a foreign slot",
            foreign_slot,
            ExpertAdmissionError::ResultSlotOccupied { layer, index },
        ),
    ] {
        let verdict = cx.verdict("colliding_result_race", what, rt.admit(op), is(expected))?;
        cx.count_in("colliding_result_race_cases", "Merge", 1);
        cx.count("duplicate_result_commits", verdict.violation());
    }

    // Split.
    let mut rt = Rt::fresh()?;
    for pod in ["a", "c", "e"] {
        rt.birth(pod)?;
    }
    let split = |rt: &Rt, source: &str, results: Vec<ptr_pods::ExpertPodSpec>| {
        Ok::<_, String>(StructuralOp::Split {
            source: rt.target(source)?,
            results,
        })
    };
    let first = split(&rt, "a", vec![rt.spec("x"), rt.spec("y")])?;
    let same_pod = split(&rt, "c", vec![rt.spec("x"), rt.spec("z")])?;
    let foreign_slot = split(
        &rt,
        "c",
        vec![spec_at(&rt.project, "z", 1, level_of("e")), rt.spec("w")],
    )?;
    control("the first split", rt.admit(first))?;
    let (layer, index) = slot("e");
    for (what, op, expected) in [
        (
            "a second split onto the same result pod",
            same_pod,
            ExpertAdmissionError::ResultPodOccupied { pod: "x".into() },
        ),
        (
            "a split onto a foreign slot",
            foreign_slot,
            ExpertAdmissionError::ResultSlotOccupied { layer, index },
        ),
    ] {
        let verdict = cx.verdict("colliding_result_race", what, rt.admit(op), is(expected))?;
        cx.count_in("colliding_result_race_cases", "Split", 1);
        cx.count("duplicate_result_commits", verdict.violation());
    }

    // Reactivate of a retired target: the result may only be the target's own next generation
    // at its own slot (the proposal constructor refuses another pod or level). The collisions
    // are a foreign birth that takes the released slot between the proposal and its commit,
    // and a second reactivation of the same target.
    let mut rt = Rt::fresh()?;
    rt.birth("a")?;
    rt.retire("a", RetiredReason::GateFailed)?;
    let target = rt.target("a")?;
    let reactivate = StructuralOp::Reactivate {
        target: target.clone(),
        result: Some(spec_at(&rt.project, "a", 2, target.level())),
    };
    let thief = StructuralOp::Birth {
        result: spec_at(&rt.project, "z", 1, target.level()),
    };
    control(
        "a birth onto the slot the retired target released",
        rt.admit(thief),
    )?;
    let (layer, index) = slot("a");
    let verdict = cx.verdict(
        "colliding_result_race",
        "a reactivation onto a slot another expert took",
        rt.admit(reactivate),
        is(ExpertAdmissionError::ResultSlotOccupied { layer, index }),
    )?;
    cx.count_in(
        "colliding_result_race_cases",
        "Reactivate_retired_target",
        1,
    );
    cx.count("duplicate_result_commits", verdict.violation());

    let mut rt = Rt::fresh()?;
    rt.birth("a")?;
    rt.retire("a", RetiredReason::GateFailed)?;
    let target = rt.target("a")?;
    let next = || spec_at(&rt.project, "a", 2, target.level());
    let first = StructuralOp::Reactivate {
        target: target.clone(),
        result: Some(next()),
    };
    let second = StructuralOp::Reactivate {
        target: target.clone(),
        result: Some(next()),
    };
    control("the first reactivation", rt.admit(first))?;
    let verdict = cx.verdict(
        "colliding_result_race",
        "a second reactivation of the same target",
        rt.admit(second),
        is_where(|e| matches!(e, ExpertAdmissionError::StaleOperand { .. })),
    )?;
    cx.count_in(
        "colliding_result_race_cases",
        "Reactivate_retired_target",
        1,
    );
    cx.count("duplicate_result_commits", verdict.violation());
    Ok(())
}

/// The controls: a result may take over a pod or slot of an operand of the same operation.
fn takeovers(cx: &mut Cx) -> Res<()> {
    cx.sub("successor_takeover");
    // Reactivate of a retired target: the same pod, the same slot, the next generation.
    let mut rt = Rt::fresh()?;
    rt.birth("a")?;
    rt.retire("a", RetiredReason::GateFailed)?;
    rt.reactivate_retired("a", None)?;
    match rt.state("a") {
        Some((ExpertState::Serving, _)) if rt.routable("a", 2) => {}
        other => return Err(format!("the reactivated successor is {other:?}")),
    }
    cx.count_in(
        "successor_takeover_controls_accepted",
        "Reactivate_same_pod",
        1,
    );
    // A Merge result and a Split result on the slot of one of their sources.
    let mut rt = Rt::fresh()?;
    rt.birth("a")?;
    rt.birth("b")?;
    let result = spec_at(&rt.project, "m", 1, level_of("a"));
    rt.merge(&["a", "b"], result, None)?;
    if !rt.routable("m", 1) {
        return Err("the merge result on a source slot is not routable".to_string());
    }
    cx.count_in(
        "successor_takeover_controls_accepted",
        "Split_Merge_slot_reuse",
        1,
    );
    let mut rt = Rt::fresh()?;
    rt.birth("a")?;
    let results = vec![spec_at(&rt.project, "x", 1, level_of("a")), rt.spec("y")];
    rt.split("a", results, None)?;
    if !rt.routable("x", 1) {
        return Err("the split result on the source slot is not routable".to_string());
    }
    cx.count_in(
        "successor_takeover_controls_accepted",
        "Split_Merge_slot_reuse",
        1,
    );
    Ok(())
}

/// A planned succession (an operation retiring its own sources) does not fence its result,
/// with and without a dependency block of the result's own; an independent later invalidation
/// of an inherited input still propagates.
fn planned_succession(cx: &mut Cx) -> Res<()> {
    cx.sub("planned_succession");
    let on_train = || inputs(&[(TRAIN, 1, D1)], &[]);
    let mut successors: Vec<(&'static str, Rt, &'static str, u64)> = Vec::new();

    // Merge: both sources rest on TRAIN; the result names the same input.
    let mut rt = Rt::fresh()?;
    rt.birth_with("a", on_train())?;
    rt.birth_with("b", on_train())?;
    let result = rt.spec("m");
    rt.merge(&["a", "b"], result, Some(on_train()))?;
    successors.push(("Merge", rt, "m", 1));

    // Split: the result names no block of its own and inherits the source's lineage.
    let mut rt = Rt::fresh()?;
    rt.birth_with("a", on_train())?;
    let results = vec![rt.spec("x"), rt.spec("y")];
    rt.split("a", results, None)?;
    successors.push(("Split", rt, "x", 1));

    // Reactivate of a retired target: the next generation rests on the same input.
    let mut rt = Rt::fresh()?;
    rt.birth_with("a", on_train())?;
    rt.retire("a", RetiredReason::GateFailed)?;
    rt.reactivate_retired("a", Some(on_train()))?;
    successors.push(("Reactivate_retired_target", rt, "a", 2));

    for (kind, mut rt, pod, generation) in successors {
        cx.count_in("planned_succession_cases", kind, 1);
        let routable = rt.routable(pod, generation);
        let fenced = matches!(rt.state(pod), Some((ExpertState::Fenced, _)));
        cx.attempt("planned_succession_routing_control")?;
        cx.count("self_fenced_results", i64::from(fenced));
        cx.count("planned_successor_routes_rejected", i64::from(!routable));
        if routable {
            cx.count_in("planned_succession_routing_controls", kind, 1);
        }
        // The independent invalidation of the inherited input still reaches the result.
        let outcome = rt
            .invalidate(InputInvalidation::revoked(TRAIN, Generation(1), D1))
            .map_err(|error| format!("{kind}: revoking the inherited input: {error:?}"))?;
        cx.note("planned_succession_propagation_checks");
        let reached = outcome.fenced().iter().any(|member| member.pod.0 == pod);
        cx.count(
            "derived_routable_after_transitive_invalidation",
            i64::from(!reached || rt.routable(pod, generation)),
        );
    }
    Ok(())
}
