//! C10: the compound `InputInvalidated` event, the dependency index and the fence set.
//!
//! Revocation or supersession of a generation-bearing input is one ledger event that derives
//! and applies the fence set atomically. The cases revoke and supersede training and evaluation
//! inputs, the base model, an adapter and the input of a source expert, after the gate pass and
//! after promotion; race a revocation of generation `g` with a replacement `g+1`; fan one
//! input out to more experts than an explicit list would fit in a record; crash between the
//! append and anything else; refuse a tampered fence set on replay; and run the normal, lag
//! and restore paths of the invalidation matrix.
//!
//! The derived dependency set of every scenario comes from the harness's own bookkeeping of
//! the topology it built (which expert rests on which input, which result descends from which
//! source), not from the runtime's index: `fenced_and_probed_dependents` is what the runtime
//! fenced and refused to route, `derived_dependency_set_*` is what the topology says must be.

use super::fixture::{
    base_model_input, inputs, report_for, ReportSpec, Rt, Setup, ADAPTER, D1, D2, D3, EVAL, OTHER,
    REPLACEMENT, TRAIN,
};
use super::{
    classify, classify_route, control, is_where, probe_bystanders, probe_routes, route_control, Cx,
    FenceDiff, RouteExpect, RouteTally,
};
use crate::experiments::layered_kv::emit::Json;
use crate::experiments::layered_kv::lab::{fail, Res};
use ptr_ledger::{ExpertFenceAction, ExpertFenceResume, InputInvalidationKind, LedgerEvent};
use ptr_pods::{ExpertInputs, ExpertState, StructuralOp};
use ptr_runtime::expert::{
    expert_fence_set_digest, ExpertAdmissionError, ExpertFenceSetMember, InputInvalidation,
    InputInvalidationOutcome, RouteRefusal,
};
use ptr_runtime::RuntimeError;
use ptr_types::{Generation, PodId, Validity};
use std::collections::BTreeSet;

pub fn c10_compound_invalidation(cx: &mut Cx) -> Res<()> {
    window_before_the_fence(cx)?;
    compound_event_and_crash(cx)?;
    replay_refuses_a_tampered_set(cx)?;
    stale_commits(cx)?;
    wrong_generation_race(cx)?;
    transitive_and_lineage(cx)?;
    supersession_atomicity(cx)?;
    stale_structural_commits(cx)?;
    large_fanout(cx)?;
    matrix(cx)?;
    Ok(())
}

// ---- the scenario bookkeeping ----------------------------------------------------------------

/// What the harness knows about the experts of a scenario: the topology it built.
struct Topology {
    scenario: String,
    /// The experts that must end up fenced, by the harness's own derivation.
    derived: BTreeSet<String>,
    /// The experts that rest on nothing invalidated and must stay routable.
    bystanders: BTreeSet<String>,
}

impl Topology {
    fn new(scenario: &str, derived: &[&str], bystanders: &[&str]) -> Self {
        Self {
            scenario: scenario.to_string(),
            derived: derived.iter().map(|pod| (*pod).to_string()).collect(),
            bystanders: bystanders.iter().map(|pod| (*pod).to_string()).collect(),
        }
    }
}

/// `pod` is `Fenced` in the registry and the route probe refuses it as fenced.
fn fenced_and_refused(rt: &Rt, pod: &str) -> bool {
    let generation = rt.init(pod).map(|init| init.generation.0).unwrap_or(1);
    matches!(rt.state(pod), Some((ExpertState::Fenced, _)))
        && matches!(rt.route(pod, generation), Err(RouteRefusal::Fenced))
}

/// Fill the set metrics from every scenario of the run (C10 and C12 add to the sets).
pub fn finish_sets(cx: &mut Cx) {
    // The set the runtime fenced and the set the topology derives, in every scenario and in
    // the three probes of it, must be the same identities.
    let fenced = cx
        .sets
        .get("fenced_and_probed")
        .cloned()
        .unwrap_or_default();
    let derived = cx
        .sets
        .get("derived_dependents")
        .cloned()
        .unwrap_or_default();
    cx.tally
        .set("fenced_and_probed_dependents", fenced.len() as i64);
    cx.tally
        .set("derived_dependency_set_size", derived.len() as i64);
    let digest = |set: &BTreeSet<String>| {
        crate::experiments::layered_kv::lab::id_set_digest(set.iter().map(String::as_str))
    };
    cx.tally.set_json(
        "fenced_and_probed_dependents_set_digest",
        Json::str(digest(&fenced)),
    );
    cx.tally
        .set_json("derived_dependency_set_digest", Json::str(digest(&derived)));
}

/// The name of the set of experts the probe `at` of `scenario` saw fenced.
fn probe_set(scenario: &str, at: &str) -> String {
    format!("fenced_dependents@{scenario}@{at}")
}

/// Open the set of probe `at` of `scenario` before anything is added to it: a probe that sees
/// no member still takes part in the intersection `fold_scenario` takes, so a fence that is
/// missing at that probe shows up as a set mismatch instead of the probe dropping out.
fn open_probe(cx: &mut Cx, scenario: &str, at: &str) {
    cx.sets.entry(probe_set(scenario, at)).or_default();
}

/// Record the fenced set of `scenario` at probe `at` (`now`, `reopen`, `replay`, `restore`).
fn probe_fenced(cx: &mut Cx, topology: &Topology, rt: &Rt, universe: &[&str], at: &str) {
    open_probe(cx, &topology.scenario, at);
    for pod in &topology.derived {
        cx.set_add(
            "derived_dependents",
            &format!("{}/{pod}", topology.scenario),
        );
    }
    for pod in universe {
        if fenced_and_refused(rt, pod) {
            cx.set_add(
                &probe_set(&topology.scenario, at),
                &format!("{}/{pod}", topology.scenario),
            );
        }
    }
}

/// Fold the probes of one scenario into the global fenced set: the identities every probe saw
/// fenced. The probes are kept apart by scenario.
fn fold_scenario(cx: &mut Cx, scenario: &str) {
    let prefix = format!("fenced_dependents@{scenario}@");
    let probes: Vec<(String, BTreeSet<String>)> = cx
        .sets
        .iter()
        .filter(|(name, _)| name.starts_with(&prefix))
        .map(|(name, set)| (name.clone(), set.clone()))
        .collect();
    let mut all: Option<BTreeSet<String>> = None;
    for (name, set) in &probes {
        all = Some(match all {
            None => set.clone(),
            Some(previous) => previous.intersection(set).cloned().collect(),
        });
        cx.sets.remove(name);
    }
    if let Some(set) = all {
        cx.sets
            .entry("fenced_and_probed".to_string())
            .or_default()
            .extend(set);
    }
}

/// The three probes of a scenario after the event: now, after the runtime was dropped and the
/// ledger reopened, and after a replay of the history. Returns what the probes found over all
/// three: dependents still routable after the event, dependents refused without a fence and
/// bystanders closed. Every refusal is kind-checked: a dependent must be refused as `Fenced`.
fn probe_all(cx: &mut Cx, topology: &Topology, rt: &mut Rt, universe: &[&str]) -> Res<RouteTally> {
    let dependents: Vec<&str> = topology.derived.iter().map(String::as_str).collect();
    let bystanders: Vec<&str> = topology.bystanders.iter().map(String::as_str).collect();
    probe_fenced(cx, topology, rt, universe, "now");
    let mut found = probe_routes(rt, &dependents, RouteExpect::Fenced)?;
    found.merge(probe_bystanders(rt, &bystanders)?);
    rt.reopen()?;
    probe_fenced(cx, topology, rt, universe, "reopen");
    found.merge(probe_routes(rt, &dependents, RouteExpect::Fenced)?);
    found.merge(probe_bystanders(rt, &bystanders)?);
    let replayed = rt.replayed()?;
    probe_fenced_runtime(cx, topology, rt, &replayed, universe, "replay");
    for pod in &dependents {
        let generation = rt.init(pod).map(|init| init.generation.0).unwrap_or(1);
        found.add(classify_route(
            pod,
            replayed.expert_route_admission(
                &rt.project,
                &PodId::from(*pod),
                Generation(generation),
            ),
            RouteExpect::Fenced,
        )?);
    }
    for pod in &bystanders {
        let generation = rt.init(pod).map(|init| init.generation.0).unwrap_or(1);
        found.overfenced += i64::from(super::classify_bystander(
            pod,
            replayed.expert_route_admission(
                &rt.project,
                &PodId::from(*pod),
                Generation(generation),
            ),
        )?);
    }
    fold_scenario(cx, &topology.scenario);
    Ok(found)
}

/// [`probe_fenced`] for a runtime rebuilt by replay.
fn probe_fenced_runtime(
    cx: &mut Cx,
    topology: &Topology,
    rt: &Rt,
    replayed: &ptr_runtime::PtrRuntime,
    universe: &[&str],
    at: &str,
) {
    let set = probe_set(&topology.scenario, at);
    open_probe(cx, &topology.scenario, at);
    for pod in universe {
        let entry = replayed
            .expert_registry()
            .get(&rt.project, &PodId::from(*pod));
        let Some(entry) = entry else { continue };
        let refused = matches!(
            replayed.expert_route_admission(&rt.project, &PodId::from(*pod), entry.generation),
            Err(RouteRefusal::Fenced)
        );
        if entry.state == ExpertState::Fenced && refused {
            cx.set_add(&set, &format!("{}/{pod}", topology.scenario));
        }
    }
}

// ---- 1. the window between a plain revocation and the fence ------------------------------

/// A plain `Revoked` record without a fence leaves the registry state `Serving`; the route
/// probe still closes it for an expert that recorded its inputs, also across a crash before
/// the fence. (An expert admitted without a dependency block is outside the mechanism.)
fn window_before_the_fence(cx: &mut Cx) -> Res<()> {
    cx.sub("revoked_training_input_route");
    let mut rt = Rt::fresh()?;
    rt.birth_with("x", inputs(&[(TRAIN, 1, D1)], &[]))?;
    rt.birth_with("y", inputs(&[(TRAIN, 1, D1)], &[(EVAL, 1, D2)]))?;
    rt.birth_with("z", inputs(&[(OTHER, 1, D3)], &[]))?;
    // The live controls: before the revocation every expert is routable.
    for pod in ["x", "y", "z"] {
        route_control(&format!("{pod} before the revocation"), rt.route(pod, 1))?;
    }
    rt.revoke(TRAIN, 1)?;
    // The registry still says Serving (no fence was written) ...
    let unfenced = matches!(rt.state("x"), Some((ExpertState::Serving, _)));
    cx.tally.set(
        "internal_plain_revocation_leaves_state_serving",
        i64::from(unfenced),
    );
    // ... but the use-time admission refuses the dependents as `DependencyNotLive` (any other
    // refusal is a harness error), and only them.
    let found = probe_routes(&rt, &["x", "y"], RouteExpect::DependencyNotLive)?;
    cx.attempt("revoked_training_input_route")?;
    cx.count("expert_routes_after_revocation_before_fence", found.served);
    cx.count("routes_to_revoked_dependency_experts", found.served);
    cx.count("revoked_expert_served_count", found.served);
    let bystander = probe_bystanders(&rt, &["z"])?;
    cx.count_fence_findings(&FenceDiff::default(), &bystander);
    // The crash before the fence: the process dies after the append.
    cx.sub("fault_between_revocation_and_fence");
    rt.reopen()?;
    let found = probe_routes(&rt, &["x", "y"], RouteExpect::DependencyNotLive)?;
    cx.attempt("fault_between_revocation_and_fence")?;
    cx.count("expert_routes_after_revocation_before_fence", found.served);
    cx.count("routes_to_revoked_dependency_experts", found.served);
    cx.count("revoked_expert_served_count", found.served);
    // The bystander again after the reopen.
    let bystander = probe_bystanders(&rt, &["z"])?;
    cx.count_fence_findings(&FenceDiff::default(), &bystander);
    Ok(())
}

// ---- 2. the compound event and the crash after its append ----------------------------------

fn members(rt: &Rt, pods: &[&str]) -> Res<Vec<ExpertFenceSetMember>> {
    // The set the harness derived independently: every member serving at generation 1 and
    // state version 1, fenced from Serving, in canonical (ascending) order.
    let mut sorted: Vec<&str> = pods.to_vec();
    sorted.sort_unstable();
    sorted
        .into_iter()
        .map(|pod| {
            Ok(ExpertFenceSetMember {
                project: rt.project.clone(),
                pod: PodId::from(pod),
                generation: Generation(1),
                expected_state_version: 1,
                action: ExpertFenceAction::Fence,
                resume: ExpertFenceResume::Serving,
            })
        })
        .collect()
}

fn invalidation_event(rt: &Rt) -> Res<(LedgerEvent, u32, [u8; 32])> {
    match rt
        .runtime
        .committed_events()
        .last()
        .map(|c| c.event.clone())
    {
        Some(
            event @ LedgerEvent::InputInvalidated {
                fence_count,
                fence_digest,
                ..
            },
        ) => Ok((event, fence_count, fence_digest)),
        other => Err(format!("the last event is not an invalidation: {other:?}")),
    }
}

fn compound_event_and_crash(cx: &mut Cx) -> Res<()> {
    cx.sub("crash_between_append_and_apply");
    let mut rt = Rt::fresh()?;
    rt.birth_with("x", inputs(&[(TRAIN, 1, D1)], &[]))?;
    rt.birth_with("y", inputs(&[(TRAIN, 1, D1)], &[(EVAL, 1, D2)]))?;
    rt.birth_with("z", inputs(&[(OTHER, 1, D3)], &[]))?;
    let before = rt.event_count();
    let outcome = rt
        .invalidate(InputInvalidation::revoked(TRAIN, Generation(1), D1))
        .map_err(fail("revoking a training input"))?;
    if rt.event_count() != before + 1 {
        return Err("the compound invalidation was not one event".to_string());
    }
    // The count and digest of the event are the independently derived ones.
    let (_, count, digest) = invalidation_event(&rt)?;
    let independent = members(&rt, &["x", "y"])?;
    cx.count(
        "fence_set_digest_mismatches",
        i64::from(
            count != 2
                || digest != expert_fence_set_digest(TRAIN, Generation(1), &independent)
                || outcome.fence_digest() != &digest,
        ),
    );
    // The fence set the event applied is the one the harness derived, as sets.
    let diff = FenceDiff::of(
        ["x", "y"],
        outcome.fenced().iter().map(|member| member.pod.0.as_str()),
    );
    cx.count_fence_findings(&diff, &RouteTally::default());
    // The crash after the append: the runtime is dropped before anything else and reopened;
    // `open_durable` derives the fence set again, so the dependents are `Fenced`.
    rt.reopen()?;
    let found = probe_routes(&rt, &["x", "y"], RouteExpect::Fenced)?;
    cx.attempt("crash_between_append_and_apply")?;
    cx.count(
        "expert_routes_after_crash_between_append_and_apply",
        found.served,
    );
    cx.count("revoked_expert_served_count", found.served);
    let bystander = probe_bystanders(&rt, &["z"])?;
    cx.count_fence_findings(&FenceDiff::default(), &found);
    cx.count_fence_findings(&FenceDiff::default(), &bystander);
    Ok(())
}

// ---- 3. replay refuses a tampered fence set ---------------------------------------------------

fn replay_refuses_a_tampered_set(cx: &mut Cx) -> Res<()> {
    cx.sub("replay_fence_set_mismatch");
    let mut rt = Rt::fresh()?;
    rt.birth_with("x", inputs(&[(TRAIN, 1, D1)], &[]))?;
    rt.birth_with("y", inputs(&[(TRAIN, 1, D1)], &[]))?;
    rt.invalidate(InputInvalidation::revoked(TRAIN, Generation(1), D1))
        .map_err(fail("revoking a training input"))?;
    // The honest history replays.
    control("an honest replay", Rt::replay_of(&rt.events()).map(|_| ()))?;
    let last = rt.event_count() - 1;
    for (what, tamper) in [
        ("a count one too low", Tamper::Count),
        ("a flipped digest", Tamper::Digest),
    ] {
        let mut events = rt.events();
        if let LedgerEvent::InputInvalidated {
            fence_count,
            fence_digest,
            ..
        } = &mut events[last].event
        {
            match tamper {
                Tamper::Count => *fence_count -= 1,
                Tamper::Digest => fence_digest[0] ^= 0xFF,
            }
        }
        let verdict = cx.verdict(
            "replay_fence_set_mismatch",
            what,
            Rt::replay_of(&events).map(|_| ()),
            is_where(|e| matches!(e, ExpertAdmissionError::FenceSetMismatch { .. })),
        )?;
        cx.count("replay_fence_set_mismatch_served", verdict.violation());
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Tamper {
    Count,
    Digest,
}

// ---- 4. stale input commits ---------------------------------------------------------------

/// The identity the lifecycle authority knows as a committed capsule, at generation 1.
fn commit_capsule(rt: &mut Rt, identity: &str) -> Res<()> {
    rt.runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: rt.project.clone(),
            capsule: ptr_types::CapsuleId::from(identity),
            generation: Generation(1),
        })
        .map(|_| ())
        .map_err(fail("committing a capsule"))
}

fn supersede_capsule(rt: &mut Rt, identity: &str) -> Res<()> {
    rt.runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: ptr_types::CapsuleId::from(identity),
            old: Generation(1),
            new: Generation(2),
        })
        .map(|_| ())
        .map_err(fail("superseding a capsule"))
}

fn birth_naming(
    rt: &mut Rt,
    pod: &str,
    training: &[(&str, u64, [u8; 32])],
    evaluation: &[(&str, u64, [u8; 32])],
) -> Result<ptr_types::CommitIndex, RuntimeError> {
    let op = StructuralOp::Birth {
        result: rt.spec(pod),
    };
    rt.admit_as(
        op,
        Some(inputs(training, evaluation)),
        &ReportSpec::correct(),
        Setup::PRE,
    )
}

fn stale_commits(cx: &mut Cx) -> Res<()> {
    // The input is NOT revoked, but the recorded generation is stale: the lifecycle authority
    // superseded it. Training input.
    cx.sub("stale_input_generation_commit");
    let mut rt = Rt::fresh()?;
    commit_capsule(&mut rt, TRAIN)?;
    supersede_capsule(&mut rt, TRAIN)?;
    if rt.runtime.generation_validity(TRAIN, Generation(1)) != Some(Validity::Superseded) {
        return Err("the training capsule is not superseded".to_string());
    }
    let verdict = cx.verdict(
        "stale_input_generation_commit",
        "a birth on a superseded training generation",
        birth_naming(&mut rt, "x", &[(TRAIN, 1, D1)], &[]),
        is_where(|e| matches!(e, ExpertAdmissionError::InputSuperseded { .. })),
    )?;
    cx.count("stale_input_generation_commit_attempts", 1);
    cx.count(
        "stale_input_generation_commits_admitted",
        verdict.violation(),
    );
    control(
        "a birth on the live generation",
        birth_naming(&mut rt, "x", &[(TRAIN, 2, D2)], &[]),
    )?;
    cx.count("stale_input_commit_controls_accepted", 1);

    // The recorded digest is stale relative to the digest the first committed use fixed.
    cx.sub("stale_input_digest_commit");
    let mut rt = Rt::fresh()?;
    control(
        "the first use of the input",
        birth_naming(&mut rt, "x", &[(TRAIN, 1, D1)], &[]),
    )?;
    let verdict = cx.verdict(
        "stale_input_digest_commit",
        "a birth on another digest of the same generation",
        birth_naming(&mut rt, "y", &[(TRAIN, 1, D2)], &[]),
        is_where(|e| matches!(e, ExpertAdmissionError::InputDigestStale { .. })),
    )?;
    cx.count("stale_input_digest_commit_attempts", 1);
    cx.count("stale_input_digest_commits_admitted", verdict.violation());
    control(
        "a second use on the fixed digest",
        birth_naming(&mut rt, "y", &[(TRAIN, 1, D1)], &[]),
    )?;
    cx.count("stale_input_commit_controls_accepted", 1);

    // The evaluation input: a stale generation, a stale digest, and a supersession by the
    // compound event (the input is not revoked).
    cx.sub("stale_eval_input_commit");
    let mut rt = Rt::fresh()?;
    commit_capsule(&mut rt, EVAL)?;
    supersede_capsule(&mut rt, EVAL)?;
    let verdict = cx.verdict(
        "stale_eval_input_commit",
        "a birth on a superseded evaluation generation",
        birth_naming(&mut rt, "x", &[(OTHER, 1, D3)], &[(EVAL, 1, D2)]),
        is_where(|e| matches!(e, ExpertAdmissionError::InputSuperseded { .. })),
    )?;
    cx.count("stale_eval_input_commit_attempts", 1);
    cx.count("stale_eval_input_commits_admitted", verdict.violation());
    let mut rt = Rt::fresh()?;
    control(
        "the first use of the evaluation input",
        birth_naming(&mut rt, "x", &[(OTHER, 1, D3)], &[(EVAL, 1, D2)]),
    )?;
    let verdict = cx.verdict(
        "stale_eval_input_commit",
        "a birth on another digest of the evaluation generation",
        birth_naming(&mut rt, "y", &[(OTHER, 1, D3)], &[(EVAL, 1, D1)]),
        is_where(|e| matches!(e, ExpertAdmissionError::InputDigestStale { .. })),
    )?;
    cx.count("stale_eval_input_commit_attempts", 1);
    cx.count("stale_eval_input_commits_admitted", verdict.violation());

    cx.sub("stale_eval_input_supersession_commit");
    let mut rt = Rt::fresh()?;
    birth_naming(&mut rt, "x", &[(OTHER, 1, D3)], &[(EVAL, 1, D2)])
        .map_err(|error| format!("the first use of the evaluation input: {error:?}"))?;
    rt.invalidate(InputInvalidation::superseded(
        EVAL,
        Generation(1),
        D2,
        Generation(2),
        D1,
        REPLACEMENT,
    ))
    .map_err(fail("superseding the evaluation input"))?;
    if rt
        .runtime
        .expert_registry()
        .dependencies()
        .is_revoked(EVAL, Generation(1))
    {
        return Err("a supersession revoked the evaluation input".to_string());
    }
    let verdict = cx.verdict(
        "stale_eval_input_supersession_commit",
        "a birth on a superseded evaluation generation",
        birth_naming(&mut rt, "y", &[(OTHER, 1, D3)], &[(EVAL, 1, D2)]),
        is_where(|e| matches!(e, ExpertAdmissionError::InputSuperseded { .. })),
    )?;
    cx.count("stale_eval_input_supersession_commit_attempts", 1);
    cx.count(
        "stale_eval_input_supersession_commits_admitted",
        verdict.violation(),
    );
    // The valid control for the superseded evaluation input: the new generation commits.
    control(
        "a birth on the replacement evaluation generation",
        birth_naming(&mut rt, "y", &[(OTHER, 1, D3)], &[(EVAL, 2, D1)]),
    )?;
    cx.count("stale_input_commit_controls_accepted", 1);

    // A revocation between the gate pass and the commit, per role: the report was produced for
    // the block, the input is revoked, the commit is refused.
    cx.sub("input_revoked_after_gate_pass_before_commit");
    let roles: [(&str, &str, u64, [u8; 32]); 3] = [
        ("training", TRAIN, 1, D1),
        ("evaluation", EVAL, 1, D2),
        (
            "base_model",
            "base-model:base-a",
            1,
            super::fixture::BASE_DIGEST,
        ),
    ];
    for (role, identity, generation, digest) in roles {
        let mut rt = Rt::fresh()?;
        let block = inputs(&[(TRAIN, 1, D1)], &[(EVAL, 1, D2)]);
        let proposal = rt.proposal(StructuralOp::Birth {
            result: rt.spec("n"),
        })?;
        let report = report_for(&proposal, Some(&block), &ReportSpec::correct())?;
        let grant = rt.admission_grant(Setup::PRE)?;
        rt.invalidate(InputInvalidation::revoked(
            identity,
            Generation(generation),
            digest,
        ))
        .map_err(|error| format!("revoking the {role}: {error:?}"))?;
        let verdict = cx.verdict(
            "input_revoked_after_gate_pass_before_commit",
            &format!("a commit after the {role} input was revoked"),
            rt.commit_proposal(&proposal, &report, &grant, Some(block)),
            is_where(|e| matches!(e, ExpertAdmissionError::InputRevoked { .. })),
        )?;
        cx.count("commits_with_revoked_input_accepted", verdict.violation());
    }
    Ok(())
}

// ---- 5. the cells: kinds x types x paths --------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Training,
    Evaluation,
    BaseModel,
    Adapter,
    SourceExpert,
}

impl Kind {
    const ALL: [Kind; 5] = [
        Self::Training,
        Self::Evaluation,
        Self::BaseModel,
        Self::Adapter,
        Self::SourceExpert,
    ];

    /// The kind as the matrix cells name it.
    fn cell(self) -> &'static str {
        match self {
            Self::Training => "training_input",
            Self::Evaluation => "evaluation_input",
            Self::BaseModel => "base_model",
            Self::Adapter => "adapter",
            Self::SourceExpert => "source_expert",
        }
    }

    /// The input the invalidation names: identity, generation and digest.
    fn input(self) -> (String, u64, [u8; 32]) {
        match self {
            Self::Training | Self::SourceExpert => (TRAIN.to_string(), 1, D1),
            Self::Evaluation => (EVAL.to_string(), 1, D2),
            Self::BaseModel => {
                let input = base_model_input();
                (
                    input.identity().to_string(),
                    input.generation().0,
                    *input.digest(),
                )
            }
            Self::Adapter => (ADAPTER.to_string(), 1, D2),
        }
    }

    /// The dependency block of an expert that rests on the kind's input (the base model is
    /// derived from the record, so its block names something else).
    fn block(self) -> ExpertInputs {
        match self {
            Self::Training | Self::SourceExpert => inputs(&[(TRAIN, 1, D1)], &[]),
            Self::Evaluation => inputs(&[(OTHER, 1, D3)], &[(EVAL, 1, D2)]),
            Self::BaseModel => inputs(&[(OTHER, 1, D3)], &[]),
            Self::Adapter => inputs(&[(ADAPTER, 1, D2)], &[]),
        }
    }

    /// The same block, for a probe that names the dead input.
    fn probe_block(self) -> ExpertInputs {
        self.block()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Type {
    Revocation,
    Supersession,
}

impl Type {
    fn name(self) -> &'static str {
        match self {
            Self::Revocation => "revocation",
            Self::Supersession => "supersession",
        }
    }

    fn request(self, kind: Kind) -> InputInvalidation {
        let (identity, generation, digest) = kind.input();
        match self {
            Self::Revocation => {
                InputInvalidation::revoked(identity, Generation(generation), digest)
            }
            Self::Supersession => InputInvalidation::superseded(
                identity,
                Generation(generation),
                digest,
                Generation(generation + 1),
                D3,
                REPLACEMENT,
            ),
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
    fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Lag => "lag",
            Self::Restore => "restore",
        }
    }
}

/// The experts a cell builds: the dependents the invalidation must fence and the bystanders it
/// must leave routable.
struct Cell {
    rt: Rt,
    dependents: Vec<&'static str>,
    bystanders: Vec<&'static str>,
    universe: Vec<&'static str>,
}

fn build_cell(kind: Kind) -> Res<Cell> {
    let mut rt = Rt::fresh()?;
    let (dependents, bystanders): (Vec<&'static str>, Vec<&'static str>) = match kind {
        Kind::SourceExpert => {
            // The dependent rests on the revoked input only through its source expert.
            rt.birth_with("s", Kind::SourceExpert.block())?;
            rt.birth_with("t", inputs(&[(OTHER, 1, D3)], &[]))?;
            let result = rt.spec("m");
            rt.merge(&["s", "t"], result, None)?;
            rt.birth_with("w", inputs(&[(OTHER, 1, D3)], &[]))?;
            (vec!["m"], vec!["w"])
        }
        Kind::BaseModel => {
            rt.birth_with("x", kind.block())?;
            // An expert admitted without a dependency block records no base model.
            rt.birth("w")?;
            (vec!["x"], vec!["w"])
        }
        _ => {
            rt.birth_with("x", kind.block())?;
            rt.birth_with("w", inputs(&[(OTHER, 1, D3)], &[]))?;
            (vec!["x"], vec!["w"])
        }
    };
    let mut universe = dependents.clone();
    universe.extend(&bystanders);
    if kind == Kind::SourceExpert {
        universe.extend(["s", "t"]);
    }
    Ok(Cell {
        rt,
        dependents,
        bystanders,
        universe,
    })
}

/// What the probes of a cell found.
struct Probed {
    /// Dependents still routable after the event.
    routable: i64,
    /// Dependents refused without a fence, and bystanders closed (a fence-set mismatch).
    tally: RouteTally,
    /// Admissions naming the dead input that the runtime accepted.
    admitted: i64,
}

/// Probe the dependents and the dead input on `rt` as it stands now. A dependent must be
/// refused as `Fenced` and a bystander must stay routable; a fence missing or one too many is a
/// finding of the cell (counted by the caller), never a harness error. A refusal of any other
/// kind is a harness error.
fn probe_cell(cell: &mut Cell, kind: Kind, ty: Type) -> Res<Probed> {
    let mut tally = probe_routes(&cell.rt, &cell.dependents, RouteExpect::Fenced)?;
    let routable = tally.served;
    tally.merge(probe_bystanders(&cell.rt, &cell.bystanders)?);
    // A record naming the dead input is refused.
    let pod = "n";
    let result = birth_with_block(&mut cell.rt, pod, kind.probe_block());
    let verdict = match ty {
        Type::Revocation => classify(
            "a record naming the revoked input",
            result,
            is_where(|e| matches!(e, ExpertAdmissionError::InputRevoked { .. })),
        )?,
        Type::Supersession => classify(
            "a record naming the superseded input",
            result,
            is_where(|e| matches!(e, ExpertAdmissionError::InputSuperseded { .. })),
        )?,
    };
    Ok(Probed {
        routable,
        tally,
        admitted: verdict.violation(),
    })
}

fn birth_with_block(
    rt: &mut Rt,
    pod: &str,
    block: ExpertInputs,
) -> Result<ptr_types::CommitIndex, RuntimeError> {
    let op = StructuralOp::Birth {
        result: rt.spec(pod),
    };
    rt.admit_as(op, Some(block), &ReportSpec::correct(), Setup::PRE)
}

/// Run one cell of the matrix on `path`; the hits are the dependents the runtime still routed
/// plus the dead-input records it accepted.
fn run_cell(cx: &mut Cx, kind: Kind, ty: Type, path: Path) -> Res<Probed> {
    let mut cell = build_cell(kind)?;
    let topology = Topology::new(
        &format!("{}_{}_{}", kind.cell(), path.name(), ty.name()),
        &cell.dependents.clone(),
        &cell.bystanders.clone(),
    );
    let request = ty.request(kind);
    let outcome = cell
        .rt
        .invalidate(request)
        .map_err(|error| format!("{}: {error:?}", topology.scenario))?;
    // The experts the event fenced against the experts the topology says it owes: a missed or
    // an extra expert is a finding of the cell (a stale hit and a fence-set mismatch), and the
    // cell goes on to its probes.
    let diff = FenceDiff::of(
        cell.dependents.iter().copied(),
        outcome.fenced().iter().map(|member| member.pod.0.as_str()),
    );
    match path {
        Path::Normal => {}
        // The runtime dropped after the append and reopened: `open_durable` derives the fence
        // set again.
        Path::Lag => cell.rt.reopen()?,
        // A restart from a compacted snapshot after the invalidation was fully applied.
        Path::Restore => cell.rt.compact_restore()?,
    }
    let probed = probe_cell(&mut cell, kind, ty)?;
    let universe = cell.universe.clone();
    probe_fenced(cx, &topology, &cell.rt, &universe, "path");
    fold_scenario(cx, &topology.scenario);
    let name = format!("{}_{}_{}", kind.cell(), path.name(), ty.name());
    let fence_findings = cx.count_fence_findings(&diff, &probed.tally);
    cx.count_in("invalidation_matrix_cases", &name, 1);
    cx.count_in(
        "invalidation_matrix_stale_hits",
        &name,
        probed.routable + probed.admitted + fence_findings,
    );
    Ok(probed)
}

/// The cells the matrix names for the kinds this case builds, on the three paths; the first
/// run of every (kind, type) is the post-promotion case the named metrics count.
fn matrix(cx: &mut Cx) -> Res<()> {
    for kind in Kind::ALL {
        let types: &[Type] = if kind == Kind::SourceExpert {
            &[Type::Revocation]
        } else {
            &[Type::Revocation, Type::Supersession]
        };
        for &ty in types {
            for path in [Path::Normal, Path::Lag, Path::Restore] {
                cx.sub(&format!("{}_{}_{}", kind.cell(), path.name(), ty.name()));
                let probed = run_cell(cx, kind, ty, path)?;
                let routed = probed.routable + probed.admitted;
                if path == Path::Normal {
                    named_counters(cx, kind, ty, routed)?;
                }
            }
        }
    }
    Ok(())
}

/// The metrics of the post-promotion and structural cases, counted on the normal path.
fn named_counters(cx: &mut Cx, kind: Kind, ty: Type, routed: i64) -> Res<()> {
    match (kind, ty) {
        (Kind::Training, Type::Revocation) => {
            cx.attempt("post_promotion_training_revocation")?;
            cx.count("post_promotion_training_revocation_cases", 1);
            cx.count("experts_serving_after_revocation", routed);
        }
        (Kind::Evaluation, Type::Revocation) => {
            cx.attempt("post_promotion_eval_revocation")?;
            cx.count("post_promotion_eval_revocation_cases", 1);
            cx.count("routable_after_eval_revocation", routed);
            cx.count("experts_serving_after_revocation", routed);
        }
        (Kind::Training | Kind::Evaluation, Type::Supersession) => {
            let key = if kind == Kind::Training {
                "training"
            } else {
                "evaluation"
            };
            cx.attempt("post_promotion_supersession")?;
            cx.count_in("post_promotion_supersession_cases", key, 1);
            cx.count("routable_after_supersession", routed);
        }
        (Kind::BaseModel | Kind::Adapter | Kind::SourceExpert, _) => {
            let case = match kind {
                Kind::BaseModel => "structural_base_model_invalidation",
                Kind::Adapter => "structural_adapter_invalidation",
                _ => "structural_source_expert_invalidation",
            };
            cx.attempt(case)?;
            cx.count_in(
                "structural_dependency_invalidation_cases",
                &format!("{}.{}", kind_key(kind), ty.name()),
                1,
            );
            cx.count("routable_after_structural_invalidation", routed);
        }
    }
    Ok(())
}

fn kind_key(kind: Kind) -> &'static str {
    match kind {
        Kind::BaseModel => "base_model",
        Kind::Adapter => "adapter",
        Kind::SourceExpert => "source_expert",
        Kind::Training => "training",
        Kind::Evaluation => "evaluation",
    }
}

// ---- 6. a revocation of generation g racing a replacement g+1 ------------------------------

fn wrong_generation_race(cx: &mut Cx) -> Res<()> {
    cx.sub("revocation_supersession_race");
    // Order 1: the replacement g+1 commits first (fencing the dependents of g), a retrained
    // expert rests on g+1, then the revocation of g arrives.
    let mut rt = Rt::fresh()?;
    rt.birth_with("x", inputs(&[(TRAIN, 1, D1)], &[]))?;
    let outcome = rt
        .invalidate(InputInvalidation::superseded(
            TRAIN,
            Generation(1),
            D1,
            Generation(2),
            D3,
            REPLACEMENT,
        ))
        .map_err(fail("superseding the input"))?;
    wrong_generation_check(cx, &outcome, &["x"])?;
    rt.birth_with("w", inputs(&[(TRAIN, 2, D3)], &[]))?;
    let late = rt.invalidate(InputInvalidation::revoked(TRAIN, Generation(1), D1));
    cx.attempt("revocation_supersession_race")?;
    cx.count("revocation_supersession_race_cases", 1);
    // What the late revocation did is information (internal): committed or refused typed.
    cx.tally.set(
        "internal_race_late_revocation_committed",
        i64::from(late.is_ok()),
    );
    race_invariants(cx, &rt, late.as_ref(), "w")?;

    // Order 2: the revocation of g commits first, then the replacement g+1.
    let mut rt = Rt::fresh()?;
    rt.birth_with("x", inputs(&[(TRAIN, 1, D1)], &[]))?;
    let outcome = rt
        .invalidate(InputInvalidation::revoked(TRAIN, Generation(1), D1))
        .map_err(fail("revoking the input"))?;
    wrong_generation_check(cx, &outcome, &["x"])?;
    let replacement = rt.invalidate(InputInvalidation::superseded(
        TRAIN,
        Generation(1),
        D1,
        Generation(2),
        D3,
        REPLACEMENT,
    ));
    // The retrained expert on g+1 is admitted when the replacement committed.
    let retrained = rt.birth_with("w", inputs(&[(TRAIN, 2, D3)], &[]));
    cx.attempt("revocation_supersession_race")?;
    cx.count("revocation_supersession_race_cases", 1);
    cx.tally.set(
        "internal_race_late_replacement_committed",
        i64::from(replacement.is_ok()),
    );
    if let Ok(late) = &replacement {
        retrained.map_err(|error| format!("the retrained expert on g+1: {error}"))?;
        race_invariants(cx, &rt, Ok(late), "w")?;
    } else if let Err(RuntimeError::Expert(_)) = &replacement {
        // A typed refusal of the replacement: the revoked generation stays revoked and g+1
        // is not revoked by it.
        cx.count(
            "wrong_generation_revocations",
            i64::from(
                rt.runtime
                    .expert_registry()
                    .dependencies()
                    .is_revoked(TRAIN, Generation(2)),
            ),
        );
    } else {
        return Err(format!("the replacement failed untyped: {replacement:?}"));
    }
    Ok(())
}

/// The fence set of an event holds exactly the experts the harness built on the generation the
/// event names, and no other.
fn wrong_generation_check(
    cx: &mut Cx,
    outcome: &InputInvalidationOutcome,
    expected: &[&str],
) -> Res<()> {
    let fenced: BTreeSet<&str> = outcome.fenced().iter().map(|m| m.pod.0.as_str()).collect();
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    cx.count(
        "wrong_generation_revocations",
        fenced.symmetric_difference(&expected).count() as i64,
    );
    Ok(())
}

/// After the race: the generation g+1 is not revoked, the expert resting on it is routable,
/// and the late event did not fence it.
fn race_invariants(
    cx: &mut Cx,
    rt: &Rt,
    late: Result<&InputInvalidationOutcome, &RuntimeError>,
    retrained: &str,
) -> Res<()> {
    let mut wrong = 0;
    if rt
        .runtime
        .expert_registry()
        .dependencies()
        .is_revoked(TRAIN, Generation(2))
        || rt.runtime.generation_validity(TRAIN, Generation(2)) == Some(Validity::Revoked)
    {
        wrong += 1;
    }
    if !rt.routable(retrained, 1) {
        wrong += 1;
    }
    match late {
        Ok(outcome) => {
            wrong += outcome
                .fenced()
                .iter()
                .filter(|member| member.pod.0 == retrained)
                .count() as i64;
        }
        Err(RuntimeError::Expert(_)) => {}
        Err(other) => return Err(format!("the late event failed untyped: {other:?}")),
    }
    cx.count("wrong_generation_revocations", wrong);
    Ok(())
}

// ---- 7. transitive closure and reactivate lineage ------------------------------------------

fn transitive_and_lineage(cx: &mut Cx) -> Res<()> {
    // X (the input) to A to B: A rests on TRAIN; Merge(A, A2) -> B; Split(B) -> C1, C2. A and B
    // are superseded by their successors; only C1 and C2 serve, and they rest on TRAIN through
    // the chain alone.
    cx.sub("transitive_dependency_invalidation");
    let mut rt = Rt::fresh()?;
    rt.birth_with("a", inputs(&[(TRAIN, 1, D1)], &[]))?;
    rt.birth_with("e", inputs(&[(OTHER, 1, D3)], &[]))?;
    rt.birth_with("w", inputs(&[(OTHER, 1, D3)], &[]))?;
    let result = rt.spec("b");
    rt.merge(&["a", "e"], result, None)?;
    let results = vec![rt.spec("x"), rt.spec("y")];
    rt.split("b", results, None)?;
    let topology = Topology::new("transitive", &["x", "y"], &["w"]);
    rt.invalidate(InputInvalidation::revoked(TRAIN, Generation(1), D1))
        .map_err(fail("revoking the root input"))?;
    cx.attempt("transitive_dependency_invalidation")?;
    cx.count("transitive_dependency_cases", 1);
    let found = probe_all(cx, &topology, &mut rt, &["a", "e", "w", "b", "x", "y"])?;
    cx.count(
        "derived_routable_after_transitive_invalidation",
        found.served,
    );
    cx.count("experts_serving_after_revocation", found.served);
    cx.count_fence_findings(&FenceDiff::default(), &found);

    // X to g to Reactivate(g+1): the reactivated generation rests on X through its seed.
    cx.sub("reactivate_lineage");
    let mut rt = Rt::fresh()?;
    rt.birth_with("g", inputs(&[(TRAIN, 1, D1)], &[]))?;
    rt.birth_with("w", inputs(&[(OTHER, 1, D3)], &[]))?;
    rt.retire("g", ptr_pods::RetiredReason::GateFailed)?;
    rt.reactivate_retired("g", None)?;
    let topology = Topology::new("reactivate_lineage", &["g"], &["w"]);
    rt.invalidate(InputInvalidation::revoked(TRAIN, Generation(1), D1))
        .map_err(fail("revoking the root input"))?;
    cx.attempt("reactivate_lineage")?;
    cx.count("reactivate_lineage_cases", 1);
    let generation_two = rt.init("g")?.generation.0;
    if generation_two != 2 {
        return Err("the reactivated generation is not 2".to_string());
    }
    let found = probe_all(cx, &topology, &mut rt, &["g", "w"])?;
    cx.count("experts_serving_after_revocation", found.served);
    cx.count(
        "derived_routable_after_transitive_invalidation",
        found.served,
    );
    cx.count_fence_findings(&FenceDiff::default(), &found);
    Ok(())
}

// ---- 8. the replacement and the crash --------------------------------------------------------

fn supersession_atomicity(cx: &mut Cx) -> Res<()> {
    cx.sub("supersession_atomicity_crash");
    let blob = b"replacement artifact of the training input";
    let reference = ptr_ledger::integrity::sha256(blob);
    // A crash before the event: the blob is written, the event never appended.
    let mut rt = Rt::fresh()?;
    rt.birth_with("x", inputs(&[(TRAIN, 1, D1)], &[]))?;
    std::fs::write(rt.scratch("replacement.bin"), blob).map_err(fail("writing the blob"))?;
    rt.reopen()?;
    let index = rt.runtime.expert_registry().dependencies();
    let visible = index.supersession_floor(TRAIN).is_some()
        || index.registered_digest(TRAIN, Generation(2)).is_some();
    let before = rt.route("x", 1);
    cx.count(
        "replacement_visible_without_fence",
        i64::from(visible && before.is_ok()),
    );
    if visible {
        return Err("the replacement is visible before the event".into());
    }
    route_control("x after a crash before the event", before)?;
    // A crash after the event's append: the replacement is visible exactly with the fence.
    rt.invalidate(InputInvalidation::superseded(
        TRAIN,
        Generation(1),
        D1,
        Generation(2),
        D3,
        reference,
    ))
    .map_err(fail("superseding the input"))?;
    rt.reopen()?;
    cx.attempt("supersession_atomicity_crash")?;
    cx.count("supersession_atomicity_crash_cases", 1);
    let index = rt.runtime.expert_registry().dependencies();
    let visible = index.supersession_floor(TRAIN) == Some(Generation(2))
        && index.registered_digest(TRAIN, Generation(2)) == Some(D3);
    if !visible {
        return Err("the replacement is not visible after the event and a crash".to_string());
    }
    cx.count(
        "replacement_visible_without_fence",
        i64::from(visible && !fenced_and_refused(&rt, "x")),
    );
    // The event records the digest of the blob the harness wrote.
    match rt.runtime.committed_events().last().map(|c| &c.event) {
        Some(LedgerEvent::InputInvalidated {
            kind:
                InputInvalidationKind::Superseded {
                    replacement_ref, ..
                },
            ..
        }) if *replacement_ref == reference => {}
        other => return Err(format!("the supersession event is {other:?}")),
    }
    Ok(())
}

// ---- 9. stale structural dependencies at commit ----------------------------------------------

fn stale_structural_commits(cx: &mut Cx) -> Res<()> {
    cx.sub("stale_structural_dependency_commit");
    for kind in [Kind::BaseModel, Kind::Adapter] {
        for ty in [Type::Revocation, Type::Supersession] {
            let mut rt = Rt::fresh()?;
            let block = kind.block();
            let proposal = rt.proposal(StructuralOp::Birth {
                result: rt.spec("n"),
            })?;
            let report = report_for(&proposal, Some(&block), &ReportSpec::correct())?;
            let grant = rt.admission_grant(Setup::PRE)?;
            rt.invalidate(ty.request(kind))
                .map_err(|error| format!("{} {}: {error:?}", kind.cell(), ty.name()))?;
            let expected: Box<dyn Fn(&RuntimeError) -> bool> = match ty {
                Type::Revocation => Box::new(is_where(|e| {
                    matches!(e, ExpertAdmissionError::InputRevoked { .. })
                })),
                Type::Supersession => Box::new(is_where(|e| {
                    matches!(e, ExpertAdmissionError::InputSuperseded { .. })
                })),
            };
            let verdict = cx.verdict(
                "stale_structural_dependency_commit",
                &format!("a commit after the {} was {}", kind.cell(), ty.name()),
                rt.commit_proposal(&proposal, &report, &grant, Some(block)),
                expected,
            )?;
            cx.count("stale_structural_dependency_commit_cases", 1);
            cx.count(
                "structural_commits_with_stale_dependency",
                verdict.violation(),
            );
        }
    }
    Ok(())
}

// ---- 10. the large fan-out -----------------------------------------------------------------

/// One input, more dependents than an explicit list would fit in a record: one event, a count
/// and a digest, every dependent fenced and unroutable, also after a reopen.
fn large_fanout(cx: &mut Cx) -> Res<()> {
    cx.sub("large_fanout_revocation");
    let n = if cx.reduced() {
        super::super::params::FANOUT_EXPERTS_REDUCED
    } else {
        super::super::params::FANOUT_EXPERTS
    };
    let id_len = super::super::params::FANOUT_ID_LEN;
    let project = "p".repeat(id_len);
    let mut rt = Rt::fresh_in(&project)?;
    let pods: Vec<String> = (0..n).map(|index| format!("{index:0>id_len$}")).collect();
    let block = || inputs(&[("data:big", 1, D1)], &[]);
    for pod in &pods {
        rt.birth_segment(pod, block())?;
    }
    rt.birth_segment(&"b".repeat(id_len), inputs(&[("data:other", 1, D2)], &[]))?;
    let bystander = "b".repeat(id_len);
    let before = rt.event_count();
    let outcome = rt
        .invalidate(InputInvalidation::revoked("data:big", Generation(1), D1))
        .map_err(fail("revoking the fan-out input"))?;
    if rt.event_count() != before + 1 {
        return Err("the fan-out invalidation was not one event".to_string());
    }
    // The explicit list the event does not carry would not have fit in a record.
    let explicit: usize = pods
        .iter()
        .map(|pod| 8 + project.len() + 8 + pod.len() + 8 + 8 + 1 + 1)
        .sum();
    cx.tally
        .set("internal_fanout_explicit_list_bytes", explicit as i64);
    cx.tally.set("internal_fanout_experts", n as i64);
    let exceeds = explicit > super::super::params::MAX_RECORD_BYTES as usize;
    cx.tally.set(
        "internal_fanout_exceeds_max_record_bytes",
        i64::from(exceeds),
    );
    if !cx.reduced() && !exceeds {
        return Err(format!(
            "the fan-out of {n} experts would fit in {explicit} bytes, below MAX_RECORD_BYTES"
        ));
    }
    cx.attempt("large_fanout_revocation")?;
    cx.count("large_fanout_revocation_cases", 1);
    // The count and the digest are the independently derived ones (sorted by pod id).
    let independent: Vec<ExpertFenceSetMember> = {
        let mut sorted = pods.clone();
        sorted.sort_unstable();
        sorted
            .into_iter()
            .map(|pod| ExpertFenceSetMember {
                project: rt.project.clone(),
                pod: PodId::from(pod.as_str()),
                generation: Generation(1),
                expected_state_version: 1,
                action: ExpertFenceAction::Fence,
                resume: ExpertFenceResume::Serving,
            })
            .collect()
    };
    let (_, count, digest) = invalidation_event(&rt)?;
    cx.count(
        "fence_set_digest_mismatches",
        i64::from(
            count as usize != n
                || digest != expert_fence_set_digest("data:big", Generation(1), &independent)
                || outcome.fence_digest() != &digest,
        ),
    );
    // The experts the event fenced are the experts built on the input, as sets.
    let diff = FenceDiff::of(
        pods.iter().map(String::as_str),
        outcome.fenced().iter().map(|member| member.pod.0.as_str()),
    );
    cx.count_fence_findings(&diff, &RouteTally::default());
    // Every dependent is fenced and refused as `Fenced`; the bystander is untouched. The set of
    // the probe is opened before the loop, so a fence that is not live at this probe is a
    // mismatch of the intersection and not a probe that dropped out.
    open_probe(cx, "fanout", "now");
    let mut found = RouteTally::default();
    for pod in &pods {
        found.add(classify_route(pod, rt.route(pod, 1), RouteExpect::Fenced)?);
        if fenced_and_refused(&rt, pod) {
            cx.set_add(&probe_set("fanout", "now"), &format!("fanout/{pod}"));
        }
        cx.set_add("derived_dependents", &format!("fanout/{pod}"));
    }
    found.merge(probe_bystanders(&rt, &[bystander.as_str()])?);
    cx.count("experts_serving_after_revocation", found.served);
    cx.count_fence_findings(&FenceDiff::default(), &found);
    // The set again after a reopen: `open_durable` replays the whole history and re-derives it.
    rt.reopen()?;
    open_probe(cx, "fanout", "reopen");
    let mut found = RouteTally::default();
    for pod in &pods {
        if fenced_and_refused(&rt, pod) {
            cx.set_add(&probe_set("fanout", "reopen"), &format!("fanout/{pod}"));
        }
        found.add(classify_route(pod, rt.route(pod, 1), RouteExpect::Fenced)?);
    }
    found.merge(probe_bystanders(&rt, &[bystander.as_str()])?);
    cx.count("experts_serving_after_revocation", found.served);
    cx.count_fence_findings(&FenceDiff::default(), &found);
    fold_scenario(cx, "fanout");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::Scale;

    /// The fan-out's live probe used to create its set only when a member was fenced, so a
    /// runtime that applied no fence live dropped the probe out of the fold and the reopen
    /// probe alone decided. `open_probe` creates the set first.
    #[test]
    fn a_probe_that_saw_no_fence_takes_part_in_the_fold() {
        let mut cx = Cx::new(17, Scale::Reduced).expect("a context");
        open_probe(&mut cx, "s", "now");
        cx.set_add(&probe_set("s", "reopen"), "s/x");
        fold_scenario(&mut cx, "s");
        assert!(
            cx.sets
                .get("fenced_and_probed")
                .is_none_or(|set| set.is_empty()),
            "the live probe saw no fence, so no member was fenced and probed in every probe"
        );

        // What the fan-out did before: without the opened set the reopen probe alone decides.
        let mut cx = Cx::new(17, Scale::Reduced).expect("a context");
        cx.set_add(&probe_set("s", "reopen"), "s/x");
        fold_scenario(&mut cx, "s");
        assert_eq!(cx.sets["fenced_and_probed"].len(), 1);
    }

    /// A cell whose event left the dependent unfenced (a plain revocation, no fence) and a cell
    /// whose bystander was closed are findings of the cell, not harness errors.
    #[test]
    fn a_cell_with_a_missing_or_an_extra_fence_is_a_finding() {
        let mut cell = build_cell(Kind::Training).expect("a cell");
        cell.rt.revoke(TRAIN, 1).expect("a plain revocation");
        let probed = probe_cell(&mut cell, Kind::Training, Type::Revocation).expect("probes");
        assert_eq!(probed.routable, 0);
        assert_eq!(probed.tally.unfenced, 1);
        assert_eq!(probed.tally.overfenced, 0);

        let mut cell = build_cell(Kind::Training).expect("a cell");
        cell.rt
            .invalidate(Type::Revocation.request(Kind::Training))
            .expect("the invalidation");
        cell.rt.fence_quarantine("w").expect("a fence nobody owed");
        let probed = probe_cell(&mut cell, Kind::Training, Type::Revocation).expect("probes");
        assert_eq!(probed.routable, 0);
        assert_eq!(probed.tally.unfenced, 0);
        assert_eq!(probed.tally.overfenced, 1);
    }
}
