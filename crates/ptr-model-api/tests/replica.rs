//! The replica router through its public API: rotation, bounded failover, the errors that end a
//! request, the counters, and resume.
mod common;

use common::{request, resume_request, FlakyReplica};
use ptr_model_api::{
    InferenceBackend, ModelError, ModelErrorKind, ModelEvent, PlacementOptions, ReplicaError,
    ReplicaPolicy, ReplicaRouter, ReplicaStats, ResumableInferenceBackend, ResumeFailover,
    MAX_REPLICAS,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn policy(max_failover: u8) -> ReplicaPolicy {
    ReplicaPolicy {
        max_failover,
        resume: ResumeFailover::Interchangeable,
    }
}

fn router(replicas: &[Arc<FlakyReplica>], max_failover: u8) -> ReplicaRouter {
    ReplicaRouter::new(
        replicas
            .iter()
            .map(|r| Arc::clone(r) as Arc<dyn ResumableInferenceBackend>)
            .collect(),
        policy(max_failover),
    )
    .expect("router")
}

fn healthy(label: &'static str) -> Arc<FlakyReplica> {
    Arc::new(FlakyReplica::new(label, 0, ModelError::backend_failure()))
}

fn label(events: Vec<ModelEvent>) -> String {
    match events.as_slice() {
        [ModelEvent::Token(label)] => label.clone(),
        other => panic!("unexpected events {other:?}"),
    }
}

#[test]
fn requests_rotate_over_the_replicas() {
    let replicas = [healthy("a"), healthy("b"), healthy("c")];
    let router = router(&replicas, 0);
    let served: Vec<String> = (0..6)
        .map(|n| label(router.infer(&request(&format!("r{n}"))).expect("served")))
        .collect();
    assert_eq!(served, ["a", "b", "c", "a", "b", "c"]);
    for stats in router.stats() {
        assert_eq!(
            stats,
            ReplicaStats {
                attempts: 2,
                failures: 0,
                failovers: 0
            }
        );
    }
}

#[test]
fn a_failed_replica_fails_over_to_the_next_and_is_counted() {
    let broken = Arc::new(FlakyReplica::new("a", usize::MAX, ModelError::overloaded()));
    let replicas = [Arc::clone(&broken), healthy("b")];
    let router = router(&replicas, 1);
    assert_eq!(label(router.infer(&request("r0")).expect("served")), "b");
    let stats = router.stats();
    assert_eq!(
        stats[0],
        ReplicaStats {
            attempts: 1,
            failures: 1,
            failovers: 0
        }
    );
    assert_eq!(
        stats[1],
        ReplicaStats {
            attempts: 1,
            failures: 0,
            failovers: 1
        }
    );
}

#[test]
fn failover_is_bounded_and_each_replica_is_tried_at_most_once() {
    let a = Arc::new(FlakyReplica::new("a", usize::MAX, ModelError::overloaded()));
    let b = Arc::new(FlakyReplica::new("b", usize::MAX, ModelError::overloaded()));
    let c = healthy("c");
    // One failover allowed: a then b, never c.
    let router = router(&[Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)], 1);
    assert_eq!(router.infer(&request("r0")), Err(ModelError::overloaded()));
    assert_eq!(c.calls.load(Ordering::SeqCst), 0);
    assert_eq!(a.calls.load(Ordering::SeqCst), 1);
    assert_eq!(b.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn errors_another_replica_cannot_fix_end_the_request() {
    for error in [
        ModelError::deadline_exceeded(),
        ModelError::cancelled(),
        ModelError::revoked(),
        ModelError::closed(),
    ] {
        let first = Arc::new(FlakyReplica::new("a", 1, error.clone()));
        let second = healthy("b");
        let router = router(&[Arc::clone(&first), Arc::clone(&second)], 1);
        assert_eq!(router.infer(&request("r0")), Err(error));
        assert_eq!(second.calls.load(Ordering::SeqCst), 0, "no failover");
    }
}

#[test]
fn resume_rotates_and_fails_over_like_infer() {
    let first = Arc::new(FlakyReplica::new("a", 1, ModelError::backend_failure()));
    let router = router(&[first, healthy("b")], 1);
    let events = router.resume(&resume_request("r0")).expect("served");
    assert_eq!(label(events), "b");
}

#[test]
fn a_router_needs_replicas_within_bounds() {
    assert_eq!(
        ReplicaRouter::new(Vec::new(), policy(0)).err(),
        Some(ReplicaError::NoReplicas)
    );
    let one = healthy("a");
    assert_eq!(
        ReplicaRouter::new(vec![one as Arc<dyn ResumableInferenceBackend>], policy(1)).err(),
        Some(ReplicaError::FailoverExceedsReplicas {
            max_failover: 1,
            replicas: 1
        })
    );
    let many: Vec<Arc<dyn ResumableInferenceBackend>> = (0..=MAX_REPLICAS)
        .map(|_| healthy("x") as Arc<dyn ResumableInferenceBackend>)
        .collect();
    assert_eq!(
        ReplicaRouter::new(many, policy(0)).err(),
        Some(ReplicaError::TooManyReplicas {
            limit: MAX_REPLICAS,
            actual: MAX_REPLICAS + 1
        })
    );
}

#[test]
fn a_router_composes_over_an_adaptive_batcher() {
    use ptr_model_api::{AdaptiveBatcher, BatchPolicy, ReferenceEchoBackend};
    let batcher = AdaptiveBatcher::new(
        Arc::new(ReferenceEchoBackend),
        BatchPolicy {
            max_batch: 2,
            max_wait_millis: 5,
            queue_capacity: 8,
            deadline_margin_millis: 1,
        },
    )
    .expect("batcher");
    let router = ReplicaRouter::new(
        vec![Arc::new(batcher) as Arc<dyn ResumableInferenceBackend>],
        policy(0),
    )
    .expect("router");
    let events = router.infer(&request("r0")).expect("served");
    assert_eq!(
        events.first(),
        Some(&ModelEvent::Token("text of r0".into()))
    );
}

#[test]
fn a_refusal_of_the_request_itself_is_not_repeated_on_another_replica() {
    // Validation and policy errors are kind Other; so is any error built with ModelError::new.
    for error in [
        ModelError::new("request refused by policy"),
        ModelError::with_kind(ModelErrorKind::Other, "invalid input"),
    ] {
        let first = Arc::new(FlakyReplica::new("a", 1, error.clone()));
        let second = healthy("b");
        let router = router(&[Arc::clone(&first), Arc::clone(&second)], 1);
        assert_eq!(router.infer(&request("r0")), Err(error));
        assert_eq!(second.calls.load(Ordering::SeqCst), 0, "no second try");
    }
}

#[test]
fn only_overload_and_backend_failure_fail_over() {
    for kind in [ModelErrorKind::Overloaded, ModelErrorKind::BackendFailure] {
        let first = Arc::new(FlakyReplica::new("a", 1, ModelError::with_kind(kind, "x")));
        let router = router(&[first, healthy("b")], 1);
        assert_eq!(label(router.infer(&request("r0")).expect("served")), "b");
    }
}

#[test]
fn a_message_that_looks_like_overload_does_not_make_an_error_fail_over() {
    let lookalike = ModelError::new("model backend overloaded");
    let second = healthy("b");
    let router = router(
        &[
            Arc::new(FlakyReplica::new("a", 1, lookalike.clone())),
            Arc::clone(&second),
        ],
        1,
    );
    assert_eq!(router.infer(&request("r0")), Err(lookalike));
    assert_eq!(second.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn resume_does_not_fail_over_unless_the_replicas_are_declared_interchangeable() {
    let replicas = |failures| {
        vec![
            Arc::new(FlakyReplica::new(
                "a",
                failures,
                ModelError::backend_failure(),
            )) as Arc<dyn ResumableInferenceBackend>,
            healthy("b") as Arc<dyn ResumableInferenceBackend>,
        ]
    };
    let strict = ReplicaRouter::new(
        replicas(1),
        ReplicaPolicy {
            max_failover: 1,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    assert_eq!(
        strict.resume(&resume_request("r0")),
        Err(ModelError::backend_failure())
    );
    // infer still fails over under the same policy.
    let still = ReplicaRouter::new(
        replicas(1),
        ReplicaPolicy {
            max_failover: 1,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    assert_eq!(label(still.infer(&request("r0")).expect("served")), "b");
}

/// A replica whose every call panics.
struct PanickingReplica {
    calls: std::sync::atomic::AtomicUsize,
}

impl InferenceBackend for PanickingReplica {
    fn name(&self) -> &'static str {
        "panicking"
    }

    fn infer(&self, _request: &ptr_model_api::ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        panic!("replica panics on purpose");
    }
}

impl ResumableInferenceBackend for PanickingReplica {
    fn resume(
        &self,
        _request: &ptr_model_api::ModelResumeRequest,
    ) -> Result<Vec<ModelEvent>, ModelError> {
        panic!("replica panics on purpose");
    }
}

#[test]
fn a_panicking_replica_is_a_failed_replica_that_fails_over_and_is_counted() {
    let panicking = Arc::new(PanickingReplica {
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let router = ReplicaRouter::new(
        vec![
            Arc::clone(&panicking) as Arc<dyn ResumableInferenceBackend>,
            healthy("b") as Arc<dyn ResumableInferenceBackend>,
        ],
        policy(1),
    )
    .expect("router");
    assert_eq!(label(router.infer(&request("r0")).expect("served")), "b");
    assert_eq!(panicking.calls.load(Ordering::SeqCst), 1);
    let stats = router.stats();
    assert_eq!(
        stats[0],
        ReplicaStats {
            attempts: 1,
            failures: 1,
            failovers: 0
        }
    );
    assert_eq!(stats[1].failovers, 1);
    // With no replica to fail over to, the caller gets a typed error and not an unwind.
    let alone = ReplicaRouter::new(
        vec![panicking as Arc<dyn ResumableInferenceBackend>],
        policy(0),
    )
    .expect("router");
    assert_eq!(
        alone.resume(&resume_request("r1")),
        Err(ModelError::backend_failure())
    );
}

#[test]
fn a_placed_resume_without_failover_lands_on_the_replica_of_its_own_invocation_every_round() {
    let replicas = [healthy("a"), healthy("b"), healthy("c")];
    let strict = ReplicaRouter::new(
        replicas
            .iter()
            .map(|r| Arc::clone(r) as Arc<dyn ResumableInferenceBackend>)
            .collect(),
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    // Two live invocations, two placements (a and b). The rotation now points at c.
    let first = strict
        .infer_placed(&request("id-a"), &PlacementOptions::default())
        .expect("served");
    let second = strict
        .infer_placed(&request("id-b"), &PlacementOptions::default())
        .expect("served");
    assert_eq!(label(first.events), "a");
    assert_eq!(label(second.events), "b");
    // Three rounds in a row on each: every round returns the placement of the replica that
    // served it, so the affinity holds for the whole run.
    let (mut on_a, mut on_b) = (
        first.placement.expect("waiting"),
        second.placement.expect("waiting"),
    );
    for _ in 0..3 {
        let a = strict
            .resume_placed(on_a, &resume_request("id-a"))
            .expect("served");
        assert_eq!(label(a.events), "a");
        on_a = a.placement.expect("waiting");
        let b = strict
            .resume_placed(on_b, &resume_request("id-b"))
            .expect("served");
        assert_eq!(label(b.events), "b");
        on_b = b.placement.expect("waiting");
    }
    assert_eq!(
        replicas[2].calls.load(Ordering::SeqCst),
        0,
        "c never saw either run"
    );
}

#[test]
fn placements_are_bound_to_their_invocation_router_request_and_context() {
    use ptr_model_api::{ModelRequest, ModelResumeRequest};
    use ptr_types::{Revision, SemanticContext};
    let at = |revision: u64| SemanticContext {
        revision: Revision(revision),
        entries: vec![],
    };
    // Each invocation has its own live id; the context is the second binding.
    let call = |revision: u64| ModelRequest {
        semantic_context: at(revision),
        ..request(&format!("id-{revision}"))
    };
    let resume = |revision: u64| ModelResumeRequest {
        semantic_context: at(revision),
        ..resume_request(&format!("id-{revision}"))
    };
    // The right id under the other invocation's context.
    let resume_under = |id: u64, revision: u64| ModelResumeRequest {
        semantic_context: at(revision),
        ..resume_request(&format!("id-{id}"))
    };
    let replicas = [healthy("a"), healthy("b"), healthy("c")];
    let new_router = || {
        ReplicaRouter::new(
            replicas
                .iter()
                .map(|r| Arc::clone(r) as Arc<dyn ResumableInferenceBackend>)
                .collect(),
            ReplicaPolicy {
                max_failover: 0,
                resume: ResumeFailover::Never,
            },
        )
        .expect("router")
    };
    let router = new_router();
    // Two invocations on different replicas, told apart by their id and their context.
    let first = router
        .infer_placed(&call(1), &PlacementOptions::default())
        .expect("served");
    let second = router
        .infer_placed(&call(2), &PlacementOptions::default())
        .expect("served");
    assert_eq!(label(first.events), "a");
    assert_eq!(label(second.events), "b");
    // Swapping the placements is refused: each belongs to the context of its own invocation.
    assert_eq!(
        router
            .resume_placed(second.placement.expect("waiting"), &resume_under(2, 1))
            .err(),
        Some(ModelError::scope_mismatch())
    );
    assert_eq!(
        router
            .resume_placed(first.placement.expect("waiting"), &resume_under(1, 2))
            .err(),
        Some(ModelError::scope_mismatch())
    );
    // A refused attempt spends the placement; fresh invocations route to their own replica.
    let first = router
        .infer_placed(&call(1), &PlacementOptions::default())
        .expect("served");
    let second = router
        .infer_placed(&call(2), &PlacementOptions::default())
        .expect("served");
    let on_first = router
        .resume_placed(first.placement.expect("waiting"), &resume(1))
        .expect("resumes");
    let on_second = router
        .resume_placed(second.placement.expect("waiting"), &resume(2))
        .expect("resumes");
    // The rotation gave the fresh pair c and a.
    assert_eq!(label(on_first.events), "c");
    assert_eq!(label(on_second.events), "a");
    // Their next placements still hold the ids; release them.
    drop(on_first.placement);
    drop(on_second.placement);
    // Another router's placement is refused, and so is a different request id.
    let other = new_router();
    let foreign = other
        .infer_placed(&call(1), &PlacementOptions::default())
        .expect("served");
    assert_eq!(
        router
            .resume_placed(foreign.placement.expect("waiting"), &resume(1))
            .err(),
        Some(ModelError::scope_mismatch())
    );
    let own = router
        .infer_placed(&call(1), &PlacementOptions::default())
        .expect("served");
    assert_eq!(
        router
            .resume_placed(
                own.placement.expect("waiting"),
                &resume_request("another-id")
            )
            .err(),
        Some(ModelError::scope_mismatch())
    );
}

#[test]
fn open_placements_are_bounded_and_dropping_one_releases_it() {
    use ptr_model_api::{MAX_OPEN_PLACEMENTS, MAX_PLACEMENTS_PER_PRINCIPAL};
    let replicas = [healthy("a")];
    let router = ReplicaRouter::new(
        replicas
            .iter()
            .map(|r| Arc::clone(r) as Arc<dyn ResumableInferenceBackend>)
            .collect(),
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    let mut held = Vec::new();
    // Spread over principals so that none is at its own quota: only the router-wide bound is hit.
    for n in 0..MAX_OPEN_PLACEMENTS {
        held.push(
            router
                .infer_placed(
                    &request(&format!("r{n}")),
                    &principal(&format!("p{}", n / MAX_PLACEMENTS_PER_PRINCIPAL)),
                )
                .expect("served"),
        );
    }
    let calls = replicas[0].calls.load(Ordering::SeqCst);
    assert_eq!(
        router
            .infer_placed(&request("one-too-many"), &principal("late"))
            .err(),
        Some(ModelError::overloaded())
    );
    assert_eq!(
        replicas[0].calls.load(Ordering::SeqCst),
        calls,
        "refused before any replica ran"
    );
    drop(held);
    assert!(router
        .infer_placed(&request("after"), &principal("late"))
        .is_ok());
}

fn principal(name: &str) -> PlacementOptions {
    PlacementOptions {
        principal: Some(name.into()),
    }
}

fn single_router() -> ReplicaRouter {
    ReplicaRouter::new(
        vec![healthy("a") as Arc<dyn ResumableInferenceBackend>],
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router")
}

#[test]
fn a_principal_at_its_quota_is_refused_and_another_principal_is_not() {
    use ptr_model_api::MAX_PLACEMENTS_PER_PRINCIPAL as QUOTA;
    let router = single_router();
    let mut mine = Vec::new();
    for n in 0..QUOTA {
        mine.push(
            router
                .infer_placed(&request(&format!("mine-{n}")), &principal("alice"))
                .expect("served"),
        );
    }
    assert_eq!(
        router
            .infer_placed(&request("mine-over"), &principal("alice"))
            .err(),
        Some(ModelError::overloaded())
    );
    // Another principal, and the anonymous bucket, are unaffected and nothing of alice's is lost.
    let bob = router
        .infer_placed(&request("bob"), &principal("bob"))
        .expect("served");
    assert!(router
        .infer_placed(&request("anon"), &PlacementOptions::default())
        .is_ok());
    // A resume keeps alice's charge through its successor ...
    let first = mine.remove(0);
    let resumed = router
        .resume_placed(first.placement.expect("waiting"), &resume_request("mine-0"))
        .expect("resumes");
    assert_eq!(
        router
            .infer_placed(&request("mine-over"), &principal("alice"))
            .err(),
        Some(ModelError::overloaded())
    );
    // ... and dropping a placement gives the slot back.
    drop(resumed);
    assert!(router
        .infer_placed(&request("mine-over"), &principal("alice"))
        .is_ok());
    drop(bob);
}

#[test]
fn calls_without_a_principal_share_one_limited_bucket() {
    use ptr_model_api::MAX_PLACEMENTS_PER_PRINCIPAL as QUOTA;
    let router = single_router();
    let mut held = Vec::new();
    for n in 0..QUOTA {
        held.push(
            router
                .infer_placed(&request(&format!("anon-{n}")), &PlacementOptions::default())
                .expect("served"),
        );
    }
    assert_eq!(
        router
            .infer_placed(&request("anon-over"), &PlacementOptions::default())
            .err(),
        Some(ModelError::overloaded())
    );
    assert!(router
        .infer_placed(&request("named"), &principal("alice"))
        .is_ok());
}

#[test]
fn a_finished_run_and_a_refused_one_do_not_hold_a_principals_quota() {
    use ptr_model_api::MAX_PLACEMENTS_PER_PRINCIPAL as QUOTA;
    let router = ReplicaRouter::new(
        vec![Arc::new(Finishing) as Arc<dyn ResumableInferenceBackend>],
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    for n in 0..QUOTA * 3 {
        let answer = router
            .infer_placed(&request(&format!("r{n}")), &principal("alice"))
            .expect("served");
        assert!(answer.placement.is_none());
    }
    let failing = Arc::new(FlakyReplica::new(
        "a",
        QUOTA * 3,
        ModelError::new("refused"),
    ));
    let router = ReplicaRouter::new(
        vec![failing as Arc<dyn ResumableInferenceBackend>],
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    for n in 0..QUOTA * 3 {
        assert!(router
            .infer_placed(&request(&format!("r{n}")), &principal("alice"))
            .is_err());
    }
    assert!(router
        .infer_placed(&request("ok"), &principal("alice"))
        .is_ok());
}

/// A replica whose runs and resumes always finish.
struct Finishing;

impl InferenceBackend for Finishing {
    fn name(&self) -> &'static str {
        "finishing"
    }

    fn infer(&self, _request: &ptr_model_api::ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

impl ResumableInferenceBackend for Finishing {
    fn resume(
        &self,
        _request: &ptr_model_api::ModelResumeRequest,
    ) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

#[test]
fn a_finished_run_creates_no_placement_and_never_fills_the_table() {
    let router = ReplicaRouter::new(
        vec![Arc::new(Finishing) as Arc<dyn ResumableInferenceBackend>],
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    for n in 0..5000 {
        let answer = router
            .infer_placed(&request(&format!("r{n}")), &PlacementOptions::default())
            .expect("served");
        assert!(answer.placement.is_none());
    }
    // A resume that finishes also leaves no placement behind. Open a placement with a replica
    // that waits, resume it on one that finishes: not possible with one router, so the
    // resume path is covered with a replica that waits first and finishes on resume.
    struct WaitsThenFinishes;
    impl InferenceBackend for WaitsThenFinishes {
        fn name(&self) -> &'static str {
            "waits-then-finishes"
        }
        fn infer(
            &self,
            _request: &ptr_model_api::ModelRequest,
        ) -> Result<Vec<ModelEvent>, ModelError> {
            Ok(vec![ModelEvent::Token("waiting".into())])
        }
    }
    impl ResumableInferenceBackend for WaitsThenFinishes {
        fn resume(
            &self,
            _request: &ptr_model_api::ModelResumeRequest,
        ) -> Result<Vec<ModelEvent>, ModelError> {
            Ok(vec![ModelEvent::Finished])
        }
    }
    let router = ReplicaRouter::new(
        vec![Arc::new(WaitsThenFinishes) as Arc<dyn ResumableInferenceBackend>],
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    for n in 0..5000 {
        let id = format!("r{n}");
        let placed = router
            .infer_placed(&request(&id), &PlacementOptions::default())
            .expect("served");
        let resumed = router
            .resume_placed(placed.placement.expect("waiting"), &resume_request(&id))
            .expect("resumes");
        assert!(resumed.placement.is_none());
    }
}

#[test]
fn a_live_request_id_is_refused_before_any_replica_runs_and_freed_when_it_ends() {
    let replicas = [healthy("a"), healthy("b")];
    let strict = ReplicaRouter::new(
        replicas
            .iter()
            .map(|r| Arc::clone(r) as Arc<dyn ResumableInferenceBackend>)
            .collect(),
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    let calls = || -> usize {
        replicas
            .iter()
            .map(|r| r.calls.load(Ordering::SeqCst))
            .sum()
    };
    let first = strict
        .infer_placed(&request("same-id"), &PlacementOptions::default())
        .expect("served");
    assert_eq!(label(first.events), "a");
    let before = calls();
    // The second invocation under the live id is refused, typed, and no replica ran.
    let refused = strict
        .infer_placed(&request("same-id"), &PlacementOptions::default())
        .err();
    assert_eq!(refused, Some(ModelError::duplicate_request()));
    assert!(refused.is_some_and(|error| error.is_duplicate_request() && !error.allows_failover()));
    assert_eq!(calls(), before, "refused before any replica ran");
    // Another id is unaffected, and the resume of the first lands on its own replica.
    let other = strict
        .infer_placed(&request("other-id"), &PlacementOptions::default())
        .expect("served");
    assert_eq!(label(other.events), "b");
    let resumed = strict
        .resume_placed(
            first.placement.expect("waiting"),
            &resume_request("same-id"),
        )
        .expect("resumes");
    assert_eq!(label(resumed.events), "a");
    // The id stays live across the resume (its next placement holds it) ...
    let held = resumed.placement.expect("waiting");
    assert_eq!(
        strict
            .infer_placed(&request("same-id"), &PlacementOptions::default())
            .err(),
        Some(ModelError::duplicate_request())
    );
    // ... and is free once the placement is dropped.
    drop(held);
    let again = strict
        .infer_placed(&request("same-id"), &PlacementOptions::default())
        .expect("served");
    assert_eq!(label(again.events), "a");
}

#[test]
fn a_request_id_is_free_again_after_a_failed_or_finished_run() {
    // A run that fails frees its id.
    let failing = Arc::new(FlakyReplica::new("a", 1, ModelError::new("refused")));
    let router = ReplicaRouter::new(
        vec![failing as Arc<dyn ResumableInferenceBackend>],
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    assert!(router
        .infer_placed(&request("r"), &PlacementOptions::default())
        .is_err());
    assert!(router
        .infer_placed(&request("r"), &PlacementOptions::default())
        .is_ok());
    // A run that finishes frees its id.
    let router = ReplicaRouter::new(
        vec![Arc::new(Finishing) as Arc<dyn ResumableInferenceBackend>],
        ReplicaPolicy {
            max_failover: 0,
            resume: ResumeFailover::Never,
        },
    )
    .expect("router");
    for _ in 0..3 {
        assert!(router
            .infer_placed(&request("r"), &PlacementOptions::default())
            .is_ok());
    }
}
