//! The replica router through its public API: rotation, bounded failover, the errors that end a
//! request, the counters, and resume.
mod common;

use common::{request, resume_request, FlakyReplica};
use ptr_model_api::{
    InferenceBackend, ModelError, ModelErrorKind, ModelEvent, ReplicaError, ReplicaPolicy,
    ReplicaRouter, ReplicaStats, ResumableInferenceBackend, ResumeFailover, MAX_REPLICAS,
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
    // The same request id twice: two invocations, two placements (a and b). The rotation now
    // points at c.
    let first = strict.infer_placed(&request("same-id")).expect("served");
    let second = strict.infer_placed(&request("same-id")).expect("served");
    assert_eq!(label(first.events), "a");
    assert_eq!(label(second.events), "b");
    // Three rounds in a row on each: every round returns the placement of the replica that
    // served it, so the affinity holds for the whole run.
    let (mut on_a, mut on_b) = (first.placement, second.placement);
    for _ in 0..3 {
        let a = strict
            .resume_placed(on_a, &resume_request("same-id"))
            .expect("served");
        assert_eq!(label(a.events), "a");
        on_a = a.placement;
        let b = strict
            .resume_placed(on_b, &resume_request("same-id"))
            .expect("served");
        assert_eq!(label(b.events), "b");
        on_b = b.placement;
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
    let call = |revision: u64| ModelRequest {
        semantic_context: at(revision),
        ..request("same-id")
    };
    let resume = |revision: u64| ModelResumeRequest {
        semantic_context: at(revision),
        ..resume_request("same-id")
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
    // One request id, two invocations on different replicas, told apart by their context.
    let first = router.infer_placed(&call(1)).expect("served");
    let second = router.infer_placed(&call(2)).expect("served");
    assert_eq!(label(first.events), "a");
    assert_eq!(label(second.events), "b");
    // Swapping the placements is refused: each belongs to the context of its own invocation.
    assert_eq!(
        router.resume_placed(second.placement, &resume(1)).err(),
        Some(ModelError::scope_mismatch())
    );
    assert_eq!(
        router.resume_placed(first.placement, &resume(2)).err(),
        Some(ModelError::scope_mismatch())
    );
    // A refused attempt spends the placement; fresh invocations route to their own replica.
    let first = router.infer_placed(&call(1)).expect("served");
    let second = router.infer_placed(&call(2)).expect("served");
    let on_first = router
        .resume_placed(first.placement, &resume(1))
        .expect("resumes");
    let on_second = router
        .resume_placed(second.placement, &resume(2))
        .expect("resumes");
    // The rotation gave the fresh pair c and a.
    assert_eq!(label(on_first.events), "c");
    assert_eq!(label(on_second.events), "a");
    // Another router's placement is refused, and so is a different request id.
    let other = new_router();
    let foreign = other.infer_placed(&call(1)).expect("served");
    assert_eq!(
        router.resume_placed(foreign.placement, &resume(1)).err(),
        Some(ModelError::scope_mismatch())
    );
    let own = router.infer_placed(&call(1)).expect("served");
    assert_eq!(
        router
            .resume_placed(own.placement, &resume_request("another-id"))
            .err(),
        Some(ModelError::scope_mismatch())
    );
}

#[test]
fn open_placements_are_bounded_and_dropping_one_releases_it() {
    use ptr_model_api::MAX_OPEN_PLACEMENTS;
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
    for n in 0..MAX_OPEN_PLACEMENTS {
        held.push(
            router
                .infer_placed(&request(&format!("r{n}")))
                .expect("served"),
        );
    }
    let calls = replicas[0].calls.load(Ordering::SeqCst);
    assert_eq!(
        router.infer_placed(&request("one-too-many")).err(),
        Some(ModelError::overloaded())
    );
    assert_eq!(
        replicas[0].calls.load(Ordering::SeqCst),
        calls,
        "refused before any replica ran"
    );
    drop(held);
    assert!(router.infer_placed(&request("after")).is_ok());
}
