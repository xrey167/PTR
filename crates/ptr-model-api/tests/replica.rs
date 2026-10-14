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
fn a_resume_without_failover_goes_back_to_the_replica_that_started_the_run() {
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
    // r0 starts on a, r1 on b. The rotation now points at c.
    assert_eq!(label(strict.infer(&request("r0")).expect("served")), "a");
    assert_eq!(label(strict.infer(&request("r1")).expect("served")), "b");
    // Each resume lands on the replica that started its run, not on the next in rotation.
    assert_eq!(
        label(strict.resume(&resume_request("r1")).expect("served")),
        "b"
    );
    assert_eq!(
        label(strict.resume(&resume_request("r0")).expect("served")),
        "a"
    );
    assert_eq!(
        replicas[2].calls.load(Ordering::SeqCst),
        0,
        "c never saw either run"
    );
}
