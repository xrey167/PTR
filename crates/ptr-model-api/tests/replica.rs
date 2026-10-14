//! The replica router through its public API: rotation, bounded failover, the errors that end a
//! request, the counters, and resume.
mod common;

use common::{request, resume_request, FlakyReplica};
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ReplicaError, ReplicaRouter, ReplicaStats,
    ResumableInferenceBackend, MAX_REPLICAS,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn router(replicas: &[Arc<FlakyReplica>], max_failover: u8) -> ReplicaRouter {
    ReplicaRouter::new(
        replicas
            .iter()
            .map(|r| Arc::clone(r) as Arc<dyn ResumableInferenceBackend>)
            .collect(),
        max_failover,
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
        ReplicaRouter::new(Vec::new(), 0).err(),
        Some(ReplicaError::NoReplicas)
    );
    let one = healthy("a");
    assert_eq!(
        ReplicaRouter::new(vec![one as Arc<dyn ResumableInferenceBackend>], 1).err(),
        Some(ReplicaError::FailoverExceedsReplicas {
            max_failover: 1,
            replicas: 1
        })
    );
    let many: Vec<Arc<dyn ResumableInferenceBackend>> = (0..=MAX_REPLICAS)
        .map(|_| healthy("x") as Arc<dyn ResumableInferenceBackend>)
        .collect();
    assert_eq!(
        ReplicaRouter::new(many, 0).err(),
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
        0,
    )
    .expect("router");
    let events = router.infer(&request("r0")).expect("served");
    assert_eq!(
        events.first(),
        Some(&ModelEvent::Token("text of r0".into()))
    );
}
