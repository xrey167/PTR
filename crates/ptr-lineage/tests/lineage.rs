use std::collections::BTreeSet;

use ptr_lineage::{
    AccuracyMatrix, AdapterId, AdapterRecord, AdapterStatus, BaseModel, ConsolidationPolicy,
    ForgettingGate, Lineage, LineageError, Origin, PublicSuite,
};
use ptr_types::ArtifactId;

fn base() -> BaseModel {
    BaseModel {
        id: "qwen3-8b".into(),
        revision: "rev-a".into(),
    }
}

fn record(id: &str, origin: Origin) -> AdapterRecord {
    AdapterRecord {
        id: AdapterId::from(id),
        domain: "invoices".into(),
        base: base(),
        origin,
        rank: 16,
        target_modules: BTreeSet::from(["q_proj".to_owned(), "v_proj".to_owned()]),
        artifact: ArtifactId::from(id),
        artifact_sha256: [7; 32],
        data_fingerprint: [9; 32],
        data_manifest: BTreeSet::from([format!("{id}-input")]),
        status: AdapterStatus::Serving,
    }
}

fn trained(id: &str, parent: Option<&str>) -> AdapterRecord {
    record(
        id,
        Origin::Trained {
            parent: parent.map(AdapterId::from),
        },
    )
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.05,
        max_task_forgetting: 0.1,
        min_backward_transfer: -0.05,
        max_public_regression: 0.01,
    }
}

fn passing_report() -> ptr_lineage::GateReport {
    let matrix = AccuracyMatrix::new(vec![vec![0.8, 0.1], vec![0.79, 0.9]]).unwrap();
    gate()
        .evaluate(
            &AdapterId::from("a1"),
            &matrix,
            PublicSuite {
                serving: 0.6,
                candidate: 0.6,
            },
        )
        .unwrap()
}

fn failing_report() -> ptr_lineage::GateReport {
    let matrix = AccuracyMatrix::new(vec![vec![0.8, 0.1], vec![0.5, 0.9]]).unwrap();
    gate()
        .evaluate(
            &AdapterId::from("a1"),
            &matrix,
            PublicSuite {
                serving: 0.6,
                candidate: 0.6,
            },
        )
        .unwrap()
}

#[test]
fn a_registered_adapter_starts_as_a_candidate_whatever_status_it_claims() {
    let mut lineage = Lineage::new(base());
    lineage.register(trained("a1", None)).unwrap();
    assert_eq!(
        lineage.get(&AdapterId::from("a1")).unwrap().status,
        AdapterStatus::Candidate
    );
    assert_eq!(lineage.serving().count(), 0);
}

#[test]
fn only_a_passing_gate_lets_an_adapter_serve() {
    let mut lineage = Lineage::new(base());
    lineage.register(trained("a1", None)).unwrap();
    let id = AdapterId::from("a1");
    assert_eq!(
        lineage.serve(&id).unwrap_err().code(),
        "PTR_LINEAGE_INVALID_TRANSITION"
    );
    assert_eq!(
        lineage.gate(&id, &failing_report()).unwrap_err(),
        LineageError::GateFailed { id: "a1".into() }
    );
    assert_eq!(lineage.get(&id).unwrap().status, AdapterStatus::Candidate);
    lineage.gate(&id, &passing_report()).unwrap();
    lineage.serve(&id).unwrap();
    assert_eq!(lineage.serving().count(), 1);
    lineage.retire(&id).unwrap();
    assert_eq!(lineage.serving().count(), 0);
    assert!(lineage.retire(&id).is_err());
}

#[test]
fn a_passing_report_cannot_gate_another_adapter() {
    let mut lineage = Lineage::new(base());
    lineage.register(trained("a1", None)).unwrap();
    lineage.register(trained("a2", None)).unwrap();
    let report = passing_report();
    let other = AdapterId::from("a2");
    assert_eq!(report.adapter(), &AdapterId::from("a1"));
    assert_eq!(
        lineage.gate(&other, &report),
        Err(LineageError::GateFailed { id: "a2".into() })
    );
    assert_eq!(
        lineage.get(&other).unwrap().status,
        AdapterStatus::Candidate
    );
    lineage.gate(&AdapterId::from("a1"), &report).unwrap();
}

#[test]
fn an_adapter_for_another_base_revision_is_refused() {
    let mut lineage = Lineage::new(base());
    let mut foreign = trained("a1", None);
    foreign.base.revision = "rev-b".into();
    assert!(matches!(
        lineage.register(foreign),
        Err(LineageError::BaseMismatch { .. })
    ));
}

#[test]
fn a_parent_must_exist_before_its_child() {
    let mut lineage = Lineage::new(base());
    assert_eq!(
        lineage.register(trained("a2", Some("a1"))).unwrap_err(),
        LineageError::UnknownAdapter { id: "a1".into() }
    );
}

#[test]
fn consolidation_resets_depth_and_is_due_past_the_policy_limits() {
    let mut lineage = Lineage::new(base());
    lineage.register(trained("a1", None)).unwrap();
    lineage.register(trained("a2", Some("a1"))).unwrap();
    lineage.register(trained("a3", Some("a2"))).unwrap();
    assert_eq!(lineage.depth(&AdapterId::from("a3")).unwrap(), 2);

    let policy = ConsolidationPolicy {
        max_depth: 2,
        max_overlap: 0.3,
    };
    assert!(policy.consolidation_due(2, 0.0));
    assert!(policy.consolidation_due(0, 0.5));
    assert!(!policy.consolidation_due(1, 0.1));

    lineage
        .register(record(
            "c1",
            Origin::Consolidated {
                from: BTreeSet::from([AdapterId::from("a1"), AdapterId::from("a3")]),
            },
        ))
        .unwrap();
    lineage.register(trained("a4", Some("c1"))).unwrap();
    assert_eq!(lineage.depth(&AdapterId::from("a4")).unwrap(), 1);
}

#[test]
fn revoking_one_input_names_its_adapter_every_descendant_and_every_consolidation() {
    let mut lineage = Lineage::new(base());
    lineage.register(trained("a1", None)).unwrap();
    lineage.register(trained("a2", Some("a1"))).unwrap();
    lineage.register(trained("b1", None)).unwrap();
    lineage
        .register(record(
            "c1",
            Origin::Consolidated {
                from: BTreeSet::from([AdapterId::from("a2"), AdapterId::from("b1")]),
            },
        ))
        .unwrap();
    lineage.register(trained("c2", Some("c1"))).unwrap();
    lineage.register(trained("d1", None)).unwrap();

    let affected = lineage.affected_by(&BTreeSet::from(["a1-input".to_owned()]));
    let expected: BTreeSet<AdapterId> = ["a1", "a2", "c1", "c2"]
        .into_iter()
        .map(AdapterId::from)
        .collect();
    assert_eq!(affected, expected);
    assert!(lineage
        .affected_by(&BTreeSet::from(["unrelated".to_owned()]))
        .is_empty());
}

#[test]
fn a_consolidation_must_name_a_registered_source_other_than_itself() {
    let mut lineage = Lineage::new(base());
    lineage.register(trained("a1", None)).unwrap();
    // An empty source set would restart the depth count and leave erasure
    // nothing to follow from the inputs merged into it.
    assert_eq!(
        lineage.register(record(
            "c1",
            Origin::Consolidated {
                from: BTreeSet::new(),
            },
        )),
        Err(LineageError::Empty {
            field: "consolidation sources"
        })
    );
    assert!(lineage.get(&AdapterId::from("c1")).is_none());
    // Naming itself is naming an adapter that is not registered yet.
    assert_eq!(
        lineage.register(record(
            "c1",
            Origin::Consolidated {
                from: BTreeSet::from([AdapterId::from("a1"), AdapterId::from("c1")]),
            },
        )),
        Err(LineageError::UnknownAdapter { id: "c1".into() })
    );
    assert!(lineage.get(&AdapterId::from("c1")).is_none());
    lineage
        .register(record(
            "c1",
            Origin::Consolidated {
                from: BTreeSet::from([AdapterId::from("a1")]),
            },
        ))
        .unwrap();
    assert_eq!(
        lineage.affected_by(&BTreeSet::from(["a1-input".to_owned()])),
        BTreeSet::from([AdapterId::from("a1"), AdapterId::from("c1")])
    );
}
