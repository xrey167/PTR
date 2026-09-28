//! Host writes under a semantic grant: what the grant's verifiers see,
//! what admits a change, what a refusal reports, and the lifecycle checks a
//! certified plan is committed with.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ptr_config::PtrConfig;
use ptr_events::RuntimeEvent;
use ptr_ledger::{Attestation, LedgerEvent, SemanticOrigin};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::semantic::{StaleReliance, StaleTarget};
use ptr_runtime::{ChangeOrigin, PtrRuntime, RuntimeError, SemanticGrant, SemanticRefusal};
use ptr_search::{retain_live, SearchHit};
use ptr_semdb::{SemanticDelta, SemanticError};
use ptr_types::{
    CapsuleId, Generation, Probability, ProjectId, Revision, Validity, VerificationLevel,
};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

#[path = "common/semantic.rs"]
mod semantic_common;
use semantic_common::{finding, host_write, operator, pass, scripted_runtime, FnVerifier};

fn report(status: VerificationStatus, level: VerificationLevel, hard: bool) -> VerificationReport {
    VerificationReport {
        status,
        level,
        score: Probability::new(0.9).unwrap(),
        findings: if hard {
            vec![Finding {
                code: "constraint".into(),
                message: "hard constraint violated".into(),
                hard: true,
            }]
        } else {
            vec![]
        },
    }
}

fn passing() -> VerificationReport {
    report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        false,
    )
}

fn delta(key: &str, value: &str) -> SemanticDelta {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.into(), value.into());
    delta
}

/// A runtime whose only grant verifier is `verify`, named `judge`, under
/// `required`, with host writes on.
fn judged_by<F>(required: RequiredVerification, verify: F) -> PtrRuntime
where
    F: Fn(&ptr_runtime::SemanticChange<'_>) -> VerificationReport + Send + Sync + 'static,
{
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(required)
                .with_verifier(FnVerifier::new("judge", verify))
                .allow_host_writes(),
        )
        .unwrap();
    runtime
}

/// The origin of the last record `runtime` committed.
fn last_origin(runtime: &PtrRuntime) -> SemanticOrigin {
    match &runtime.committed_events().last().unwrap().event {
        LedgerEvent::SemanticDeltaCommitted { origin, .. } => origin.clone(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn noops_still_require_verification_and_never_append_a_record() {
    for status in [VerificationStatus::Pass, VerificationStatus::Fail] {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let mut runtime = judged_by(RequiredVerification::Deterministic, move |change| {
            seen.fetch_add(1, Ordering::SeqCst);
            assert_eq!(change.after().keys().count(), 0);
            report(status, VerificationLevel::Deterministic, false)
        });
        let revision = runtime.revision();
        let events = runtime.committed_events().len();
        let result = host_write(&mut runtime, revision, SemanticDelta::default());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        match status {
            VerificationStatus::Pass => {
                let commit = result.unwrap();
                assert_eq!(commit.revision, revision);
                assert_eq!(commit.commit_index, None);
                assert!(commit.affected.is_empty());
            }
            _ => assert_eq!(
                result,
                Err(RuntimeError::SemanticVerificationRejected(
                    SemanticRefusal {
                        status: VerificationStatus::Fail,
                        level: VerificationLevel::Deterministic,
                        hard_findings: Vec::new(),
                    }
                ))
            ),
        }
        assert_eq!(runtime.revision(), revision);
        assert_eq!(runtime.committed_events().len(), events);
    }
}

#[test]
fn malformed_deltas_are_rejected_before_the_verifier_runs() {
    let mut runtime = judged_by(RequiredVerification::Deterministic, |_| {
        panic!("invalid input must not reach verification")
    });
    let revision = runtime.revision();
    let events = runtime.committed_events().len();
    let result = host_write(&mut runtime, revision, delta("", "invalid"));
    assert!(matches!(
        result,
        Err(RuntimeError::Semantic(SemanticError::InvalidKey))
    ));
    assert_eq!(runtime.revision(), revision);
    assert_eq!(runtime.committed_events().len(), events);
}

#[test]
fn rejecting_an_overwrite_preserves_values_and_allows_a_verified_retry() {
    let (mut runtime, verifier) = scripted_runtime(RequiredVerification::Deterministic, passing());
    host_write(&mut runtime, Revision(0), delta("price", "10")).unwrap();
    let revision = runtime.revision();
    let events = runtime.committed_events().len();
    let mut refused = report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        true,
    );
    refused.findings.push(finding("second", true));
    refused.findings.push(finding("advice", false));
    verifier.set(refused);
    let result = host_write(&mut runtime, revision, delta("price", "12"));
    assert_eq!(
        result,
        Err(RuntimeError::SemanticVerificationRejected(
            SemanticRefusal {
                status: VerificationStatus::Pass,
                level: VerificationLevel::Deterministic,
                hard_findings: vec!["scripted/constraint".into(), "scripted/second".into()],
            }
        ))
    );
    assert_eq!(runtime.snapshot().get("price"), Some("10"));
    assert_eq!(runtime.revision(), revision);
    assert_eq!(runtime.committed_events().len(), events);
    verifier.set(passing());
    let commit = host_write(&mut runtime, revision, delta("price", "12")).unwrap();
    assert!(commit.commit_index.is_some());
    assert_eq!(runtime.committed_events().len(), events + 1);
    assert_eq!(runtime.snapshot().get("price"), Some("12"));
}

#[test]
fn full_semantic_verification_does_not_satisfy_a_deterministic_requirement() {
    let (mut runtime, _) = scripted_runtime(
        RequiredVerification::Deterministic,
        report(
            VerificationStatus::Pass,
            VerificationLevel::FullSemantic,
            false,
        ),
    );
    let revision = runtime.revision();
    let events = runtime.committed_events().len();
    let result = host_write(&mut runtime, revision, delta("price", "12"));
    assert_eq!(
        result,
        Err(RuntimeError::SemanticVerificationRejected(
            SemanticRefusal {
                status: VerificationStatus::Pass,
                level: VerificationLevel::FullSemantic,
                hard_findings: Vec::new(),
            }
        ))
    );
    assert_eq!(runtime.revision(), revision);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.snapshot().get("price"), None);
}

#[test]
fn a_revoked_generation_is_revoked_although_it_is_still_the_live_generation() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("fact:a"),
            generation: Generation(1),
        })
        .unwrap();
    assert_eq!(
        runtime.generation_validity("fact:a", Generation(1)),
        Some(Validity::Live)
    );
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "fact:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    // The live-generation map alone still names generation 1.
    assert_eq!(runtime.live_generation("fact:a"), Some(Generation(1)));
    assert_eq!(
        runtime.generation_validity("fact:a", Generation(1)),
        Some(Validity::Revoked)
    );
}

#[test]
fn search_hits_filtered_by_generation_validity_drop_a_revoked_live_generation() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    for capsule in ["fact:a", "fact:b"] {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from(capsule),
                generation: Generation(1),
            })
            .unwrap();
    }
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "fact:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    let hits = vec![
        SearchHit::new(CapsuleId::from("fact:a"), Generation(1), 9.0, "dense"),
        SearchHit::new(CapsuleId::from("fact:b"), Generation(1), 1.0, "dense"),
    ];
    // Filtered by equality with the live generation, the revoked hit stayed:
    // the revocation leaves generation 1 live there.
    let kept = retain_live(hits, |capsule, generation| {
        runtime.generation_validity(&capsule.0, generation)
    });
    assert_eq!(
        kept.iter()
            .map(|hit| hit.capsule().0.as_str())
            .collect::<Vec<_>>(),
        vec!["fact:b"]
    );
}

#[test]
fn superseded_and_unknown_generations_are_not_live() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("fact:b"),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("fact:b"),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    assert_eq!(
        runtime.generation_validity("fact:b", Generation(1)),
        Some(Validity::Superseded)
    );
    assert_eq!(
        runtime.generation_validity("fact:b", Generation(2)),
        Some(Validity::Live)
    );
    assert_eq!(runtime.generation_validity("fact:b", Generation(3)), None);
    assert_eq!(
        runtime.generation_validity("fact:none", Generation(1)),
        None
    );
}

#[test]
fn a_verified_delta_commits_the_state_its_verifier_saw() {
    let mut runtime = judged_by(RequiredVerification::FullSemantic, |change| {
        assert_eq!(change.after().get("price:sku-1"), Some("12"));
        report(
            VerificationStatus::Pass,
            VerificationLevel::FullSemantic,
            false,
        )
    });
    let before = runtime.revision();
    let commit = host_write(&mut runtime, before, delta("price:sku-1", "12")).unwrap();
    assert!(commit.commit_index.is_some());
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("12"));
}

#[test]
fn a_verifier_sees_and_can_refuse_the_dependency_set_a_delta_would_install() {
    // A verifier that admits a derived key only as `tax` derived from
    // `rate:a`, judged on the state the change would publish; the published
    // state it applies to has no `tax` yet.
    let mut runtime = judged_by(RequiredVerification::Deterministic, |change| {
        assert_eq!(change.before().inputs("tax").count(), 0);
        let after = change.after();
        let declared = after
            .derived_keys()
            .all(|key| key == "tax" && after.inputs("tax").collect::<Vec<_>>() == ["rate:a"]);
        let status = if declared {
            VerificationStatus::Pass
        } else {
            VerificationStatus::Fail
        };
        report(status, VerificationLevel::Deterministic, false)
    });
    let mut setup = delta("rate:a", "0.19");
    setup.upserts.insert("rate:b".into(), "0.19".into());
    let base = runtime.revision();
    host_write(&mut runtime, base, setup).unwrap();
    // Both deltas publish the same value; only the declared input differs.
    let derive = |input: &str| {
        let mut derived = delta("tax", "19");
        derived
            .dependencies
            .insert("tax".into(), [input.to_owned()].into());
        derived
    };
    let revision = runtime.revision();
    let events = runtime.committed_events().len();
    assert!(matches!(
        host_write(&mut runtime, revision, derive("rate:b")),
        Err(RuntimeError::SemanticVerificationRejected(
            SemanticRefusal {
                status: VerificationStatus::Fail,
                ..
            }
        ))
    ));
    assert_eq!(runtime.revision(), revision);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.snapshot().get("tax"), None);
    host_write(&mut runtime, revision, derive("rate:a")).unwrap();
    let snapshot = runtime.snapshot();
    assert_eq!(snapshot.get("tax"), Some("19"));
    assert_eq!(snapshot.inputs("tax").collect::<Vec<_>>(), ["rate:a"]);
}

#[test]
fn no_score_or_shallow_level_or_hard_finding_gets_a_delta_past_verification() {
    let refusals = [
        report(
            VerificationStatus::Pass,
            VerificationLevel::SampleVerified,
            false,
        ),
        report(
            VerificationStatus::Unknown,
            VerificationLevel::Deterministic,
            false,
        ),
        report(
            VerificationStatus::Disputed,
            VerificationLevel::Deterministic,
            false,
        ),
        report(
            VerificationStatus::Pass,
            VerificationLevel::Deterministic,
            true,
        ),
    ];
    for refused in refusals {
        let (mut runtime, _) = scripted_runtime(RequiredVerification::FullSemantic, refused);
        let before = runtime.revision();
        let committed_before = runtime.committed_events().len();
        let result = host_write(&mut runtime, before, delta("price:sku-1", "1"));
        assert!(matches!(
            result,
            Err(RuntimeError::SemanticVerificationRejected(_))
        ));
        assert_eq!(runtime.revision(), before);
        assert_eq!(runtime.committed_events().len(), committed_before);
    }
}

#[test]
fn a_verified_delta_against_a_moved_revision_is_refused_before_verification() {
    let (mut runtime, verifier) = scripted_runtime(RequiredVerification::Deterministic, passing());
    let stale = runtime.revision();
    host_write(&mut runtime, stale, delta("other", "x")).unwrap();
    assert_eq!(verifier.calls(), 1);
    let result = host_write(&mut runtime, stale, delta("price:sku-1", "1"));
    assert!(matches!(result, Err(RuntimeError::Semantic(_))));
    assert_eq!(verifier.calls(), 1);
}

/// A runtime where capsule `fact:a` is live at generation 1, whose grant
/// passes every change.
fn runtime_with_fact() -> PtrRuntime {
    let (mut runtime, _) = scripted_runtime(RequiredVerification::Deterministic, passing());
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("fact:a"),
            generation: Generation(1),
        })
        .unwrap();
    runtime
}

#[test]
fn a_certified_delta_is_refused_while_a_generation_it_relied_on_is_not_live_and_appends_nothing() {
    let revoke = LedgerEvent::Revoked {
        subject: "fact:a".into(),
        generation: Generation(1),
    };
    let supersede = LedgerEvent::CapsuleSuperseded {
        capsule: CapsuleId::from("fact:a"),
        old: Generation(1),
        new: Generation(2),
    };
    // Relied at generation 1 after a revocation or a supersession, and at a
    // generation the authority never published.
    for (change, relied, validity) in [
        (Some(revoke), Generation(1), Some(Validity::Revoked)),
        (Some(supersede), Generation(1), Some(Validity::Superseded)),
        (None, Generation(7), None),
    ] {
        let mut runtime = runtime_with_fact();
        if let Some(change) = change {
            runtime.commit(change).unwrap();
        }
        let revision = runtime.revision();
        let events = runtime.committed_events().len();
        let relied = BTreeMap::from([("fact:a".to_owned(), relied)]);
        let refused = runtime
            .apply_certified_semantic_delta(revision, delta("price", "12"), &relied, &operator())
            .unwrap_err();
        let RuntimeError::StaleReliance(stale) = refused else {
            panic!("{refused:?}");
        };
        assert_eq!(
            stale,
            StaleReliance {
                targets: BTreeMap::from([(
                    "fact:a".to_owned(),
                    StaleTarget {
                        relied: relied["fact:a"],
                        validity,
                    },
                )]),
            }
        );
        assert_eq!(stale.code(), "PTR_RUNTIME_STALE_RELIANCE");
        assert_eq!(runtime.revision(), revision);
        assert_eq!(runtime.committed_events().len(), events);
        assert_eq!(runtime.snapshot().get("price"), None);
        // A plan that would change nothing is refused too.
        assert!(matches!(
            runtime.apply_certified_semantic_delta(
                revision,
                SemanticDelta::default(),
                &relied,
                &operator(),
            ),
            Err(RuntimeError::StaleReliance(_))
        ));
        assert_eq!(runtime.committed_events().len(), events);
    }

    // A live reliance commits, and only the targets that are not live are
    // named.
    let mut runtime = runtime_with_fact();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("fact:b"),
            generation: Generation(1),
        })
        .unwrap();
    let revision = runtime.revision();
    let both = BTreeMap::from([
        ("fact:a".to_owned(), Generation(1)),
        ("fact:b".to_owned(), Generation(1)),
    ]);
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "fact:b".into(),
            generation: Generation(1),
        })
        .unwrap();
    assert!(matches!(
        runtime.apply_certified_semantic_delta(revision, delta("price", "12"), &both, &operator()),
        Err(RuntimeError::StaleReliance(StaleReliance { ref targets }))
            if targets.keys().collect::<Vec<_>>() == ["fact:b"]
    ));
    let live = BTreeMap::from([("fact:a".to_owned(), Generation(1))]);
    let commit = runtime
        .apply_certified_semantic_delta(revision, delta("price", "12"), &live, &operator())
        .unwrap();
    assert!(commit.commit_index.is_some());
    assert_eq!(runtime.snapshot().get("price"), Some("12"));
}

#[test]
fn a_verifier_sees_before_after_delta_affected_and_its_origin() {
    #[derive(Debug, PartialEq)]
    struct Seen {
        base: Revision,
        next: Revision,
        before: Option<String>,
        after: Option<String>,
        derived_after: Option<String>,
        delta: SemanticDelta,
        affected: Vec<String>,
        principal: String,
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = seen.clone();
    let mut runtime = judged_by(RequiredVerification::Deterministic, move |change| {
        let ChangeOrigin::Host { principal } = change.origin();
        record.lock().unwrap().push(Seen {
            base: change.base_revision(),
            next: change.next_revision(),
            before: change.before().get("rate").map(str::to_owned),
            after: change.after().get("rate").map(str::to_owned),
            derived_after: change.after().get("tax").map(str::to_owned),
            delta: change.delta().clone(),
            affected: change.affected().iter().cloned().collect(),
            principal: principal.0.clone(),
        });
        pass(VerificationLevel::Deterministic)
    });
    let mut setup = delta("rate", "0.19");
    setup.upserts.insert("tax".into(), "19".into());
    setup
        .dependencies
        .insert("tax".into(), ["rate".to_owned()].into());
    host_write(&mut runtime, Revision(0), setup).unwrap();
    // Changing the input evicts the derived key: the verifier sees the
    // published value before, the new one after, the eviction, and every key
    // the change invalidates.
    let change = delta("rate", "0.2");
    host_write(&mut runtime, Revision(1), change.clone()).unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen[1],
        Seen {
            base: Revision(1),
            next: Revision(2),
            before: Some("0.19".into()),
            after: Some("0.2".into()),
            derived_after: None,
            delta: change,
            affected: vec!["rate".into(), "tax".into()],
            principal: operator().0,
        }
    );
}

#[test]
fn the_weakest_verifier_level_decides() {
    let grant = |required| {
        SemanticGrant::new(required)
            .with_verifier(FnVerifier::new("strong", |_| {
                pass(VerificationLevel::Deterministic)
            }))
            .with_verifier(FnVerifier::new("weaker", |_| {
                pass(VerificationLevel::FullSemantic)
            }))
            .allow_host_writes()
    };
    // Under a deterministic requirement the weaker verifier's full-semantic
    // level decides, and the change is refused.
    let mut strict = PtrRuntime::new(PtrConfig::default()).unwrap();
    strict
        .install_semantic_grant(grant(RequiredVerification::Deterministic))
        .unwrap();
    assert_eq!(
        host_write(&mut strict, Revision(0), delta("price", "1")),
        Err(RuntimeError::SemanticVerificationRejected(
            SemanticRefusal {
                status: VerificationStatus::Pass,
                level: VerificationLevel::FullSemantic,
                hard_findings: Vec::new(),
            }
        ))
    );
    // Nothing was appended, and each verifier's own result was emitted: the
    // strong one passed the requirement, the weaker one did not.
    assert!(strict.committed_events().is_empty());
    assert_eq!(strict.revision(), Revision(0));
    assert_eq!(
        verifier_results(&strict),
        [("strong".to_owned(), true), ("weaker".to_owned(), false)]
    );
    // Under a full-semantic requirement it is admitted, and the record names
    // both verifiers in grant order and the weakest level.
    let mut lenient = PtrRuntime::new(PtrConfig::default()).unwrap();
    lenient
        .install_semantic_grant(grant(RequiredVerification::FullSemantic))
        .unwrap();
    host_write(&mut lenient, Revision(0), delta("price", "1")).unwrap();
    assert_eq!(
        last_origin(&lenient),
        SemanticOrigin::Host {
            principal: operator().0,
            verification: Attestation {
                required: VerificationLevel::FullSemantic,
                level: VerificationLevel::FullSemantic,
                verifiers: vec!["strong".into(), "weaker".into()],
                findings: Vec::new(),
            },
        }
    );
    // Each verifier's result was emitted as it was judged.
    assert_eq!(
        verifier_results(&lenient),
        [("strong".to_owned(), true), ("weaker".to_owned(), true)]
    );
}

/// Every verifier result the runtime emitted, in order.
fn verifier_results(runtime: &PtrRuntime) -> Vec<(String, bool)> {
    runtime
        .events()
        .iter()
        .filter_map(|envelope| match &envelope.event {
            RuntimeEvent::VerifierResult { verifier, passed } => Some((verifier.clone(), *passed)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_refusal_reports_hard_finding_codes_never_messages() {
    let mut runtime = judged_by(RequiredVerification::Deterministic, |_| {
        let mut report = pass(VerificationLevel::Deterministic);
        report.findings = vec![finding("stock-negative", true), finding("style", false)];
        report
    });
    let refused = host_write(&mut runtime, Revision(0), delta("stock", "-1")).unwrap_err();
    let RuntimeError::SemanticVerificationRejected(refusal) = refused else {
        panic!("{refused:?}");
    };
    assert_eq!(refusal.hard_findings, ["judge/stock-negative"]);
    assert_eq!(refusal.code(), "PTR_RUNTIME_SEMANTIC_VERIFICATION_REJECTED");
    let shown = format!("{refusal} {refusal:?}");
    assert!(!shown.contains("message for"), "{shown}");
    assert!(runtime.committed_events().is_empty());

    // Soft findings of an admitted change are recorded by code, sorted and
    // distinct.
    let mut runtime = judged_by(RequiredVerification::Deterministic, |_| {
        let mut report = pass(VerificationLevel::Deterministic);
        report.findings = vec![
            finding("zeta", false),
            finding("alpha", false),
            finding("zeta", false),
        ];
        report
    });
    host_write(&mut runtime, Revision(0), delta("stock", "1")).unwrap();
    let SemanticOrigin::Host { verification, .. } = last_origin(&runtime) else {
        panic!("a host write records a host origin");
    };
    assert_eq!(verification.findings, ["judge/alpha", "judge/zeta"]);
}

#[test]
fn an_invalid_finding_code_fails_closed() {
    let too_long = "c".repeat(ptr_runtime::merge::MAX_FINDING_CODE + 1);
    for code in ["", " padded", "line\nbreak", too_long.as_str()] {
        let code = code.to_owned();
        let mut runtime = judged_by(RequiredVerification::Deterministic, move |_| {
            let mut report = pass(VerificationLevel::Deterministic);
            report.findings = vec![finding(&code, false)];
            report
        });
        assert_eq!(
            host_write(&mut runtime, Revision(0), delta("price", "1")),
            Err(RuntimeError::InvalidVerificationReport {
                verifier: "judge".into()
            })
        );
        assert!(runtime.committed_events().is_empty());
    }
    // More soft findings than a record carries fail closed too; exactly the
    // bound is recorded.
    for (count, admitted) in [
        (ptr_runtime::merge::MAX_ATTESTED_FINDINGS, true),
        (ptr_runtime::merge::MAX_ATTESTED_FINDINGS + 1, false),
    ] {
        let mut runtime = judged_by(RequiredVerification::Deterministic, move |_| {
            let mut report = pass(VerificationLevel::Deterministic);
            report.findings = (0..count)
                .map(|n| finding(&format!("f{n:02}"), false))
                .collect();
            report
        });
        let result = host_write(&mut runtime, Revision(0), delta("price", "1"));
        assert_eq!(result.is_ok(), admitted, "{count}: {result:?}");
    }
}

#[test]
fn a_report_that_fails_closed_is_emitted_as_not_passed_and_stops_the_verifiers_after_it() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(FnVerifier::new("first", |_| {
                    pass(VerificationLevel::Deterministic)
                }))
                .with_verifier(FnVerifier::new("second", |_| {
                    let mut report = pass(VerificationLevel::Deterministic);
                    report.findings = vec![finding(" bad", false)];
                    report
                }))
                .with_verifier(FnVerifier::new("third", |_| {
                    panic!("a verifier after one that failed closed is not asked")
                }))
                .allow_host_writes(),
        )
        .unwrap();
    assert_eq!(
        host_write(&mut runtime, Revision(0), delta("price", "1")),
        Err(RuntimeError::InvalidVerificationReport {
            verifier: "second".into()
        })
    );
    assert!(runtime.committed_events().is_empty());
    // The verifiers that judged are in the telemetry: the first passed, the
    // one whose report failed closed did not, and the third never judged.
    assert_eq!(
        verifier_results(&runtime),
        [("first".to_owned(), true), ("second".to_owned(), false)]
    );
}
