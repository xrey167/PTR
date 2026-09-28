//! Merging an agent branch, the one way a branch reaches semantic state: the
//! runtime certifies a sealed branch against its own state, lets every
//! verifier of the host's grant judge the exact state the merge would
//! publish, and commits it under the grant's merge policy or a listed
//! reviewer's approval of the plan digest, at most once per branch id, as one
//! record naming who and what admitted it.
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ptr_branch::{
    calibration_draw, counter_value, merge_plan_digest, read_counter_value, AutoThreshold, Branch,
    BranchError, BranchId, BranchOp, CertificationKind, PolicyRecord, SealedBranch, TriageDecision,
    TriageOutcome, TriagePolicy,
};
use ptr_config::PtrConfig;
use ptr_events::RuntimeEvent;
use ptr_ledger::{
    Attestation, CommittedEvent, LedgerEvent, MergeAuthorityRecord, MergeRecord, SemanticOrigin,
};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::semantic::{pod_output_key, request_raw_key};
use ptr_runtime::{
    ChangeOrigin, HoldReason, MergeAuthority, MergeHold, MergeOutcome, MergeReceipt, PtrRuntime,
    RuntimeError, SemanticChange, SemanticGrant,
};
use ptr_semdb::{SemanticDelta, SemanticValue};
use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, Generation, PodId, PrincipalId, Probability,
    ProjectId, RequestId, Revision, Validity, VerificationLevel,
};
use ptr_verifier::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// The version of the merge policy every grant here installs.
const POLICY: &str = "pricing-policy-v1";
/// The seed of that policy's calibration draws.
const SEED: u64 = 7;

fn report(status: VerificationStatus, level: VerificationLevel) -> VerificationReport {
    VerificationReport {
        status,
        level,
        score: Probability::new(0.95).unwrap(),
        findings: vec![],
    }
}

fn passing() -> VerificationReport {
    report(VerificationStatus::Pass, VerificationLevel::Deterministic)
}

/// What the grant's verifier was shown of one change: the price the change
/// would publish, and for a merge the branch and how it certified.
#[derive(Clone, Debug, PartialEq)]
struct Seen {
    price: Option<String>,
    merge: Option<(String, CertificationKind)>,
}

/// The grant verifier of these tests: remembers what each change would
/// publish, and returns the report its test last set (passing at the
/// deterministic level until one is set). Clones share both.
#[derive(Clone, Default)]
struct PriceWatch {
    seen: Arc<Mutex<Vec<Seen>>>,
    report: Arc<Mutex<Option<VerificationReport>>>,
}

impl PriceWatch {
    fn set(&self, report: VerificationReport) {
        *self.report.lock().unwrap() = Some(report);
    }

    /// How many changes it has judged.
    fn count(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    fn last(&self) -> Option<Seen> {
        self.seen.lock().unwrap().last().cloned()
    }
}

impl<'a> Verifier<SemanticChange<'a>> for PriceWatch {
    fn verify(&self, change: &SemanticChange<'a>) -> VerificationReport {
        let merge = match change.origin() {
            ChangeOrigin::Merge { branch, kind, .. } => Some((branch.id().0.clone(), kind)),
            ChangeOrigin::Host { .. } => None,
        };
        self.seen.lock().unwrap().push(Seen {
            price: change.after().get("price:sku-1").map(str::to_owned),
            merge,
        });
        self.report.lock().unwrap().clone().unwrap_or_else(passing)
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for PriceWatch {
    fn name(&self) -> &'static str {
        "price-watch"
    }
}

/// The author every branch here is sealed by.
fn agent() -> PrincipalId {
    PrincipalId::from("pricing-agent")
}

/// The reviewer every grant here lists.
fn reviewer() -> PrincipalId {
    PrincipalId::from("reviewer-1")
}

/// A merge policy auto-proposing a branch that scores at least 0.8, sending
/// `calibration_rate` of the eligible ones to a person.
fn policy(calibration_rate: f64) -> PolicyRecord {
    PolicyRecord::manual(
        POLICY,
        TriagePolicy::new(AutoThreshold::AtLeast(0.8), calibration_rate).unwrap(),
    )
    .unwrap()
}

/// A grant requiring the deterministic level of `watch` alone, with host
/// writes on, `policy` as its merge policy under [`SEED`], and [`reviewer`].
fn grant_with(watch: &PriceWatch, policy: PolicyRecord) -> SemanticGrant {
    SemanticGrant::new(RequiredVerification::Deterministic)
        .with_verifier(watch.clone())
        .allow_host_writes()
        .with_merge_policy(policy, SEED)
        .with_reviewer(reviewer())
}

/// [`grant_with`] a policy with no calibration slice.
fn grant(watch: &PriceWatch) -> SemanticGrant {
    grant_with(watch, policy(0.0))
}

/// Install [`grant`] on `runtime`, commit `pricing-policy` live at generation
/// 1, and write a price of 10 and a stock counter of 5 as a host write.
fn set_up(runtime: &mut PtrRuntime, watch: &PriceWatch) {
    runtime.install_semantic_grant(grant(watch)).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("shop"),
            capsule: CapsuleId::from("pricing-policy"),
            generation: Generation(1),
        })
        .unwrap();
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("price:sku-1".into(), "10".into());
    delta.upserts.insert("stock".into(), counter_value(5));
    operator_writes(runtime, delta);
}

/// A runtime [`set_up`] under a fresh [`PriceWatch`], which is returned.
fn runtime_watching_prices() -> (PtrRuntime, PriceWatch) {
    let watch = PriceWatch::default();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    set_up(&mut runtime, &watch);
    (runtime, watch)
}

/// `delta` written by the pipeline operator as a host write at the current
/// revision.
fn operator_writes(runtime: &mut PtrRuntime, delta: SemanticDelta) {
    let revision = runtime.revision();
    runtime
        .apply_verified_semantic_delta(revision, delta, &PrincipalId::from("pipeline-operator"))
        .unwrap();
}

fn put(key: &str, value: impl Into<SemanticValue>) -> SemanticDelta {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.into(), value.into());
    delta
}

/// A branch `id` over `runtime`'s current state that relies on
/// `pricing-policy` at generation 1, reads the price and reprices to 11.
fn repricing(runtime: &PtrRuntime, id: &str) -> SealedBranch {
    let mut work = Branch::open(BranchId::from(id), agent(), runtime.snapshot());
    work.rely_on("pricing-policy", Generation(1)).unwrap();
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    work.seal().unwrap()
}

/// A branch `id` over `runtime`'s current state that adds `amount` to the
/// stock counter.
fn restocking(runtime: &PtrRuntime, id: &str, amount: i64) -> SealedBranch {
    let mut work = Branch::open(BranchId::from(id), agent(), runtime.snapshot());
    work.stage_commutative(BranchOp::Add {
        key: "stock".into(),
        amount,
    })
    .unwrap();
    work.seal().unwrap()
}

fn triaged(score: f32) -> MergeAuthority {
    MergeAuthority::Triage {
        score: Probability::new(score).unwrap(),
    }
}

/// The runtime's own triage at a score its policy auto-proposes.
fn auto() -> MergeAuthority {
    triaged(0.9)
}

/// [`reviewer`]'s approval of the plan whose digest is `plan_digest`.
fn reviewed(plan_digest: [u8; 32]) -> MergeAuthority {
    MergeAuthority::Reviewed {
        plan_digest,
        reviewer: reviewer(),
    }
}

fn committed(outcome: MergeOutcome) -> MergeReceipt {
    match outcome {
        MergeOutcome::Committed(receipt) => receipt,
        other => panic!("not committed: {other:?}"),
    }
}

fn held(outcome: MergeOutcome) -> MergeHold {
    match outcome {
        MergeOutcome::Held(hold) => hold,
        other => panic!("not held: {other:?}"),
    }
}

/// The base revision, encoded delta and merge record committed at `index`.
fn merge_at(runtime: &PtrRuntime, index: CommitIndex) -> (Revision, Vec<u8>, MergeRecord) {
    let committed = runtime
        .committed_events()
        .iter()
        .find(|committed| committed.index == index)
        .expect("a record at that index");
    match &committed.event {
        LedgerEvent::SemanticDeltaCommitted {
            base_revision,
            encoded_delta,
            origin: SemanticOrigin::Merge(record),
            ..
        } => (*base_revision, encoded_delta.clone(), record.clone()),
        other => panic!("not a merge record: {other:?}"),
    }
}

/// Whether the runtime still commits: it is not fenced.
fn unfenced(runtime: &PtrRuntime) -> bool {
    runtime.snapshot_is_current(&runtime.snapshot())
}

/// Whether any materialized key records a merged branch.
fn marks_a_merge(runtime: &PtrRuntime) -> bool {
    runtime
        .materialized_state()
        .values
        .keys()
        .any(|key| key.starts_with(ptr_state::MERGED_BRANCH_PREFIX))
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptr-branch-merge-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn log(&self) -> PathBuf {
        self.0.join("log")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_certified_and_verified_branch_reaches_semantic_state_only_through_the_runtime() {
    let (mut runtime, watch) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let events = runtime.committed_events().len();

    let receipt = committed(runtime.merge_branch(&sealed, auto()).unwrap());

    // The grant's verifier judged the state the merge would publish, as a
    // merge of this branch.
    assert_eq!(
        watch.last(),
        Some(Seen {
            price: Some("11".into()),
            merge: Some(("b1".into(), CertificationKind::Clean)),
        })
    );
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
    assert_eq!(runtime.committed_events().len(), events + 1);
    let index = receipt.commit.commit_index.unwrap();
    assert_eq!(runtime.merged_at(&BranchId::from("b1")), Some(index));
    assert_eq!(receipt.kind, CertificationKind::Clean);
    assert_eq!(
        receipt.triage.unwrap().decision(),
        TriageDecision::AutoPropose
    );
    // Its record is a merge of the branch by its author, not a host write.
    let (_, _, record) = merge_at(&runtime, index);
    assert_eq!(
        (record.branch.as_str(), record.author.as_str()),
        ("b1", "pricing-agent")
    );
}

#[test]
fn a_revocation_or_supersession_after_certification_refuses_the_commit_and_appends_nothing() {
    let revoke = LedgerEvent::Revoked {
        subject: "pricing-policy".into(),
        generation: Generation(1),
    };
    let supersede = LedgerEvent::CapsuleSuperseded {
        capsule: CapsuleId::from("pricing-policy"),
        old: Generation(1),
        new: Generation(2),
    };
    for (change, validity) in [
        (revoke, Validity::Revoked),
        (supersede, Validity::Superseded),
    ] {
        let (mut runtime, watch) = runtime_watching_prices();
        let sealed = repricing(&runtime, "b1");
        let preview = runtime.preview_merge(&sealed).unwrap();
        assert!(preview.verdict.admitted());
        // The lifecycle changes after certification; the semantic revision
        // the plan was certified against does not move.
        runtime.commit(change).unwrap();
        assert_eq!(preview.revision, runtime.revision());
        assert_eq!(
            runtime.generation_validity("pricing-policy", Generation(1)),
            Some(validity)
        );
        let events = runtime.committed_events().len();
        let judged = watch.count();

        // Merging certifies again, and the reliance no longer holds, whoever
        // authorizes it.
        for authority in [auto(), reviewed(preview.plan_digest)] {
            assert_eq!(
                runtime.merge_branch(&sealed, authority),
                Err(RuntimeError::Certification(BranchError::LifecycleChanged {
                    targets: BTreeSet::from(["pricing-policy".to_owned()]),
                }))
            );
        }
        // Refused before the verifier was asked, with nothing appended.
        assert_eq!(watch.count(), judged);
        assert_eq!(runtime.committed_events().len(), events);
        assert_eq!(runtime.snapshot().get("price:sku-1"), Some("10"));
        assert_eq!(runtime.merged_at(&BranchId::from("b1")), None);
        assert!(unfenced(&runtime));
    }

    // With the lifecycle unchanged, the same branch commits.
    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let events = runtime.committed_events().len();
    committed(runtime.merge_branch(&sealed, auto()).unwrap());
    assert_eq!(runtime.committed_events().len(), events + 1);
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn a_commit_that_moves_neither_the_revision_nor_a_relied_generation_does_not_refuse_the_plan() {
    // A merge certifies against the runtime's state as it is when it
    // merges: a new hard constraint, a capsule the branch did not rely on and
    // a verifier attestation change neither what the branch read nor what it
    // relied on, so the plan certified before them is certified again,
    // unchanged, and an approval of it stands. A domain rule such as the new
    // constraint is taken into account only if the grant's verifier looks
    // for it.
    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let preview = runtime.preview_merge(&sealed).unwrap();
    for unrelated in [
        LedgerEvent::HardConstraintCommitted {
            key: "max-price".into(),
            generation: Generation(1),
        },
        LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("shop"),
            capsule: CapsuleId::from("shipping-policy"),
            generation: Generation(1),
        },
        LedgerEvent::VerifierAttested {
            subject: "price:sku-1".into(),
            passed: false,
        },
    ] {
        runtime.commit(unrelated).unwrap();
        assert_eq!(preview.revision, runtime.revision());
    }
    assert!(!sealed.relied().contains_key("constraint:max-price"));
    assert_eq!(
        runtime.preview_merge(&sealed).unwrap().plan_digest,
        preview.plan_digest
    );
    let receipt = committed(
        runtime
            .merge_branch(&sealed, reviewed(preview.plan_digest))
            .unwrap(),
    );
    assert_eq!(
        receipt.authority,
        MergeAuthorityRecord::Reviewed {
            reviewer: "reviewer-1".into()
        }
    );
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn an_unsettled_effect_attempt_fences_the_plan_until_it_is_reconciled() {
    // An effect record moves neither the revision nor a relied generation.
    // But an attempt that is neither settled nor reconciled fences every
    // commit of the runtime, this merge included, until an operator or the
    // dispatch settles it.
    let (mut runtime, watch) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let preview = runtime.preview_merge(&sealed).unwrap();
    let judged = watch.count();
    let attempt = runtime
        .commit(LedgerEvent::EffectAttempted {
            key: None,
            project: ProjectId::from("shop"),
            principal: "pricing-agent".into(),
            target: "pricing-policy".into(),
            operation: "notify".into(),
            capability: CapabilityId::from("mail.send"),
            effect: Effect::External,
            generation: Generation(1),
            revision: runtime.revision(),
            verification: VerificationLevel::Deterministic,
            action_digest: [0; 32],
        })
        .unwrap();
    assert_eq!(preview.revision, runtime.revision());
    assert_eq!(
        runtime.generation_validity("pricing-policy", Generation(1)),
        Some(Validity::Live)
    );
    let events = runtime.committed_events().len();
    for authority in [auto(), reviewed(preview.plan_digest)] {
        assert_eq!(
            runtime.merge_branch(&sealed, authority),
            Err(RuntimeError::ExecutionFenced)
        );
    }
    // Refused before the grant's verifier was asked.
    assert_eq!(watch.count(), judged);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("10"));

    // Reconciling the attempt lifts the fence without moving the revision,
    // and the same approval then commits.
    runtime
        .reconcile_effect(attempt, false, "operator-confirmed-not-sent")
        .unwrap();
    assert_eq!(preview.revision, runtime.revision());
    let receipt = committed(
        runtime
            .merge_branch(&sealed, reviewed(preview.plan_digest))
            .unwrap(),
    );
    assert!(receipt.commit.commit_index.is_some());
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn a_plan_delta_written_as_a_host_write_is_recorded_as_host_not_merge() {
    // A plan's delta is an ordinary semantic delta, and a host may write it
    // under its own grant like any other. That is a host write: its record
    // names the principal, not the branch, and the branch is not merged.
    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let preview = runtime.preview_merge(&sealed).unwrap();
    let plan = preview.certification.plan();
    let commit = runtime
        .apply_verified_semantic_delta(plan.expected(), plan.delta().clone(), &agent())
        .unwrap();
    let index = commit.commit_index.unwrap();
    let written = runtime
        .committed_events()
        .iter()
        .find(|committed| committed.index == index)
        .unwrap();
    assert!(matches!(
        &written.event,
        LedgerEvent::SemanticDeltaCommitted {
            origin: SemanticOrigin::Host { principal, .. },
            ..
        } if principal == "pricing-agent"
    ));
    assert_eq!(runtime.merged_at(&BranchId::from("b1")), None);
    assert!(!marks_a_merge(&runtime));
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));

    // Merging the branch now certifies it against a price it did not read.
    let events = runtime.committed_events().len();
    assert_eq!(
        runtime.merge_branch(&sealed, auto()),
        Err(RuntimeError::Certification(BranchError::Conflict {
            keys: BTreeSet::from(["price:sku-1".to_owned()]),
        }))
    );
    assert_eq!(runtime.committed_events().len(), events);
}

#[test]
fn a_revocation_between_sealing_and_merging_stops_the_branch() {
    let (mut runtime, watch) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "pricing-policy".into(),
            generation: Generation(1),
        })
        .unwrap();
    let events = runtime.committed_events().len();
    let judged = watch.count();

    let stale = RuntimeError::Certification(BranchError::LifecycleChanged {
        targets: BTreeSet::from(["pricing-policy".to_owned()]),
    });
    assert_eq!(runtime.preview_merge(&sealed), Err(stale.clone()));
    assert_eq!(runtime.merge_branch(&sealed, auto()), Err(stale));
    assert_eq!(watch.count(), judged);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("10"));
}

#[test]
fn a_plan_certified_before_another_commit_is_refused_by_the_runtime() {
    // A reviewer approves the plan a preview showed. Another commit moves
    // the revision, so the plan certified at merge time is another plan, and
    // the approval does not reach it.
    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let approved = runtime.preview_merge(&sealed).unwrap();
    operator_writes(&mut runtime, put("unrelated", "x"));

    let current = runtime.preview_merge(&sealed).unwrap();
    assert_eq!(current.revision, runtime.revision());
    assert_ne!(current.plan_digest, approved.plan_digest);
    let events = runtime.committed_events().len();
    assert_eq!(
        runtime.merge_branch(&sealed, reviewed(approved.plan_digest)),
        Err(RuntimeError::MergePlanChanged {
            approved: approved.plan_digest,
            current: current.plan_digest,
        })
    );
    assert_eq!(runtime.committed_events().len(), events);
    assert!(unfenced(&runtime));

    // An approval of the plan certified now commits it.
    committed(
        runtime
            .merge_branch(&sealed, reviewed(current.plan_digest))
            .unwrap(),
    );
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn a_merge_records_branch_author_seal_plan_rebased_keys_verifiers_level_and_authority() {
    let (mut runtime, watch) = runtime_watching_prices();
    let sealed = restocking(&runtime, "b1", 3);
    // Another writer changes the stock after the branch's base: the addition
    // is rebased onto the value the runtime holds.
    operator_writes(&mut runtime, put("stock", counter_value(7)));
    let mut noted = passing();
    noted.findings.push(Finding {
        code: "stale-cache".into(),
        message: "the price cache is older than a day".into(),
        hard: false,
    });
    watch.set(noted);

    let preview = runtime.preview_merge(&sealed).unwrap();
    assert_eq!(preview.certification.kind(), CertificationKind::Rebased);
    let receipt = committed(runtime.merge_branch(&sealed, auto()).unwrap());
    assert_eq!(
        runtime
            .snapshot()
            .value("stock")
            .and_then(read_counter_value),
        Some(10)
    );

    let index = receipt.commit.commit_index.unwrap();
    let (base, encoded, record) = merge_at(&runtime, index);
    let plan = preview.certification.plan();
    let triage = MergeAuthorityRecord::Triage {
        policy_version: POLICY.into(),
        score_bits: 0.9_f32.to_bits(),
    };
    assert_eq!(
        record,
        MergeRecord {
            branch: "b1".into(),
            author: "pricing-agent".into(),
            seal: sealed.seal_digest().unwrap(),
            plan: preview.plan_digest,
            dependencies: *plan.dependencies(),
            rebased: BTreeSet::from(["stock".to_owned()]),
            verification: Attestation {
                required: VerificationLevel::Deterministic,
                level: VerificationLevel::Deterministic,
                verifiers: vec!["price-watch".into()],
                findings: vec!["price-watch/stale-cache".into()],
            },
            authority: triage.clone(),
        }
    );
    // Whoever holds the record recomputes the digest of the plan it merged.
    assert_eq!(base, preview.revision);
    assert_eq!(
        merge_plan_digest(
            &record.branch,
            base,
            &encoded,
            &record.dependencies,
            &record.rebased
        ),
        record.plan
    );
    assert_eq!(encoded, plan.delta().encode().unwrap());
    // The receipt names what the record names.
    assert_eq!(receipt.branch, BranchId::from("b1"));
    assert_eq!(receipt.seal_digest, record.seal);
    assert_eq!(receipt.plan_digest, record.plan);
    assert_eq!(receipt.kind, CertificationKind::Rebased);
    assert_eq!(receipt.rebased, record.rebased);
    assert_eq!(receipt.authority, triage);
    assert_eq!(receipt.policy_version.as_deref(), Some(POLICY));
    assert_eq!(receipt.verdict, preview.verdict);

    // Under a reviewer's approval the record names the reviewer, and there
    // is no triage.
    watch.set(passing());
    let sealed = repricing(&runtime, "b2");
    let preview = runtime.preview_merge(&sealed).unwrap();
    let receipt = committed(
        runtime
            .merge_branch(&sealed, reviewed(preview.plan_digest))
            .unwrap(),
    );
    let (_, _, record) = merge_at(&runtime, receipt.commit.commit_index.unwrap());
    assert_eq!(
        record.authority,
        MergeAuthorityRecord::Reviewed {
            reviewer: "reviewer-1".into()
        }
    );
    assert_eq!((record.branch.as_str(), record.rebased.len()), ("b2", 0));
    assert_eq!(record.verification.findings, Vec::<String>::new());
    assert_eq!((receipt.triage, receipt.policy_version), (None, None));
}

#[test]
fn a_redelivered_branch_is_refused_after_merge_replay_reopen_and_compaction() {
    let temp = Temp::new();
    let watch = PriceWatch::default();
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    set_up(&mut runtime, &watch);
    let sealed = repricing(&runtime, "b1");
    // A snapshot from before the merge, whose floor the merge is above.
    let early = runtime.export_compacted_snapshot().unwrap();
    let floor = runtime.committed_events().len();

    let at = committed(runtime.merge_branch(&sealed, auto()).unwrap())
        .commit
        .commit_index
        .unwrap();
    let refused = RuntimeError::BranchAlreadyMerged {
        branch: "b1".into(),
        at,
    };
    let events = runtime.committed_events().len();
    let judged = watch.count();
    assert_eq!(runtime.merge_branch(&sealed, auto()), Err(refused.clone()));
    assert_eq!(runtime.preview_merge(&sealed), Err(refused.clone()));
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(watch.count(), judged);

    let history: Vec<CommittedEvent> = runtime.committed_events().to_vec();
    let late = runtime.export_compacted_snapshot().unwrap();
    drop(runtime);
    // A runtime rebuilt from that history knows the branch is merged,
    // wherever the merge record sits.
    let rebuilt = [
        (
            "replay",
            PtrRuntime::replay(PtrConfig::default(), &history).unwrap(),
        ),
        (
            "open_durable",
            PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap(),
        ),
        (
            "restore_compacted, merge below the floor",
            PtrRuntime::restore_compacted(PtrConfig::default(), late.bytes(), late.anchor(), &[])
                .unwrap(),
        ),
        (
            "restore_compacted, merge above the floor",
            PtrRuntime::restore_compacted(
                PtrConfig::default(),
                early.bytes(),
                early.anchor(),
                &history[floor..],
            )
            .unwrap(),
        ),
    ];
    for (path, mut runtime) in rebuilt {
        assert_eq!(runtime.merged_at(&BranchId::from("b1")), Some(at), "{path}");
        // A grant is not history: the host installs it again.
        runtime.install_semantic_grant(grant(&watch)).unwrap();
        let events = runtime.committed_events().len();
        assert_eq!(
            runtime.merge_branch(&sealed, auto()),
            Err(refused.clone()),
            "{path}"
        );
        assert_eq!(runtime.committed_events().len(), events, "{path}");
    }
}

#[test]
fn a_branch_opened_over_a_foreign_host_is_recertified_here_and_conflicts() {
    let (mut runtime, watch) = runtime_watching_prices();
    let (mut foreign, _) = runtime_watching_prices();
    // Both hosts are at the same revision, with different prices.
    operator_writes(&mut runtime, put("price:sku-1", "12"));
    operator_writes(&mut foreign, put("price:sku-1", "99"));
    assert_eq!(runtime.revision(), foreign.revision());
    let sealed = repricing(&foreign, "b1");
    // A branch opened over the other host after one more commit there has a
    // base this runtime has not reached.
    operator_writes(&mut foreign, put("unrelated", "x"));
    let ahead = repricing(&foreign, "b2");
    let events = runtime.committed_events().len();
    let judged = watch.count();

    assert_eq!(
        runtime.merge_branch(&sealed, auto()),
        Err(RuntimeError::Certification(BranchError::Conflict {
            keys: BTreeSet::from(["price:sku-1".to_owned()]),
        }))
    );
    assert_eq!(
        runtime.merge_branch(&ahead, auto()),
        Err(RuntimeError::Certification(
            BranchError::SnapshotBehindBase {
                base: foreign.revision(),
                snapshot: runtime.revision(),
            }
        ))
    );
    assert_eq!(watch.count(), judged);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("12"));
}

#[test]
fn a_revocation_between_preview_and_merge_refuses_the_merge() {
    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let preview = runtime.preview_merge(&sealed).unwrap();
    assert!(preview.verdict.admitted());
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "pricing-policy".into(),
            generation: Generation(1),
        })
        .unwrap();
    let events = runtime.committed_events().len();
    assert_eq!(
        runtime.merge_branch(&sealed, reviewed(preview.plan_digest)),
        Err(RuntimeError::Certification(BranchError::LifecycleChanged {
            targets: BTreeSet::from(["pricing-policy".to_owned()]),
        }))
    );
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("10"));
    assert!(unfenced(&runtime));
}

#[test]
fn a_plan_cannot_be_changed_between_review_and_merge() {
    // The digest a reviewer approves covers the branch id with the delta:
    // an approval of one branch's plan does not merge another branch that
    // proposes the same delta against the same revision.
    let (mut runtime, _) = runtime_watching_prices();
    let first = repricing(&runtime, "b1");
    let second = repricing(&runtime, "b2");
    let approved = runtime.preview_merge(&first).unwrap();
    let other = runtime.preview_merge(&second).unwrap();
    assert_eq!(
        approved.certification.plan().delta(),
        other.certification.plan().delta()
    );
    assert_ne!(approved.plan_digest, other.plan_digest);
    let events = runtime.committed_events().len();
    assert_eq!(
        runtime.merge_branch(&second, reviewed(approved.plan_digest)),
        Err(RuntimeError::MergePlanChanged {
            approved: approved.plan_digest,
            current: other.plan_digest,
        })
    );
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.merged_at(&BranchId::from("b2")), None);

    // The approved branch merges under its approval.
    committed(
        runtime
            .merge_branch(&first, reviewed(approved.plan_digest))
            .unwrap(),
    );
}

#[test]
fn a_reserved_write_is_refused_by_the_constructor_certification_and_the_runtime() {
    let (runtime, _) = runtime_watching_prices();
    let raw = request_raw_key(&RequestId::from("r1"));
    let output = pod_output_key(&RequestId::from("r1"), &PodId::from("echo"));

    // Staging refuses a write to a key only ingress writes.
    let mut work = Branch::open(BranchId::from("b1"), agent(), runtime.snapshot());
    assert_eq!(
        work.put(&raw, "forged".into()),
        Err(BranchError::ReservedNamespace { key: raw.clone() })
    );
    assert_eq!(
        work.stage_commutative(BranchOp::Add {
            key: output.clone(),
            amount: 1,
        }),
        Err(BranchError::ReservedNamespace {
            key: output.clone()
        })
    );

    // A sealed branch rebuilt from parts that write one is refused by the
    // constructor, whose checks certification, and so every merge, runs
    // again.
    let mut parts = repricing(&runtime, "b1").into_parts();
    parts.ops.push(BranchOp::Remove { key: raw.clone() });
    assert_eq!(
        SealedBranch::from_parts(parts),
        Err(BranchError::ReservedNamespace { key: raw.clone() })
    );

    // And a merge record that writes one is refused wherever a runtime
    // rebuilds from it, though its plan digest is its delta's.
    let delta = put(&raw, "forged");
    let encoded = delta.encode().unwrap();
    let forged = LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(0),
        revision: Revision(1),
        encoded_delta: encoded.clone(),
        origin: SemanticOrigin::Merge(MergeRecord {
            branch: "b1".into(),
            author: "pricing-agent".into(),
            seal: [1; 32],
            plan: merge_plan_digest("b1", Revision(0), &encoded, &[3; 32], &BTreeSet::new()),
            dependencies: [3; 32],
            rebased: BTreeSet::new(),
            verification: Attestation {
                required: VerificationLevel::Deterministic,
                level: VerificationLevel::Deterministic,
                verifiers: vec!["price-watch".into()],
                findings: vec![],
            },
            authority: MergeAuthorityRecord::Reviewed {
                reviewer: "reviewer-1".into(),
            },
        }),
    };
    assert_eq!(
        PtrRuntime::replay(
            PtrConfig::default(),
            &[CommittedEvent {
                index: CommitIndex(1),
                event: forged,
            }],
        )
        .err(),
        Some(RuntimeError::InvalidSemanticOrigin {
            index: Some(CommitIndex(1)),
            reason: "a merge writes, removes or derives an ingress key",
        })
    );
}

#[test]
fn a_merge_ledger_record_carries_its_provenance() {
    let temp = Temp::new();
    let watch = PriceWatch::default();
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    set_up(&mut runtime, &watch);
    let sealed = restocking(&runtime, "b1", 2);
    operator_writes(&mut runtime, put("stock", counter_value(9)));
    let receipt = committed(runtime.merge_branch(&sealed, auto()).unwrap());
    let index = receipt.commit.commit_index.unwrap();
    let written = merge_at(&runtime, index);
    drop(runtime);

    // The record reads back from the durable log exactly as it was written,
    // and the reopened runtime projects the branch's merge from it.
    let reopened = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    assert_eq!(merge_at(&reopened, index), written);
    let (_, _, record) = written;
    assert_eq!(record.plan, receipt.plan_digest);
    assert_eq!(record.seal, sealed.seal_digest().unwrap());
    assert_eq!(record.rebased, BTreeSet::from(["stock".to_owned()]));
    let entry = reopened
        .materialized_state()
        .values
        .get(&ptr_state::merged_branch_key("b1"))
        .cloned();
    assert_eq!(
        entry,
        Some(ptr_state::merged_branch_entry(index.0, &record.plan))
    );
    assert_eq!(
        entry
            .as_deref()
            .and_then(ptr_state::parse_merged_branch_entry),
        Some((index.0, record.plan))
    );
    assert_eq!(
        reopened
            .materialized_state()
            .values
            .get(ptr_state::ATTESTED_MARKER)
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(reopened.merged_at(&BranchId::from("b1")), Some(index));
}

#[test]
fn a_raw_semantic_record_cannot_be_committed() {
    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let receipt = committed(runtime.merge_branch(&sealed, auto()).unwrap());
    let (_, _, record) = merge_at(&runtime, receipt.commit.commit_index.unwrap());

    // The record a merge wrote, as another branch's, and records of every
    // other origin: `commit` takes none of them.
    let encoded = put("price:sku-1", "12").encode().unwrap();
    let base = runtime.revision();
    let raw = |origin| LedgerEvent::SemanticDeltaCommitted {
        base_revision: base,
        revision: Revision(base.0 + 1),
        encoded_delta: encoded.clone(),
        origin,
    };
    let events = runtime.committed_events().len();
    for origin in [
        SemanticOrigin::Merge(MergeRecord {
            branch: "b2".into(),
            plan: merge_plan_digest("b2", base, &encoded, &record.dependencies, &record.rebased),
            ..record.clone()
        }),
        SemanticOrigin::Host {
            principal: "pipeline-operator".into(),
            verification: record.verification.clone(),
        },
        SemanticOrigin::Request {
            request: "r1".into(),
        },
        SemanticOrigin::Legacy,
    ] {
        assert_eq!(
            runtime.commit(raw(origin)),
            Err(RuntimeError::SemanticRecordOutsideSemanticPath)
        );
    }
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.merged_at(&BranchId::from("b2")), None);
    assert!(unfenced(&runtime));
}

#[test]
fn a_reviewed_approval_is_void_once_the_plan_changes() {
    let (mut runtime, _) = runtime_watching_prices();
    let sealed = restocking(&runtime, "b1", 3);
    let approved = runtime.preview_merge(&sealed).unwrap();
    assert_eq!(approved.certification.kind(), CertificationKind::Clean);

    // Another writer changes the stock: the branch still certifies, rebased,
    // but its plan now publishes another value and rebases a key.
    operator_writes(&mut runtime, put("stock", counter_value(7)));
    let current = runtime.preview_merge(&sealed).unwrap();
    assert_eq!(current.certification.kind(), CertificationKind::Rebased);
    assert_ne!(
        current.certification.plan().delta(),
        approved.certification.plan().delta()
    );
    let events = runtime.committed_events().len();
    assert_eq!(
        runtime.merge_branch(&sealed, reviewed(approved.plan_digest)),
        Err(RuntimeError::MergePlanChanged {
            approved: approved.plan_digest,
            current: current.plan_digest,
        })
    );
    assert_eq!(runtime.committed_events().len(), events);

    // Approving the plan as it is now merges it.
    committed(
        runtime
            .merge_branch(&sealed, reviewed(current.plan_digest))
            .unwrap(),
    );
    assert_eq!(
        runtime
            .snapshot()
            .value("stock")
            .and_then(read_counter_value),
        Some(10)
    );
}

#[test]
fn a_reviewer_the_grant_does_not_list_is_refused() {
    let (mut runtime, watch) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let preview = runtime.preview_merge(&sealed).unwrap();
    let judged = watch.count();
    let events = runtime.committed_events().len();
    for stranger in ["stranger", " reviewer-1", "reviewer-1 ", "Reviewer-1"] {
        assert_eq!(
            runtime.merge_branch(
                &sealed,
                MergeAuthority::Reviewed {
                    plan_digest: preview.plan_digest,
                    reviewer: PrincipalId::from(stranger),
                },
            ),
            Err(RuntimeError::UnknownReviewer {
                reviewer: stranger.into()
            })
        );
    }
    // Refused before certification or verification, with nothing appended.
    assert_eq!(watch.count(), judged);
    assert_eq!(runtime.committed_events().len(), events);
    assert!(unfenced(&runtime));
}

#[test]
fn an_escalated_or_calibration_slice_branch_appends_nothing_and_carries_its_triage() {
    // Below the threshold: escalated, eligible, outside the slice.
    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let events = runtime.committed_events().len();
    let hold = held(runtime.merge_branch(&sealed, triaged(0.5)).unwrap());
    assert_eq!(hold.reason, HoldReason::Escalated);
    assert!(hold.preview.verdict.admitted());
    assert_eq!(hold.policy_version.as_deref(), Some(POLICY));
    let triage = hold.triage.clone().unwrap();
    assert_eq!(triage.decision(), TriageDecision::Escalate);
    assert!(triage.eligible() && !triage.calibration_slice());
    assert_eq!(triage.auto_propensity(), 0.0);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.merged_at(&BranchId::from("b1")), None);
    assert!(unfenced(&runtime));
    // The person it was escalated to approves the plan the hold shows.
    committed(
        runtime
            .merge_branch(&sealed, reviewed(hold.preview.plan_digest))
            .unwrap(),
    );

    // Above the threshold, a branch whose draw falls in the calibration
    // slice is escalated too, and one whose draw does not is auto-proposed.
    let watch = PriceWatch::default();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(grant_with(&watch, policy(0.5)))
        .unwrap();
    let draw = |id: &str| calibration_draw(&BranchId::from(id), SEED);
    let ids: Vec<String> = (0..64).map(|n| format!("c{n}")).collect();
    let in_slice = ids.iter().find(|id| draw(id.as_str()) < 0.5).unwrap();
    let outside = ids.iter().find(|id| draw(id.as_str()) >= 0.5).unwrap();
    let noting = |runtime: &PtrRuntime, id: &str| {
        let mut work = Branch::open(BranchId::from(id), agent(), runtime.snapshot());
        work.read(&format!("note:{id}")).unwrap();
        work.put(&format!("note:{id}"), "x".into()).unwrap();
        work.seal().unwrap()
    };
    let events = runtime.committed_events().len();
    let hold = held(
        runtime
            .merge_branch(&noting(&runtime, in_slice.as_str()), triaged(0.95))
            .unwrap(),
    );
    assert_eq!(hold.reason, HoldReason::Escalated);
    let triage = hold.triage.unwrap();
    assert_eq!(triage.decision(), TriageDecision::Escalate);
    assert!(triage.eligible() && triage.calibration_slice());
    assert_eq!(triage.auto_propensity(), 0.5);
    assert_eq!(runtime.committed_events().len(), events);
    let receipt = committed(
        runtime
            .merge_branch(&noting(&runtime, outside.as_str()), triaged(0.95))
            .unwrap(),
    );
    let triage = receipt.triage.unwrap();
    assert_eq!(triage.decision(), TriageDecision::AutoPropose);
    assert!(triage.eligible() && !triage.calibration_slice());
}

#[test]
fn a_failed_verification_holds_the_branch_under_either_authority() {
    let hard = {
        let mut report = passing();
        report.findings.push(Finding {
            code: "below-cost".into(),
            message: "the price is below cost".into(),
            hard: true,
        });
        report
    };
    // What the verifier reports, and what the policy's triage of the hold
    // is: discarded when the verifier failed the change, escalated
    // otherwise, as a report verification decided. A pass at a level the
    // policy would accept but the grant does not is decided by the grant.
    for (verdict, decision) in [
        (
            report(VerificationStatus::Fail, VerificationLevel::Deterministic),
            TriageDecision::Discard,
        ),
        (
            report(
                VerificationStatus::Disputed,
                VerificationLevel::Deterministic,
            ),
            TriageDecision::Escalate,
        ),
        (
            report(VerificationStatus::Pass, VerificationLevel::FullSemantic),
            TriageDecision::Escalate,
        ),
        (hard, TriageDecision::Escalate),
    ] {
        let (mut runtime, watch) = runtime_watching_prices();
        let sealed = repricing(&runtime, "b1");
        watch.set(verdict.clone());
        let events = runtime.committed_events().len();

        let hold = held(runtime.merge_branch(&sealed, auto()).unwrap());
        assert_eq!(hold.reason, HoldReason::VerificationRejected, "{verdict:?}");
        assert!(!hold.preview.verdict.admitted());
        let triage = hold.triage.unwrap();
        assert_eq!(triage.decision(), decision, "{verdict:?}");
        assert!(!triage.eligible() && !triage.calibration_slice());
        assert_eq!(triage.auto_propensity(), 0.0);
        assert!(policy(0.0).policy().explains(&triage).is_ok());

        // No approval overrides verification.
        let hold = held(
            runtime
                .merge_branch(&sealed, reviewed(hold.preview.plan_digest))
                .unwrap(),
        );
        assert_eq!(hold.reason, HoldReason::VerificationRejected);
        assert_eq!((hold.triage, hold.policy_version), (None, None));

        assert_eq!(runtime.committed_events().len(), events);
        assert_eq!(runtime.snapshot().get("price:sku-1"), Some("10"));
        assert_eq!(runtime.merged_at(&BranchId::from("b1")), None);
        assert!(unfenced(&runtime));
    }
}

#[test]
fn the_runtime_triage_is_explained_by_the_granted_policy() {
    let record = policy(0.3);
    let watch = PriceWatch::default();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(grant_with(&watch, record.clone()))
        .unwrap();
    let mut seen: BTreeSet<(TriageDecision, bool, bool)> = BTreeSet::new();
    for n in 0..60 {
        let id = format!("t{n}");
        let mut work = Branch::open(BranchId::from(id.as_str()), agent(), runtime.snapshot());
        work.read(&format!("note:{id}")).unwrap();
        work.put(&format!("note:{id}"), "x".into()).unwrap();
        let sealed = work.seal().unwrap();
        let score = [0.5_f32, 0.85, 0.95][n % 3];
        let failing = n % 7 == 0;
        watch.set(if failing {
            report(VerificationStatus::Fail, VerificationLevel::Deterministic)
        } else {
            passing()
        });
        let preview = runtime.preview_merge(&sealed).unwrap();
        let triage: TriageOutcome = match runtime.merge_branch(&sealed, triaged(score)).unwrap() {
            MergeOutcome::Committed(receipt) => receipt.triage.unwrap(),
            MergeOutcome::Held(hold) => hold.triage.unwrap(),
            MergeOutcome::NoChange(preview) => panic!("{preview:?}"),
        };
        // Every triage the runtime makes is one its policy explains, and
        // for a change the verifiers admitted it is exactly the policy's own
        // triage of the verdict, the score and the branch's draw.
        record.policy().explains(&triage).unwrap();
        if !failing {
            assert_eq!(
                triage,
                record
                    .policy()
                    .triage(
                        &preview.verdict.report(),
                        Probability::new(score).unwrap(),
                        calibration_draw(sealed.id(), SEED),
                    )
                    .unwrap()
            );
        }
        seen.insert((
            triage.decision(),
            triage.eligible(),
            triage.calibration_slice(),
        ));
    }
    // Every kind of triage occurred.
    for kind in [
        (TriageDecision::AutoPropose, true, false),
        (TriageDecision::Escalate, true, true),
        (TriageDecision::Escalate, true, false),
        (TriageDecision::Discard, false, false),
    ] {
        assert!(seen.contains(&kind), "{kind:?} never occurred: {seen:?}");
    }
}

#[test]
fn a_no_op_merge_appends_nothing_and_does_not_mark_the_branch_merged() {
    let (mut runtime, watch) = runtime_watching_prices();
    let mut work = Branch::open(BranchId::from("b1"), agent(), runtime.snapshot());
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "10".into()).unwrap();
    let unchanged = work.seal().unwrap();
    let events = runtime.committed_events().len();
    let judged = watch.count();

    for authority in [
        auto(),
        reviewed(runtime.preview_merge(&unchanged).unwrap().plan_digest),
    ] {
        let MergeOutcome::NoChange(preview) = runtime.merge_branch(&unchanged, authority).unwrap()
        else {
            panic!("a merge that changes nothing is no change");
        };
        // It was verified and admitted.
        assert!(preview.verdict.admitted());
        assert!(preview.certification.plan().is_noop());
    }
    assert!(watch.count() > judged);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.merged_at(&BranchId::from("b1")), None);
    assert!(!marks_a_merge(&runtime));

    // So a later merge of the id that changes something is its first.
    committed(
        runtime
            .merge_branch(&repricing(&runtime, "b1"), auto())
            .unwrap(),
    );
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn every_merge_refusal_leaves_the_runtime_unfenced() {
    let watch = PriceWatch::default();
    let bare = || PtrRuntime::new(PtrConfig::default()).unwrap();
    let mut cases: Vec<(PtrRuntime, SealedBranch, MergeAuthority, RuntimeError)> = Vec::new();

    let runtime = bare();
    let sealed = repricing(&runtime, "b1");
    cases.push((runtime, sealed, auto(), RuntimeError::NoSemanticGrant));

    let mut runtime = bare();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic).with_verifier(watch.clone()),
        )
        .unwrap();
    let sealed = repricing(&runtime, "b1");
    cases.push((runtime, sealed, auto(), RuntimeError::NoMergePolicy));

    let (runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    cases.push((
        runtime,
        sealed,
        MergeAuthority::Reviewed {
            plan_digest: [0; 32],
            reviewer: PrincipalId::from("stranger"),
        },
        RuntimeError::UnknownReviewer {
            reviewer: "stranger".into(),
        },
    ));

    let (runtime, _) = runtime_watching_prices();
    let mut work = Branch::open(
        BranchId::from("b1"),
        PrincipalId::from(" pricing-agent"),
        runtime.snapshot(),
    );
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    cases.push((
        runtime,
        work.seal().unwrap(),
        auto(),
        RuntimeError::InvalidProvenanceText { field: "author" },
    ));

    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let at = committed(runtime.merge_branch(&sealed, auto()).unwrap())
        .commit
        .commit_index
        .unwrap();
    cases.push((
        runtime,
        sealed,
        auto(),
        RuntimeError::BranchAlreadyMerged {
            branch: "b1".into(),
            at,
        },
    ));

    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    operator_writes(&mut runtime, put("price:sku-1", "12"));
    cases.push((
        runtime,
        sealed,
        auto(),
        RuntimeError::Certification(BranchError::Conflict {
            keys: BTreeSet::from(["price:sku-1".to_owned()]),
        }),
    ));

    let (mut runtime, _) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let approved = runtime.preview_merge(&sealed).unwrap().plan_digest;
    operator_writes(&mut runtime, put("unrelated", "x"));
    let current = runtime.preview_merge(&sealed).unwrap().plan_digest;
    cases.push((
        runtime,
        sealed,
        reviewed(approved),
        RuntimeError::MergePlanChanged { approved, current },
    ));

    let (runtime, watch) = runtime_watching_prices();
    let sealed = repricing(&runtime, "b1");
    let mut invalid = passing();
    invalid.findings.push(Finding {
        code: " padded".into(),
        message: "a code that is not an identifier".into(),
        hard: false,
    });
    watch.set(invalid);
    cases.push((
        runtime,
        sealed,
        auto(),
        RuntimeError::InvalidVerificationReport {
            verifier: "price-watch".into(),
        },
    ));

    for (mut runtime, sealed, authority, refusal) in cases {
        let events = runtime.committed_events().len();
        let revision = runtime.revision();
        assert_eq!(
            runtime.merge_branch(&sealed, authority),
            Err(refusal.clone())
        );
        assert_eq!(runtime.committed_events().len(), events, "{refusal:?}");
        assert_eq!(runtime.revision(), revision, "{refusal:?}");
        assert!(unfenced(&runtime), "{refusal:?}");
    }
}

#[test]
fn a_merge_whose_record_would_exceed_the_bound_is_refused_unfenced() {
    // A counter the branch adds nothing to, but that another writer changed
    // after its base, is rebased and named by the record while the plan
    // writes nothing to it. So many long keys of that kind make a record
    // larger than the ledger frames, though the plan's delta is one small
    // write.
    const KEYS: usize = 2_100;
    const CHUNK: usize = 700;
    let (mut runtime, watch) = runtime_watching_prices();
    let keys: Vec<String> = (0..KEYS)
        .map(|n| format!("{n:04}{}", "k".repeat(3_996)))
        .collect();
    for chunk in keys.chunks(CHUNK) {
        let mut delta = SemanticDelta::default();
        for key in chunk {
            delta.upserts.insert(key.clone(), counter_value(0));
        }
        operator_writes(&mut runtime, delta);
    }
    let mut work = Branch::open(BranchId::from("b1"), agent(), runtime.snapshot());
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    for key in &keys {
        work.stage_commutative(BranchOp::Add {
            key: key.clone(),
            amount: 0,
        })
        .unwrap();
    }
    let sealed = work.seal().unwrap();
    for chunk in keys.chunks(CHUNK) {
        let mut delta = SemanticDelta::default();
        for key in chunk {
            delta.upserts.insert(key.clone(), counter_value(1));
        }
        operator_writes(&mut runtime, delta);
    }
    let preview = runtime.preview_merge(&sealed).unwrap();
    assert_eq!(preview.certification.plan().rebased().len(), KEYS);
    assert_eq!(preview.certification.plan().delta().upserts.len(), 1);

    let events = runtime.committed_events().len();
    let judged = watch.count();
    assert_eq!(
        runtime.merge_branch(&sealed, auto()),
        Err(RuntimeError::MergeRecordTooLarge)
    );
    // Refused after the verifiers admitted it, with nothing appended and the
    // runtime unfenced.
    assert_eq!(watch.count(), judged + 1);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.merged_at(&BranchId::from("b1")), None);
    assert!(unfenced(&runtime));

    // The same kind of branch over fewer keys merges.
    let mut work = Branch::open(BranchId::from("b2"), agent(), runtime.snapshot());
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "12".into()).unwrap();
    for key in &keys[..10] {
        work.stage_commutative(BranchOp::Add {
            key: key.clone(),
            amount: 0,
        })
        .unwrap();
    }
    let sealed = work.seal().unwrap();
    let mut delta = SemanticDelta::default();
    for key in &keys[..10] {
        delta.upserts.insert(key.clone(), counter_value(2));
    }
    operator_writes(&mut runtime, delta);
    let receipt = committed(runtime.merge_branch(&sealed, auto()).unwrap());
    assert_eq!(receipt.rebased.len(), 10);
}

#[test]
fn a_branch_with_an_empty_or_padded_author_is_refused() {
    let long = "a".repeat(ptr_runtime::merge::MAX_PROVENANCE_TEXT + 1);
    let fits = "a".repeat(ptr_runtime::merge::MAX_PROVENANCE_TEXT);
    let (mut runtime, watch) = runtime_watching_prices();
    let events = runtime.committed_events().len();
    let judged = watch.count();
    let sealed = |runtime: &PtrRuntime, id: &str, author: &str| {
        let mut work = Branch::open(
            BranchId::from(id),
            PrincipalId::from(author),
            runtime.snapshot(),
        );
        work.read("price:sku-1").unwrap();
        work.put("price:sku-1", "11".into()).unwrap();
        work.seal().unwrap()
    };
    let mut cases = Vec::new();
    for author in [
        "",
        " pricing-agent",
        "pricing-agent ",
        "pricing\u{7}agent",
        long.as_str(),
    ] {
        cases.push((sealed(&runtime, "b1", author), "author"));
    }
    for id in ["", " b1", "b1\n", "b\u{0}1", long.as_str()] {
        cases.push((sealed(&runtime, id, "pricing-agent"), "branch"));
    }
    // An id that is not provenance text is refused before its author.
    cases.push((sealed(&runtime, "", ""), "branch"));
    for (branch, field) in cases {
        let refused = RuntimeError::InvalidProvenanceText { field };
        assert_eq!(runtime.preview_merge(&branch), Err(refused.clone()));
        assert_eq!(runtime.merge_branch(&branch, auto()), Err(refused));
    }
    assert_eq!(watch.count(), judged);
    assert_eq!(runtime.committed_events().len(), events);
    assert!(unfenced(&runtime));

    // Provenance text of exactly the bound is accepted.
    let receipt = committed(
        runtime
            .merge_branch(&sealed(&runtime, &fits, &fits), auto())
            .unwrap(),
    );
    let (_, _, record) = merge_at(&runtime, receipt.commit.commit_index.unwrap());
    assert_eq!((record.branch, record.author), (fits.clone(), fits));
}

#[test]
fn a_merge_that_would_evict_an_ingress_key_is_refused() {
    // Before origins existed any delta could be committed, so a history may
    // hold a Pod's output derived from a key a branch writes. Merging the
    // branch would evict the output, a key only ingress writes.
    let output = pod_output_key(&RequestId::from("r1"), &PodId::from("echo"));
    let mut legacy = SemanticDelta::default();
    legacy.upserts.insert("price:sku-1".into(), "10".into());
    legacy.upserts.insert(
        output.clone(),
        SemanticValue::Payload(ptr_semdb::SemanticPayload {
            type_id: "bytes".into(),
            source: "echo".into(),
            bytes: vec![1],
        }),
    );
    legacy
        .dependencies
        .insert(output.clone(), ["price:sku-1".to_owned()].into());
    let prefix = CommittedEvent {
        index: CommitIndex(1),
        event: LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(0),
            revision: Revision(1),
            encoded_delta: legacy.encode().unwrap(),
            origin: SemanticOrigin::Legacy,
        },
    };
    let watch = PriceWatch::default();
    let mut runtime =
        PtrRuntime::replay(PtrConfig::default(), std::slice::from_ref(&prefix)).unwrap();
    runtime.install_semantic_grant(grant(&watch)).unwrap();
    let mut work = Branch::open(BranchId::from("b1"), agent(), runtime.snapshot());
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    let sealed = work.seal().unwrap();

    // Refused, previewed or merged, before any verifier is asked.
    let refused = RuntimeError::ReservedSemanticNamespace {
        key: output.clone(),
    };
    assert_eq!(runtime.preview_merge(&sealed), Err(refused.clone()));
    assert_eq!(runtime.merge_branch(&sealed, auto()), Err(refused));
    assert_eq!(watch.count(), 0);
    assert_eq!(runtime.committed_events().len(), 1);
    assert!(runtime.snapshot().value(&output).is_some());
    assert!(unfenced(&runtime));

    // Replay refuses a merge record that evicts one.
    let mut write = SemanticDelta::default();
    write.upserts.insert("price:sku-1".into(), "11".into());
    let encoded = write.encode().unwrap();
    let evicting = CommittedEvent {
        index: CommitIndex(2),
        event: LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(1),
            revision: Revision(2),
            encoded_delta: encoded.clone(),
            origin: SemanticOrigin::Merge(MergeRecord {
                branch: "b1".into(),
                author: "pricing-agent".into(),
                seal: [1; 32],
                plan: merge_plan_digest("b1", Revision(1), &encoded, &[3; 32], &BTreeSet::new()),
                dependencies: [3; 32],
                rebased: BTreeSet::new(),
                verification: Attestation {
                    required: VerificationLevel::Deterministic,
                    level: VerificationLevel::Deterministic,
                    verifiers: vec!["price-watch".into()],
                    findings: vec![],
                },
                authority: MergeAuthorityRecord::Reviewed {
                    reviewer: "reviewer-1".into(),
                },
            }),
        },
    };
    assert_eq!(
        PtrRuntime::replay(PtrConfig::default(), &[prefix, evicting]).err(),
        Some(RuntimeError::InvalidSemanticOrigin {
            index: Some(CommitIndex(2)),
            reason: "a host write or merge evicts an ingress key",
        })
    );
}

#[test]
fn each_verifier_result_of_a_merge_is_emitted_a_report_that_fails_closed_as_not_passed() {
    let results = |runtime: &PtrRuntime| -> Vec<(String, bool)> {
        runtime
            .events()
            .iter()
            .filter_map(|envelope| match &envelope.event {
                RuntimeEvent::VerifierResult { verifier, passed } => {
                    Some((verifier.clone(), *passed))
                }
                _ => None,
            })
            .collect()
    };
    // A merge the verifier admits, and one it fails: each result is emitted
    // once per merge, not by a preview.
    let (mut runtime, watch) = runtime_watching_prices();
    let before = results(&runtime).len();
    let sealed = repricing(&runtime, "b1");
    runtime.preview_merge(&sealed).unwrap();
    assert_eq!(results(&runtime).len(), before);
    committed(runtime.merge_branch(&sealed, auto()).unwrap());
    watch.set(report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
    ));
    held(
        runtime
            .merge_branch(&repricing(&runtime, "b2"), auto())
            .unwrap(),
    );
    assert_eq!(
        results(&runtime)[before..],
        [
            ("price-watch".to_owned(), true),
            ("price-watch".to_owned(), false)
        ]
    );

    // A report that fails closed is emitted as not passed, and refuses the
    // merge with nothing appended.
    let mut invalid = passing();
    invalid.findings.push(Finding {
        code: " padded".into(),
        message: "a code that is not an identifier".into(),
        hard: false,
    });
    watch.set(invalid);
    let events = runtime.committed_events().len();
    assert_eq!(
        runtime.merge_branch(&repricing(&runtime, "b3"), auto()),
        Err(RuntimeError::InvalidVerificationReport {
            verifier: "price-watch".into()
        })
    );
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(
        results(&runtime).last(),
        Some(&("price-watch".to_owned(), false))
    );
    assert_eq!(results(&runtime).len(), before + 3);
}
