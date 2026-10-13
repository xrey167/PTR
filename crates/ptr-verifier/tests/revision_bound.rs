//! Public-API tests of the revision verifier, snapshot-bound reports and the bound fabric.

use ptr_semdb::{SemanticDelta, SemanticHost};
use ptr_types::{
    ActionIr, CapabilityId, Effect, Generation, Probability, Revision, TypeId, VerificationLevel,
};
use ptr_verifier::{
    fact_key, AtSnapshot, BoundVerificationReport, BoundVerifier, BoundVerifierFabric, FactClaim,
    FactVerifier, Finding, NamedBoundVerifier, NamedVerifier, RevisionClaim, RevisionVerifier,
    SnapshotBinding, VerificationReport, VerificationStatus, Verifier, VerifierFabric,
};

fn at(revision: u64, generation: u64) -> SnapshotBinding {
    SnapshotBinding::new(Revision(revision), Generation(generation))
}

fn action(revision: u64, generation: u64) -> ActionIr {
    ActionIr {
        operation: "create".into(),
        target: "note".into(),
        capability: CapabilityId::from("note.create"),
        effect: Effect::Mutation,
        input_type: TypeId::from("note.v1"),
        generation: Generation(generation),
        revision: Revision(revision),
        payload: b"hello".to_vec(),
    }
}

fn codes(report: &VerificationReport) -> Vec<&str> {
    report.findings.iter().map(|f| f.code.as_str()).collect()
}

#[test]
fn revision_verifier_emits_each_stable_code_with_expected_and_actual() {
    let verifier = RevisionVerifier::new(at(10, 3));
    let cases = [
        (
            action(9, 3),
            "revision.stale",
            "claimed revision 9, expected 10",
        ),
        (
            action(11, 3),
            "revision.future",
            "claimed revision 11, expected 10",
        ),
        (
            action(10, 2),
            "generation.stale",
            "claimed generation 2, expected 3",
        ),
        (
            action(10, 4),
            "generation.future",
            "claimed generation 4, expected 3",
        ),
    ];
    for (candidate, code, message) in cases {
        let report = verifier.verify(&candidate);
        assert_eq!(report.status, VerificationStatus::Fail, "{code}");
        assert_eq!(report.level, VerificationLevel::Deterministic);
        assert_eq!(codes(&report), [code]);
        assert!(report.findings[0].hard);
        assert!(report.findings[0].message.contains(message), "{code}");
        assert!(ptr_verifier::code_spec(code).is_some_and(|spec| spec.hard));
    }
    let missing = verifier.verify(&RevisionClaim::default());
    assert_eq!(codes(&missing), ["binding.missing", "binding.missing"]);
    assert!(ptr_verifier::code_spec("binding.missing").is_some_and(|spec| spec.hard));
    assert_eq!(
        verifier.verify(&action(10, 3)).status,
        VerificationStatus::Pass
    );
}

fn host_with_fact() -> SemanticHost {
    let mut host = SemanticHost::default();
    let mut delta = SemanticDelta::default();
    delta
        .upserts
        .insert(fact_key("paris", "capital-of"), "france".into());
    host.apply_delta(delta).unwrap();
    host
}

#[test]
fn fact_verifier_stays_compatible_and_joins_the_bound_fabric() {
    let host = host_with_fact();
    let snapshot_revision = host.revision();
    let claims = vec![FactClaim::new("paris", "capital-of", "france")];

    // The inherent method is unchanged.
    let plain = FactVerifier::new(host.snapshot(), host.revision());
    let legacy = plain.verify_bound(&claims);
    assert_eq!(legacy.snapshot_revision, snapshot_revision);
    assert_eq!(legacy.report.status, VerificationStatus::Pass);

    // Through the trait the binding also names the generation.
    let fact = FactVerifier::new(host.snapshot(), host.revision()).with_generation(Generation(4));
    let bound = BoundVerifier::verify_at_snapshot(&fact, &claims);
    assert_eq!(
        bound.snapshot,
        SnapshotBinding::new(snapshot_revision, Generation(4))
    );
    assert_eq!(bound.report, legacy.report);
    assert_eq!(
        NamedBoundVerifier::<Vec<FactClaim>>::bound_name(&fact),
        "fact"
    );
    assert_eq!(NamedVerifier::<Vec<FactClaim>>::name(&fact), "fact");

    let mut fabric = BoundVerifierFabric::<Vec<FactClaim>>::new(bound.snapshot);
    fabric.push(fact);
    let result = fabric.verify_at_snapshot(&claims);
    assert_eq!(result.report.status, VerificationStatus::Pass);
    assert_eq!(result.snapshot, bound.snapshot);
}

#[test]
fn bound_fabric_requires_every_verifier_to_agree_on_the_snapshot() {
    let current = at(10, 3);
    let mut fabric = BoundVerifierFabric::<ActionIr>::new(current);
    fabric.push(RevisionVerifier::new(current));
    let ok = fabric.verify_at_snapshot(&action(10, 3));
    assert_eq!(ok.report.status, VerificationStatus::Pass);
    assert_eq!(ok.snapshot, current);

    // A second verifier that judged an older snapshot cannot be merged in.
    fabric.push(RevisionVerifier::new(at(9, 3)));
    let mismatch = fabric.verify_at_snapshot(&action(10, 3));
    assert_eq!(mismatch.report.status, VerificationStatus::Fail);
    assert_eq!(mismatch.snapshot, current);
    let hard: Vec<&Finding> = mismatch.report.findings.iter().filter(|f| f.hard).collect();
    assert!(hard.iter().any(|f| f.code == "snapshot.mismatch"));
    // The older verifier also reports the action as ahead of *its* snapshot.
    assert!(hard.iter().any(|f| f.code == "revision.future"));
    assert!(ptr_verifier::code_spec("snapshot.mismatch").is_some_and(|spec| spec.hard));
}

struct LearnedPositive;
impl Verifier<ActionIr> for LearnedPositive {
    fn verify(&self, _: &ActionIr) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::LatentAgreement,
            score: Probability::new(0.99).unwrap(),
            findings: vec![],
        }
    }
}
impl NamedVerifier<ActionIr> for LearnedPositive {
    fn name(&self) -> &'static str {
        "learned"
    }
}

#[test]
fn a_hard_revision_finding_dominates_a_confident_learned_pass_in_the_plain_fabric() {
    let mut fabric = VerifierFabric::<ActionIr>::default();
    fabric.push(LearnedPositive);
    fabric.push(RevisionVerifier::new(at(10, 3)));
    let report = fabric.verify(&action(7, 3));
    assert_eq!(report.status, VerificationStatus::Fail);
    assert_eq!(report.level, VerificationLevel::LatentAgreement);
    assert_eq!(report.score.get(), 0.0);
    assert!(report
        .findings
        .iter()
        .any(|f| f.code == "revision.stale" && f.hard));
    // The same learned verifier alone passes: only the deterministic finding turns it around.
    let mut only = VerifierFabric::<ActionIr>::default();
    only.push(LearnedPositive);
    assert_eq!(only.verify(&action(7, 3)).status, VerificationStatus::Pass);
}

#[test]
fn plain_verifiers_can_be_lifted_and_mixed_with_native_bound_ones() {
    let current = at(10, 3);
    let mut fabric = BoundVerifierFabric::<ActionIr>::new(current);
    fabric.push(RevisionVerifier::new(current));
    fabric.push(AtSnapshot::new(LearnedPositive, current));
    let result: BoundVerificationReport = fabric.verify_at_snapshot(&action(10, 3));
    assert_eq!(result.report.status, VerificationStatus::Pass);
    assert_eq!(result.report.level, VerificationLevel::LatentAgreement);

    let stale = fabric.verify_at_snapshot(&action(2, 3));
    assert_eq!(stale.report.status, VerificationStatus::Fail);
    assert_eq!(codes(&stale.report), ["revision.stale"]);
}
