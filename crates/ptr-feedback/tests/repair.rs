//! The loop through its public API with the real contract verifier.

use ptr_feedback::{
    Candidate, Critique, Decision, Generator, Outcome, RepairBudget, RepairLoop, RepairPolicy,
    Repairer,
};
use ptr_types::{ActionIr, CapabilityId, Effect, Generation, Revision, TypeId, VerificationLevel};
use ptr_verifier::{ActionContract, ActionContractVerifier, PayloadEncoding, VerifierFabric};

fn contract() -> ActionContract {
    ActionContract {
        target: "note".into(),
        operation: "create".into(),
        capability: CapabilityId::from("note.create"),
        input_type: TypeId::from("note.v1"),
        effect: Effect::Mutation,
        max_payload_bytes: 8,
        payload_encoding: PayloadEncoding::Utf8,
    }
}

fn action(target: &str, payload: &[u8]) -> ActionIr {
    ActionIr {
        operation: "create".into(),
        target: target.into(),
        capability: CapabilityId::from("note.create"),
        effect: Effect::Mutation,
        input_type: TypeId::from("note.v1"),
        generation: Generation(1),
        revision: Revision(0),
        payload: payload.to_vec(),
    }
}

struct Fixed(Vec<ActionIr>);
impl Generator<ActionIr> for Fixed {
    fn generate(&mut self) -> ActionIr {
        self.0.remove(0)
    }
}

/// Fixes the target when the critique names it, whatever else it names.
struct TargetFixer;
impl Repairer<ActionIr> for TargetFixer {
    fn repair(&mut self, previous: &Candidate<ActionIr>, critique: &Critique) -> Option<ActionIr> {
        let mut next = previous.value.clone();
        if critique.issues.iter().any(|issue| issue == "target") {
            next.target = "note".into();
        }
        Some(next)
    }
}

fn fabric() -> VerifierFabric<ActionIr> {
    let mut fabric = VerifierFabric::default();
    fabric.push(ActionContractVerifier::new(contract()));
    fabric
}

fn policy(calls: u32) -> RepairPolicy {
    RepairPolicy {
        required_level: VerificationLevel::Deterministic,
        budget: RepairBudget {
            max_verifier_calls: calls,
            max_rounds: calls,
        },
    }
}

#[test]
fn a_wrong_target_is_repaired_and_the_repair_is_reverified() {
    let fabric = fabric();
    let run = RepairLoop::new(&fabric, policy(4)).unwrap().run(
        &mut Fixed(vec![action("other", b"hello")]),
        &mut TargetFixer,
    );
    let Outcome::Verified(winner) = run.outcome else {
        panic!("the repaired action passes")
    };
    assert_eq!(winner.value.target, "note");
    assert_eq!(run.usage.verifier_calls, 2);
    let first = &run.trajectory.steps[0];
    assert_eq!(first.critique.as_ref().unwrap().issues, ["target"]);
    assert_eq!(first.decision, Decision::Repair);
}

#[test]
fn an_unrepaired_defect_exhausts_the_budget_without_returning_the_failure() {
    let fabric = fabric();
    let long = [b'a'; 9];
    let run = RepairLoop::new(&fabric, policy(3))
        .unwrap()
        .run(&mut Fixed(vec![action("note", &long)]), &mut TargetFixer);
    assert!(matches!(run.outcome, Outcome::Exhausted { .. }));
    assert_eq!(run.usage.verifier_calls, 3);
    assert!(run
        .trajectory
        .steps
        .iter()
        .all(|step| step.critique.as_ref().unwrap().issues == ["payload-size"]));
}
