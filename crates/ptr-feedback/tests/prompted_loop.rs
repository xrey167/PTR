use ptr_feedback::{
    Decision, Outcome, PromptOptions, PromptedGenerator, PromptedRepairer, RepairBudget,
    RepairLoop, RepairPolicy, ScriptedTextModel, TextModelError, TrajectoryRecord,
};
use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};

/// Rejects empty text and any text that contains a defect code, one hard finding per code.
struct DefectVerifier;

impl Verifier<String> for DefectVerifier {
    fn verify(&self, value: &String) -> VerificationReport {
        let findings: Vec<Finding> = ["defect-a", "defect-b"]
            .into_iter()
            .filter(|code| value.contains(code))
            .map(|code| Finding {
                code: code.into(),
                message: String::new(),
                hard: true,
            })
            .collect();
        let clean = findings.is_empty() && !value.is_empty();
        VerificationReport {
            status: if clean {
                VerificationStatus::Pass
            } else {
                VerificationStatus::Fail
            },
            level: VerificationLevel::Deterministic,
            score: Probability::new(if clean { 1.0 } else { 0.0 }).unwrap(),
            findings,
        }
    }
}

fn policy(calls: u32, rounds: u32) -> RepairPolicy {
    RepairPolicy {
        required_level: VerificationLevel::Deterministic,
        budget: RepairBudget {
            max_verifier_calls: calls,
            max_rounds: rounds,
        },
    }
}

fn options() -> PromptOptions {
    PromptOptions {
        max_tokens: 128,
        max_output_chars: 200,
        max_prompt_chars: 2000,
    }
}

#[test]
fn a_scripted_model_drives_generate_repair_and_the_verifier_accepts_the_repair() {
    let generate_model = ScriptedTextModel::from_texts(["answer with defect-a and defect-b"]);
    let repair_model = ScriptedTextModel::from_texts(["answer without problems"]);
    let mut generator =
        PromptedGenerator::<_, String>::new(&generate_model, "Write an answer", options()).unwrap();
    let mut repairer =
        PromptedRepairer::<_, String>::new(&repair_model, "Write an answer", options()).unwrap();
    let run = RepairLoop::new(&DefectVerifier, policy(4, 4))
        .unwrap()
        .run(&mut generator, &mut repairer);

    let Outcome::Verified(winner) = &run.outcome else {
        panic!("the repair should have been verified")
    };
    assert_eq!(winner.value, "answer without problems");
    assert_eq!(run.usage.verifier_calls, 2);
    assert_eq!(run.trajectory.steps[0].decision, Decision::Repair);

    let repair_prompt = &repair_model.prompts()[0].0;
    for needle in [
        "defect-a and defect-b",
        "- defect-a",
        "- defect-b",
        "Write an answer",
    ] {
        assert!(repair_prompt.contains(needle), "{needle}");
    }
    assert_eq!(repair_model.prompts()[0].1, 128);
    assert_eq!(generator.stats().calls, 1);
    assert_eq!(repairer.stats().calls, 1);
}

#[test]
fn a_model_that_never_fixes_the_defect_exhausts_the_budget_and_returns_no_candidate() {
    let generate_model =
        ScriptedTextModel::from_texts(["defect-a one", "defect-a two", "defect-a 3"]);
    let repair_model = ScriptedTextModel::from_texts(["still defect-a"]);
    let mut generator =
        PromptedGenerator::<_, String>::new(&generate_model, "t", options()).unwrap();
    let mut repairer = PromptedRepairer::<_, String>::new(&repair_model, "t", options()).unwrap();
    let run = RepairLoop::new(&DefectVerifier, policy(3, 5))
        .unwrap()
        .run(&mut generator, &mut repairer);
    assert!(matches!(run.outcome, Outcome::Exhausted { .. }));
    assert_eq!(run.usage.verifier_calls, 3);
    // The repair model ran out after one reply, so the second repair is declined and rewritten.
    assert_eq!((run.usage.repaired, run.usage.declined), (1, 1));
}

#[test]
fn model_errors_degrade_to_rejected_candidates_never_to_acceptance() {
    let generate_model = ScriptedTextModel::new([Err(TextModelError::Unavailable)]);
    let repair_model = ScriptedTextModel::default();
    let mut generator =
        PromptedGenerator::<_, String>::new(&generate_model, "t", options()).unwrap();
    let mut repairer = PromptedRepairer::<_, String>::new(&repair_model, "t", options()).unwrap();
    let run = RepairLoop::new(&DefectVerifier, policy(2, 1))
        .unwrap()
        .run(&mut generator, &mut repairer);
    assert!(matches!(run.outcome, Outcome::Exhausted { .. }));
    assert_eq!(generator.stats().model_errors, 2);
}

#[test]
fn a_model_driven_run_converts_to_a_record_and_round_trips() {
    let generate_model = ScriptedTextModel::from_texts(["bad defect-a password=hunter2"]);
    let repair_model = ScriptedTextModel::from_texts(["good"]);
    let mut generator =
        PromptedGenerator::<_, String>::new(&generate_model, "t", options()).unwrap();
    let mut repairer = PromptedRepairer::<_, String>::new(&repair_model, "t", options()).unwrap();
    let run = RepairLoop::new(&DefectVerifier, policy(4, 4))
        .unwrap()
        .run(&mut generator, &mut repairer);
    let record = TrajectoryRecord::from_run("run-7", &run).unwrap();
    let line = record.encode().unwrap();
    assert!(!line.contains("hunter2"));
    assert_eq!(TrajectoryRecord::decode(&line).unwrap(), record);
    assert_eq!(record.steps.len(), 2);
    assert_eq!(record.steps[0].issues, ["defect-a"]);
    assert_eq!(record.steps[1].origin, "repaired");
}
