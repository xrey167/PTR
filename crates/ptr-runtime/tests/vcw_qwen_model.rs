//! Real language-model experts through the verifier-closed loop. Ignored by default: it needs
//! the model directory in `PTR_QWEN_DIR` and the `qwen-cpu` feature, and takes minutes on a CPU.
//!
//! ```text
//! PTR_QWEN_DIR=<dir> cargo test -p ptr-runtime --release --features qwen-cpu \
//!     --test vcw_qwen_model -- --ignored --nocapture
//! ```
//!
//! The test asserts the properties that do not depend on the model's skill (the budgets hold, a
//! `Verified` answer is the verifier's, every call is by a routed, live Pod) and prints the
//! measured proposal accuracy and repair rate, which do depend on it.
#![cfg(feature = "qwen-cpu")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_pods::{PodRegistry, QwenCpu, QwenCpuOptions};
use ptr_router::{
    NonceSequence, RouterConfig, SamplingKey, SelfModel, SelfModelLimits, UniformScorer,
};
use ptr_runtime::expert::{StructurePolicy, StructurePolicyConfig};
use ptr_runtime::vcw::experts::{
    CallOutcome, ExpertRole, ExpertTask, LlmExpertPod, LlmExpertPods, SharedGenerator, TaskKind,
};
use ptr_runtime::vcw::{
    OutcomeTag, RequestBudget, RuntimeStanding, VcwConfig, VcwLoop, VcwOutcome, VcwParts,
    VcwRequest,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{CapsuleId, Generation, PodId, ProjectId, RequestId, VerificationLevel};
use sha2::{Digest as _, Sha256};

#[path = "vcw_experts_support/mod.rs"]
mod support;

const SUMS: [(i64, char, i64); 12] = [
    (17, '+', 25),
    (38, '+', 47),
    (91, '-', 36),
    (123, '+', 456),
    (12, '*', 7),
    (25, '*', 4),
    (48, '*', 12),
    (123, '*', 45),
    (305, '-', 178),
    (999, '+', 1),
    (64, '*', 9),
    (250, '-', 99),
];

const HARD: [(i64, char, i64); 10] = [
    (347, '*', 29),
    (586, '*', 73),
    (912, '*', 48),
    (275, '*', 36),
    (638, '*', 94),
    (419, '*', 57),
    (803, '*', 66),
    (764, '*', 31),
    (529, '*', 82),
    (951, '*', 27),
];

#[test]
#[ignore = "needs PTR_QWEN_DIR and takes minutes"]
fn real_experts_on_arithmetic_with_the_loop_bounds_holding() {
    run_suite(&SUMS);
}

#[test]
#[ignore = "needs PTR_QWEN_DIR and takes minutes"]
fn real_experts_on_hard_multiplications_measure_the_repair_rate() {
    run_suite(&HARD);
}

fn run_suite(tasks: &[(i64, char, i64)]) {
    let dir = PathBuf::from(std::env::var("PTR_QWEN_DIR").expect("PTR_QWEN_DIR"));
    let mut options = QwenCpuOptions::default();
    for (variable, pin) in [
        (
            "PTR_QWEN_SHA256",
            QwenCpuOptions::with_expected_weights_sha256 as fn(_, String) -> _,
        ),
        (
            "PTR_QWEN_CONFIG_SHA256",
            QwenCpuOptions::with_expected_config_sha256,
        ),
        (
            "PTR_QWEN_TOKENIZER_SHA256",
            QwenCpuOptions::with_expected_tokenizer_sha256,
        ),
        (
            "PTR_QWEN_TOKENIZER_CONFIG_SHA256",
            QwenCpuOptions::with_expected_tokenizer_config_sha256,
        ),
    ] {
        if let Ok(sha256) = std::env::var(variable) {
            options = pin(options, sha256);
        }
    }
    let qwen = Arc::new(Mutex::new(QwenCpu::load(&dir, &options).expect("model")));
    let generator: SharedGenerator = qwen.clone();

    let project = ProjectId::from("proj");
    let kind = TaskKind::Arithmetic;
    let make = |name: &str, role| {
        let system = match role {
            ExpertRole::Proposer => kind.proposer_system_prompt(),
            _ => kind.repairer_system_prompt(),
        };
        Arc::new(
            LlmExpertPod::new(
                &project,
                &PodId::from(name),
                role,
                kind,
                system,
                24,
                generator.clone(),
            )
            .expect("bound within 1..=128"),
        )
    };
    let mut experts = LlmExpertPods::new(ExpertTask::new(kind, "unset"));
    experts
        .add_proposer(make("proposer", ExpertRole::Proposer))
        .unwrap();
    experts
        .add_repairer(
            make("repairer_a", ExpertRole::Repairer),
            &support::ARITHMETIC_CODES,
        )
        .unwrap();
    experts
        .add_repairer(
            make("repairer_b", ExpertRole::Repairer),
            &support::ARITHMETIC_CODES,
        )
        .unwrap();

    let names = ["proposer", "repairer_a", "repairer_b"];
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let mut registry = PodRegistry::default();
    for name in names {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: project.clone(),
                capsule: CapsuleId::from(name),
                generation: Generation(1),
            })
            .unwrap();
    }
    for pod in experts.pods() {
        registry.register_new(pod).unwrap();
    }
    let mut standing = RuntimeStanding::new(&runtime, &registry, &project);
    for name in names {
        standing = standing.with_base_pod(PodId::from(name), Generation(1));
    }
    let mut vcw = VcwLoop::new(VcwParts {
        config: VcwConfig::new("local-test-verifier", [3; 32]),
        roster: experts.roster(),
        router: RouterConfig::default(),
        scorer: UniformScorer,
        key: SamplingKey::from_bytes([7; 32]),
        nonces: NonceSequence::starting_at([1; 8], 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema: None,
        structure: StructurePolicy::new(StructurePolicyConfig::default()).unwrap(),
    })
    .unwrap();

    let (mut first_try, mut repaired, mut exhausted) = (0, 0, 0);
    let (mut repair_calls, mut repair_fixes) = (0, 0);
    for (index, (a, op, b)) in tasks.iter().enumerate() {
        let expected = match op {
            '+' => a + b,
            '-' => a - b,
            _ => a * b,
        };
        experts.set_task(ExpertTask::new(kind, format!("What is {a} {op} {b}?")));
        let before = experts.calls().len();
        let id = format!("qwen-{index}");
        let request = VcwRequest {
            request_id: RequestId::from(id.as_str()),
            project: project.clone(),
            request_digest: Sha256::digest(id.as_bytes()).into(),
            required: VerificationLevel::SampleVerified,
            budget: RequestBudget::new(6, 4).unwrap(),
        };
        let run = vcw
            .run(
                &request,
                &standing,
                &support::ArithmeticVerifier { expected },
                &mut experts.guarded(&standing),
            )
            .unwrap();
        // Bounds that hold whatever the model does.
        let spent = run.trace.spent();
        assert!(
            spent.pod_calls <= 6 && spent.verifier_calls <= 6,
            "{spent:?}"
        );
        let calls: Vec<_> = experts.calls().iter().skip(before).collect();
        assert!(calls.iter().all(|c| c.new_tokens <= 24));
        // A model fault would show as a failed call, not as a quiet decline.
        assert!(calls
            .iter()
            .all(|c| !matches!(c.outcome, CallOutcome::Failed(_))));
        assert!(calls.iter().all(|c| c.model.is_some()));
        match &run.outcome {
            VcwOutcome::Verified(answer) => {
                assert_eq!(
                    answer.payload().text().trim().parse::<i64>().unwrap(),
                    expected
                );
            }
            VcwOutcome::Exhausted { .. } => {
                assert!(matches!(run.trace.outcome(), OutcomeTag::Exhausted(_)));
            }
            VcwOutcome::Escalated { .. } => panic!("no escalation target exists"),
        }
        let repairs = calls
            .iter()
            .filter(|c| c.role == ExpertRole::Repairer)
            .count();
        match (&run.outcome, repairs) {
            (VcwOutcome::Verified(_), 0) => first_try += 1,
            (VcwOutcome::Verified(_), _) => repaired += 1,
            _ => exhausted += 1,
        }
        repair_calls += repairs;
        repair_fixes += usize::from(matches!(run.outcome, VcwOutcome::Verified(_)) && repairs > 0);
        eprintln!(
            "{a} {op} {b} = {expected}: {} | answers {:?}",
            match run.outcome {
                VcwOutcome::Verified(_) => "Verified",
                VcwOutcome::Exhausted { .. } => "Exhausted",
                VcwOutcome::Escalated { .. } => "Escalated",
            },
            calls.iter().map(|c| c.answer.as_str()).collect::<Vec<_>>()
        );
    }
    let stats = qwen.lock().unwrap().stats();
    eprintln!(
        "tasks {}: correct at first proposal {first_try}, repaired {repaired}, exhausted {exhausted}; \
         repair calls {repair_calls}; model tok/s {:.2}",
        tasks.len(),
        stats.decode_tokens_per_second()
    );
    eprintln!("requests that were repaired to Verified: {repair_fixes}");
}
