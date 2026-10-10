//! One verifier-closed request end to end with a real language model as the experts.
//!
//! ```text
//! cargo run -p ptr-runtime --release --features qwen-cpu --example vcw_qwen -- \
//!     <model_dir> "17 + 25"                      # arithmetic: "<a> <+|-|*> <b>"
//! cargo run -p ptr-runtime --release --features qwen-cpu --example vcw_qwen -- \
//!     <model_dir> "Ada, aged 36" json            # JSON object with name:string, age:integer
//! ```
//!
//! `model_dir` holds `config.json`, `tokenizer.json` and `model.safetensors` of a Qwen2
//! checkpoint (set `PTR_QWEN_SHA256`, `PTR_QWEN_CONFIG_SHA256`, `PTR_QWEN_TOKENIZER_SHA256` and
//! `PTR_QWEN_TOKENIZER_CONFIG_SHA256` to pin the files). The proposer and two repair Pods run
//! on the same model with different system prompts; a minimal local verifier judges each
//! candidate. The printed trace shows rounds, the routed finding codes with their propensities
//! and the outcome.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_pods::{PodRegistry, QwenCpu, QwenCpuOptions};
use ptr_router::{
    NonceSequence, RouterConfig, SamplingKey, SelfModel, SelfModelLimits, UniformScorer,
};
use ptr_runtime::expert::{StructurePolicy, StructurePolicyConfig};
use ptr_runtime::vcw::experts::{
    ExpertRole, ExpertTask, LlmExpertPod, LlmExpertPods, SharedGenerator, TaskKind, TextPayload,
};
use ptr_runtime::vcw::{
    RequestBudget, RuntimeStanding, VcwConfig, VcwLoop, VcwOutcome, VcwParts, VcwRequest,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{CapsuleId, Generation, PodId, ProjectId, RequestId, VerificationLevel};
use ptr_verifier::Verifier;
use sha2::{Digest as _, Sha256};

#[path = "../tests/vcw_experts_support/mod.rs"]
mod support;

fn usage() -> ExitCode {
    eprintln!("usage: vcw_qwen <model_dir> \"<a> <+|-|*> <b>\" | vcw_qwen <model_dir> \"<description>\" json");
    ExitCode::from(2)
}

/// `"a op b"` with integers and `+`, `-` or `*`.
fn parse_expression(text: &str) -> Option<(i64, char, i64)> {
    let mut parts = text.split_whitespace();
    let a = parts.next()?.parse().ok()?;
    let op = parts
        .next()?
        .chars()
        .next()
        .filter(|c| matches!(c, '+' | '-' | '*'))?;
    let b = parts.next()?.parse().ok()?;
    parts.next().is_none().then_some((a, op, b))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(dir), Some(statement)) = (args.first(), args.get(1)) else {
        return usage();
    };
    let json = args.get(2).is_some_and(|kind| kind == "json");

    let (task, verifier): (ExpertTask, Box<dyn Verifier<TextPayload>>) = if json {
        (
            ExpertTask::new(
                TaskKind::JsonObject,
                format!(
                    "Write a JSON object of the form {{\"name\": <string>, \"age\": <integer>}} for: {statement}."
                ),
            ),
            Box::new(support::JsonObjectVerifier {
                fields: vec![("name", support::Kind::String), ("age", support::Kind::Integer)],
            }),
        )
    } else {
        let Some((a, op, b)) = parse_expression(statement) else {
            return usage();
        };
        let expected = match op {
            '+' => a + b,
            '-' => a - b,
            _ => a * b,
        };
        (
            ExpertTask::new(TaskKind::Arithmetic, format!("What is {a} {op} {b}?")),
            Box::new(support::ArithmeticVerifier { expected }),
        )
    };
    let kind = task.kind();

    let started = Instant::now();
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
    let qwen = match QwenCpu::load(&PathBuf::from(dir), &options) {
        Ok(model) => Arc::new(Mutex::new(model)),
        Err(error) => {
            eprintln!("cannot load the model: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("model loaded in {:.1}s", started.elapsed().as_secs_f64());
    let generator: SharedGenerator = qwen.clone();

    // The experts: one proposer, two repairers with different system prompts, one model.
    let project = ProjectId::from("vcw-qwen");
    let make = |name: &str, role: ExpertRole, system: String, bound: u32| {
        Arc::new(
            LlmExpertPod::new(
                &project,
                &PodId::from(name),
                role,
                kind,
                system,
                bound,
                generator.clone(),
            )
            .expect("a token bound within 1..=128"),
        )
    };
    let bound = if json { 96 } else { 24 };
    let mut experts = LlmExpertPods::new(task);
    experts
        .add_proposer(make(
            "proposer",
            ExpertRole::Proposer,
            kind.proposer_system_prompt().to_owned(),
            bound,
        ))
        .expect("distinct Pod ids");
    let codes: &[(&str, &str)] = if json {
        &support::JSON_CODES
    } else {
        &support::ARITHMETIC_CODES
    };
    experts
        .add_repairer(
            make(
                "repairer_a",
                ExpertRole::Repairer,
                kind.repairer_system_prompt().to_owned(),
                bound,
            ),
            codes,
        )
        .expect("distinct Pod ids");
    experts
        .add_repairer(
            make(
                "repairer_b",
                ExpertRole::Repairer,
                format!(
                    "{} Think about the problem once, then answer.",
                    kind.repairer_system_prompt()
                ),
                bound,
            ),
            codes,
        )
        .expect("distinct Pod ids");

    // Registry and lifecycle authority: every expert is registered and live at generation 1.
    let names = ["proposer", "repairer_a", "repairer_b"];
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("runtime");
    let mut registry = PodRegistry::default();
    for name in names {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: project.clone(),
                capsule: CapsuleId::from(name),
                generation: Generation(1),
            })
            .expect("commit");
    }
    for pod in experts.pods() {
        registry.register_new(pod).expect("register");
    }
    let mut standing = RuntimeStanding::new(&runtime, &registry, &project);
    for name in names {
        standing = standing.with_base_pod(PodId::from(name), Generation(1));
    }

    let mut vcw = VcwLoop::new(VcwParts {
        config: VcwConfig::new("local-example-verifier", [9; 32]),
        roster: experts.roster(),
        router: RouterConfig::default(),
        scorer: UniformScorer,
        key: SamplingKey::from_bytes([7; 32]),
        nonces: NonceSequence::starting_at([1; 8], 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema: None,
        structure: StructurePolicy::new(StructurePolicyConfig::default()).expect("policy"),
    })
    .expect("loop");

    let request = VcwRequest {
        request_id: RequestId::from("vcw-qwen-1"),
        project: project.clone(),
        request_digest: Sha256::digest(statement.as_bytes()).into(),
        required: VerificationLevel::SampleVerified,
        budget: RequestBudget::new(6, 4).expect("budget"),
    };
    let run_started = Instant::now();
    let run = match vcw.run(
        &request,
        &standing,
        verifier.as_ref(),
        &mut experts.guarded(&standing),
    ) {
        Ok(run) => run,
        Err(error) => {
            eprintln!("the loop failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed = run_started.elapsed();

    println!("task: {}", experts.task().statement());
    for step in run.trace.steps() {
        println!(
            "round {} origin {:?} decision {:?}",
            step.round, step.origin, step.decision
        );
        for route in &step.routes {
            println!(
                "  routed {:?} -> {} (generation {}) propensity {:?} support {:?}",
                route.class(),
                route.pod(),
                route.pod_generation().0,
                route.propensity(),
                route.support()
            );
        }
        for code in &step.unroutable {
            println!("  unroutable code {code}");
        }
        if let Some(candidate) = &step.candidate {
            println!(
                "  candidate report {:?} class {:?}",
                candidate.report, candidate.class
            );
        }
    }
    for call in experts.calls() {
        println!(
            "call {:?} {} gen {} round {} code {:?} outcome {} tokens {} answer {:?}",
            call.role,
            call.pod,
            call.generation.0,
            call.round,
            call.code,
            call.outcome,
            call.new_tokens,
            call.answer
        );
    }
    if let Some(model) = experts.calls().front().and_then(|call| call.model.as_ref()) {
        println!("model files: {:?}", model.files);
    }
    match &run.outcome {
        VcwOutcome::Verified(answer) => println!(
            "outcome: Verified by {:?}: {:?}",
            answer
                .authors()
                .iter()
                .map(|a| a.pod.0.clone())
                .collect::<Vec<_>>(),
            answer.payload().text()
        ),
        VcwOutcome::Exhausted { reason } => println!("outcome: Exhausted ({reason:?})"),
        VcwOutcome::Escalated { reason } => println!("outcome: Escalated ({reason:?})"),
    }
    let stats = qwen.lock().expect("model lock").stats();
    println!(
        "trace {:?}; {:.1}s total; model: {} generated tokens, prefill {:.1}s, decode {:.1}s ({:.2} tok/s), {} prompt tokens reused",
        run.trace.outcome(),
        elapsed.as_secs_f64(),
        stats.generated_tokens,
        stats.prefill_time.as_secs_f64(),
        stats.decode_time.as_secs_f64(),
        stats.decode_tokens_per_second(),
        stats.prompt_tokens_reused
    );
    ExitCode::SUCCESS
}
