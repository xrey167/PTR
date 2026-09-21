// Each integration binary compiles this module and uses a subset of it, so
// "never used" here means "not used by this one binary" rather than unused in
// the crate. The allow covers that and nothing else: a helper no binary uses is
// still worth deleting.
#![allow(dead_code)]

use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{execution::*, PtrRuntime};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, Probability, ProjectId, RequestId, TypeId,
    VerificationLevel,
};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

pub const TTL: Duration = Duration::from_secs(60);
pub type Observed = (String, ProjectId, ActionIr);

#[derive(Clone)]
pub struct Probe {
    pub verified: Arc<AtomicUsize>,
    pub calls: Arc<Mutex<Vec<Observed>>>,
    pub report: Arc<Mutex<VerificationReport>>,
}

impl Default for Probe {
    fn default() -> Self {
        Self {
            verified: Arc::new(AtomicUsize::new(0)),
            calls: Arc::new(Mutex::new(vec![])),
            report: Arc::new(Mutex::new(VerificationReport {
                status: VerificationStatus::Pass,
                level: VerificationLevel::Deterministic,
                score: Probability::new(1.0).unwrap(),
                findings: vec![],
            })),
        }
    }
}

impl Probe {
    pub fn verifications(&self) -> usize {
        self.verified.load(Ordering::SeqCst)
    }
    pub fn executions(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
    pub fn hard_finding(&self) {
        self.report.lock().unwrap().findings.push(Finding {
            code: "fixture-denial".into(),
            message: "hard finding".into(),
            hard: true,
        });
    }
}

struct TestVerifier(Probe);
impl Verifier<ActionIr> for TestVerifier {
    fn verify(&self, action: &ActionIr) -> VerificationReport {
        self.0.verified.fetch_add(1, Ordering::SeqCst);
        let mut report = self.0.report.lock().unwrap().clone();
        if action.payload != b"verified payload" {
            report.status = VerificationStatus::Fail;
        }
        report
    }
}

pub enum ExecutorMode {
    Success,
    Error,
    Panic,
}
struct TestExecutor {
    probe: Probe,
    mode: ExecutorMode,
}
impl ActionExecutor for TestExecutor {
    fn execute(&self, dispatch: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
        self.probe.calls.lock().unwrap().push((
            dispatch.principal().to_owned(),
            dispatch.project().clone(),
            dispatch.action().clone(),
        ));
        match self.mode {
            ExecutorMode::Success => Ok(b"executed".to_vec()),
            ExecutorMode::Error => Err("outcome uncertain".into()),
            ExecutorMode::Panic => panic!("simulated adapter panic after effect"),
        }
    }
}

pub fn grant(
    scope: ActionScope,
    probe: &Probe,
    required: RequiredVerification,
    mode: ExecutorMode,
) -> ExecutionGrant {
    ExecutionGrant::new(
        scope,
        required,
        TestVerifier(probe.clone()),
        TestExecutor {
            probe: probe.clone(),
            mode,
        },
    )
}

pub fn scope(action: &ActionIr) -> ActionScope {
    ActionScope {
        project: ProjectId::from("p"),
        target: action.target.clone(),
        operation: action.operation.clone(),
        capability: action.capability.clone(),
        input_type: action.input_type.clone(),
        effect: action.effect,
    }
}

pub fn fixture() -> (PtrRuntime, ActionIr) {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let revision = runtime
        .ingest_text(RequestId::from("request"), "ground state")
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    let action = ActionIr {
        operation: "write".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision,
        payload: b"verified payload".to_vec(),
    };
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
    (runtime, action)
}

pub fn session(
    runtime: &mut PtrRuntime,
    action: &ActionIr,
    probe: &Probe,
    principal: &str,
) -> ExecutionSession {
    runtime
        .register_execution_session(
            principal,
            vec![grant(
                scope(action),
                probe,
                RequiredVerification::FullSemantic,
                ExecutorMode::Success,
            )],
            TTL,
        )
        .unwrap()
}
