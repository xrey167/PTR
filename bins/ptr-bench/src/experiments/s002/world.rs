//! The fixture of one case in one arm: a fresh runtime with its grant, its target capsule, an
//! execution session whose executor logs every call into the mirror, a Pod, a verifier, and the
//! scripted model backend.
//!
//! Everything the harness supplies is deterministic and counts what it does. The backend asks
//! for `rounds` Pod runs and then for the action, stamping the action with the revision of the
//! context it was last shown (a model that re-reads its snapshot every round, the behaviour of
//! the resumable loop's `ModelResumeRequest`); the Pod echoes its input; the verifiers pass.

use super::oracle::Mirror;
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry, TypedPayload};
use ptr_runtime::execution::{
    ActionExecutor, ActionScope, ExecutionGrant, ExecutionSession, RequiredVerification,
    VerifiedDispatch,
};
use ptr_runtime::{PtrRuntime, SemanticChange, SemanticGrant};
use ptr_semdb::SemanticDelta;
use ptr_types::{
    ActionIr, CapabilityId, CapsuleId, Effect, Generation, PodId, PrincipalId, Probability,
    ProjectId, Revision, TypeId, VerificationLevel,
};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus, Verifier};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const PROJECT: &str = "s002-project";
pub const TARGET: &str = "capsule:s002";
pub const OPERATION: &str = "write";
pub const CAPABILITY: &str = "file.write";
pub const INPUT_TYPE: &str = "Bytes";
pub const POD_CAPABILITY: &str = "Echo<Text>";
pub const TTL: Duration = Duration::from_secs(600);
pub const BASE_KEY: &str = "foreign:base";
pub const BASE_VALUE: &str = "x";

/// The payload of the Pod request of round `round`: the same every time, so a Pod output
/// promoted twice under one request key is the same bytes.
pub fn round_payload(round: u64) -> Vec<u8> {
    format!("round-{round}").into_bytes()
}

/// Asks for `rounds` Pod runs, then for the action of `effect`.
pub struct Chain {
    pub rounds: u64,
    pub effect: Effect,
}

fn pod_request(round: u64) -> Vec<ModelEvent> {
    vec![ModelEvent::PodRequested {
        capability: CapabilityId::from(POD_CAPABILITY),
        input_type: TypeId::from("Text"),
        payload: round_payload(round),
    }]
}

pub fn action_at(effect: Effect, revision: Revision) -> ActionIr {
    ActionIr {
        operation: OPERATION.into(),
        target: TARGET.into(),
        capability: CapabilityId::from(CAPABILITY),
        effect,
        input_type: TypeId::from(INPUT_TYPE),
        generation: Generation(1),
        revision,
        payload: b"s002-action".to_vec(),
    }
}

impl InferenceBackend for Chain {
    fn name(&self) -> &'static str {
        "s002-chain"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(if self.rounds == 0 {
            vec![
                ModelEvent::ActionReady(action_at(self.effect, request.semantic_context.revision)),
                ModelEvent::Finished,
            ]
        } else {
            pod_request(0)
        })
    }
}

impl ResumableInferenceBackend for Chain {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(if u64::from(request.round) >= self.rounds {
            vec![
                ModelEvent::ActionReady(action_at(self.effect, request.revision())),
                ModelEvent::Finished,
            ]
        } else {
            pod_request(u64::from(request.round))
        })
    }
}

/// Echoes its input and counts how often it ran: the harness's own count of Pod work.
pub struct CountingEcho {
    manifest: PodManifest,
    runs: Arc<AtomicUsize>,
}

impl DynPod for CountingEcho {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        Ok(input)
    }
}

fn pass_report() -> VerificationReport {
    VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(1.0).expect("a bounded score"),
        findings: Vec::new(),
    }
}

/// Passes every Pod payload.
pub struct PassPayload;

impl Verifier<TypedPayload> for PassPayload {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        pass_report()
    }
}

struct PassAction;

impl Verifier<ActionIr> for PassAction {
    fn verify(&self, _: &ActionIr) -> VerificationReport {
        pass_report()
    }
}

/// Passes every semantic change of the grant (the host's writes).
struct PassSemantic;

impl<'a> Verifier<SemanticChange<'a>> for PassSemantic {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        pass_report()
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for PassSemantic {
    fn name(&self) -> &'static str {
        "s002-pass-semantic"
    }
}

/// The executor: logs the action it was handed, judged by the mirror at that moment.
struct Recording(Arc<Mutex<Mirror>>);

impl ActionExecutor for Recording {
    fn execute(&self, dispatch: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
        let mut mirror = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        mirror.record_call(dispatch.action());
        Ok(b"executed".to_vec())
    }
}

/// The fixture.
pub struct World {
    pub runtime: PtrRuntime,
    pub session: ExecutionSession,
    pub mirror: Arc<Mutex<Mirror>>,
    pub pods: PodRegistry,
    pub pod_runs: Arc<AtomicUsize>,
    pub project: ProjectId,
    foreign_counter: u64,
}

fn failed(what: &str) -> impl Fn(ptr_runtime::RuntimeError) -> String + '_ {
    move |error| format!("{what}: {error:?}")
}

impl World {
    /// A runtime whose stale-revision cancellation is `cancel`, with the target capsule at
    /// generation 1, every effect permitted for the capability, a session granting exactly the
    /// one action scope of `effect`, and the base foreign key written once (so that the no-op
    /// writer has a key to rewrite). Setup is the harness's: a refusal is `Err`.
    pub fn new(cancel: bool, effect: Effect) -> Result<Self, String> {
        let mut config = PtrConfig::default();
        config.semantic.cancel_stale_revisions = cancel;
        let mut runtime = PtrRuntime::new(config).map_err(failed("creating the runtime"))?;
        runtime
            .install_semantic_grant(
                SemanticGrant::new(RequiredVerification::Deterministic)
                    .with_verifier(PassSemantic)
                    .allow_host_writes(),
            )
            .map_err(failed("installing the semantic grant"))?;
        let project = ProjectId::from(PROJECT);
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: project.clone(),
                capsule: CapsuleId::from(TARGET),
                generation: Generation(1),
            })
            .map_err(failed("committing the target capsule"))?;
        let permissions = runtime.permissions_mut();
        permissions
            .capabilities
            .insert(CapabilityId::from(CAPABILITY));
        permissions.allow_mutation = true;
        permissions.allow_external = true;
        permissions.allow_irreversible = true;
        let mirror = Arc::new(Mutex::new(Mirror::new()));
        let scope = ActionScope {
            project: project.clone(),
            target: TARGET.to_string(),
            operation: OPERATION.to_string(),
            capability: CapabilityId::from(CAPABILITY),
            input_type: TypeId::from(INPUT_TYPE),
            effect,
        };
        let session = runtime
            .register_execution_session(
                "s002-operator",
                vec![ExecutionGrant::new(
                    scope,
                    RequiredVerification::Deterministic,
                    PassAction,
                    Recording(mirror.clone()),
                )],
                TTL,
            )
            .map_err(|error| format!("registering the execution session: {error:?}"))?;
        let pod_runs = Arc::new(AtomicUsize::new(0));
        let mut pods = PodRegistry::default();
        pods.register(Arc::new(CountingEcho {
            manifest: PodManifest {
                project: project.clone(),
                id: PodId::from("s002-echo"),
                capabilities: vec![CapabilityId::from(POD_CAPABILITY)],
                accepts: vec![TypeId::from("Text")],
                produces: vec![TypeId::from("Text")],
                effects: vec![Effect::Pure],
                protocol_version: 1,
            },
            runs: pod_runs.clone(),
        }));
        let mut world = Self {
            runtime,
            session,
            mirror,
            pods,
            pod_runs,
            project,
            foreign_counter: 0,
        };
        world.write_foreign(BASE_KEY, BASE_VALUE)?;
        Ok(world)
    }

    pub fn pod_runs(&self) -> usize {
        self.pod_runs.load(Ordering::SeqCst)
    }

    /// Runs `f` on the mirror.
    pub fn with_mirror<T>(&self, f: impl FnOnce(&mut Mirror) -> T) -> T {
        let mut mirror = self
            .mirror
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut mirror)
    }

    /// A host write of `key` to `value`; the mirror is told. `Err` if the runtime refuses it:
    /// the harness' write path is not what S002 tests.
    pub fn write_foreign(&mut self, key: &str, value: &str) -> Result<(), String> {
        let mut delta = SemanticDelta::default();
        delta.upserts.insert(key.to_string(), value.into());
        let expected = self.runtime.revision();
        self.runtime
            .apply_verified_semantic_delta(expected, delta, &PrincipalId::from("s002-writer"))
            .map_err(failed("a host write"))?;
        self.with_mirror(|mirror| mirror.foreign_write(key, value));
        Ok(())
    }

    /// A write of a key nobody wrote: the revision moves.
    pub fn write_fresh_key(&mut self) -> Result<(), String> {
        self.foreign_counter += 1;
        let key = format!("foreign:{}", self.foreign_counter);
        self.write_foreign(&key, "y")
    }

    pub fn supersede_generation(&mut self) -> Result<(), String> {
        self.runtime
            .commit(LedgerEvent::CapsuleSuperseded {
                capsule: CapsuleId::from(TARGET),
                old: Generation(1),
                new: Generation(2),
            })
            .map_err(failed("superseding the generation"))?;
        self.with_mirror(|mirror| mirror.supersede(2));
        Ok(())
    }

    pub fn revoke_generation(&mut self) -> Result<(), String> {
        self.runtime
            .commit(LedgerEvent::Revoked {
                subject: TARGET.to_string(),
                generation: Generation(1),
            })
            .map_err(failed("revoking the generation"))?;
        self.with_mirror(|mirror| mirror.revoke(1));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::s002::params;

    #[test]
    fn a_fresh_world_has_the_base_write_and_nothing_stale() {
        for effect in params::EFFECTS {
            let world = World::new(true, effect).unwrap();
            assert_eq!(world.runtime.revision().0, 1);
            assert_eq!(world.with_mirror(|m| m.revision()), 1);
            assert_eq!(world.runtime.live_generation(TARGET), Some(Generation(1)));
            assert_eq!(world.pod_runs(), 0);
        }
    }

    #[test]
    fn a_fresh_key_moves_the_revision_and_the_base_key_again_does_not() {
        let mut world = World::new(true, Effect::Mutation).unwrap();
        world.write_fresh_key().unwrap();
        assert_eq!(world.runtime.revision().0, 2);
        assert_eq!(world.with_mirror(|m| m.revision()), 2);
        world.write_foreign(BASE_KEY, BASE_VALUE).unwrap();
        assert_eq!(world.runtime.revision().0, 2, "a rewrite of the same value");
        assert_eq!(world.with_mirror(|m| m.revision()), 2);
    }

    #[test]
    fn a_lifecycle_record_moves_no_revision_in_the_runtime_or_in_the_mirror() {
        let mut revoked = World::new(true, Effect::Read).unwrap();
        revoked.revoke_generation().unwrap();
        assert_eq!(revoked.runtime.revision().0, 1);
        assert!(revoked.with_mirror(|m| m.generation_is_dead()));
        let mut superseded = World::new(true, Effect::Read).unwrap();
        superseded.supersede_generation().unwrap();
        assert_eq!(superseded.runtime.revision().0, 1);
        assert_eq!(
            superseded.runtime.live_generation(TARGET),
            Some(Generation(2))
        );
    }

    #[test]
    fn the_backend_stamps_the_action_with_the_context_it_was_last_shown() {
        let backend = Chain {
            rounds: 0,
            effect: Effect::External,
        };
        let request = ModelRequest {
            request_id: "r".into(),
            semantic_context: ptr_types::SemanticContext {
                revision: Revision(7),
                entries: vec![],
            },
            raw_text: "x".into(),
        };
        let events = backend.infer(&request).unwrap();
        assert!(matches!(
            &events[0],
            ModelEvent::ActionReady(a) if a.revision == Revision(7) && a.effect == Effect::External
        ));
    }
}
