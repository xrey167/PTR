//! Expert Pods backed by a [`TextGenerator`].

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use ptr_pods::{
    DynPod, GenerationError, GenerationOutput, GenerationRequest, ModelIdentity, PodManifest,
    TextGenerator, TypedPayload,
};
use ptr_router::{FindingCode, PodEligibility, PodRef};
use ptr_types::{PodId, ProjectId};

use super::guard::{Guarded, GuardedExperts};
use super::manifest::{expert_manifest, expert_manifest_for_model, ExpertRole};
use super::task::{ExpertTask, TaskKind, TextPayload};
use super::{CallOutcome, CallRecord, ExpertCall, ExpertErrorKind, RosterError, MAX_LOGGED_CALLS};
use crate::vcw::{ProposeCall, RepairCall, Roster};

/// Why a Pod's generator did not return text.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertError {
    /// The generator refused or failed.
    Generation(GenerationError),
    /// The shared generator's lock was poisoned by a panic in an earlier call.
    LockPoisoned,
}

impl ExpertError {
    /// The typed kind recorded in the call log.
    #[must_use]
    pub fn kind(&self) -> ExpertErrorKind {
        match self {
            Self::Generation(GenerationError::InvalidRequest(_)) => ExpertErrorKind::InvalidRequest,
            Self::Generation(_) => ExpertErrorKind::Backend,
            Self::LockPoisoned => ExpertErrorKind::LockPoisoned,
        }
    }
}

impl fmt::Display for ExpertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generation(error) => error.fmt(f),
            Self::LockPoisoned => f.write_str("generator lock poisoned"),
        }
    }
}

impl std::error::Error for ExpertError {}

impl From<GenerationError> for ExpertError {
    fn from(error: GenerationError) -> Self {
        Self::Generation(error)
    }
}

/// A generator shared by every Pod built on the same model.
pub type SharedGenerator = Arc<Mutex<dyn TextGenerator + Send>>;

/// One expert Pod: a system prompt and a token bound on a shared generator, with a manifest.
///
/// As a [`DynPod`] it takes a payload of its answer type whose bytes are the user message and
/// returns the model's text as a payload of the same type. The loop calls it through
/// [`LlmExpertPods`], which builds the message from the task, the previous candidate and one
/// finding code.
pub struct LlmExpertPod {
    manifest: PodManifest,
    role: ExpertRole,
    system_prompt: String,
    max_new_tokens: u32,
    generator: SharedGenerator,
    model: Option<Arc<ModelIdentity>>,
}

impl LlmExpertPod {
    /// A Pod on `generator`. The token bound is validated here, with the same check every
    /// generator applies (`1..=`[`MAX_NEW_TOKENS_LIMIT`](ptr_pods::MAX_NEW_TOKENS_LIMIT)), so a
    /// misconfigured Pod fails at construction instead of declining every call. When the
    /// generator reports the digests of its model files they are put in the manifest.
    ///
    /// # Errors
    /// [`GenerationError::InvalidRequest`] for a bound of zero or above the limit;
    /// [`GenerationError::Backend`] if the generator's lock is poisoned.
    pub fn new(
        project: &ProjectId,
        pod: &PodId,
        role: ExpertRole,
        kind: TaskKind,
        system_prompt: impl Into<String>,
        max_new_tokens: u32,
        generator: SharedGenerator,
    ) -> Result<Self, GenerationError> {
        GenerationRequest::new("", "", max_new_tokens).validate()?;
        let model = generator
            .lock()
            .map_err(|_| GenerationError::Backend("generator lock poisoned".into()))?
            .identity()
            .map(Arc::new);
        let payload = kind.payload_type();
        let manifest = match &model {
            Some(identity) => expert_manifest_for_model(project, pod, role, &payload, identity),
            None => expert_manifest(project, pod, role, &payload),
        };
        Ok(Self {
            manifest,
            role,
            system_prompt: system_prompt.into(),
            max_new_tokens,
            generator,
            model,
        })
    }

    /// The digests of the model files this Pod runs, when the generator reports them.
    #[must_use]
    pub fn model_identity(&self) -> Option<&ModelIdentity> {
        self.model.as_deref()
    }

    #[must_use]
    pub fn role(&self) -> ExpertRole {
        self.role
    }

    #[must_use]
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Run the model on `user` with this Pod's system prompt and bound.
    ///
    /// Not public: a direct call would skip the call-time eligibility check of [`Guarded`].
    ///
    /// # Errors
    /// [`ExpertError`]: the generator's error, or a poisoned lock.
    pub(crate) fn complete(&self, user: &str) -> Result<GenerationOutput, ExpertError> {
        let request = GenerationRequest::new(&self.system_prompt, user, self.max_new_tokens);
        let mut generator = self
            .generator
            .lock()
            .map_err(|_| ExpertError::LockPoisoned)?;
        Ok(generator.generate(&request)?)
    }

    /// Like [`Self::complete`], but asks `eligible` once the generator's lock is held, right
    /// before the model runs, and returns `Ok(None)` without running it when that is false.
    ///
    /// # Errors
    /// [`ExpertError`]: the generator's error, or a poisoned lock.
    pub(crate) fn complete_while(
        &self,
        user: &str,
        eligible: &dyn Fn() -> bool,
    ) -> Result<Option<GenerationOutput>, ExpertError> {
        let request = GenerationRequest::new(&self.system_prompt, user, self.max_new_tokens);
        let mut generator = self
            .generator
            .lock()
            .map_err(|_| ExpertError::LockPoisoned)?;
        if !eligible() {
            return Ok(None);
        }
        Ok(Some(generator.generate(&request)?))
    }
}

impl DynPod for LlmExpertPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        if !self.manifest.accepts.contains(&input.type_id) {
            return Err(format!("payload type {} is not accepted", input.type_id));
        }
        let user = String::from_utf8(input.bytes).map_err(|_| "message is not UTF-8".to_owned())?;
        let output = self.complete(&user).map_err(|error| error.to_string())?;
        Ok(TypedPayload {
            type_id: input.type_id,
            bytes: output.text.into_bytes(),
        })
    }
}

struct RepairerEntry {
    pod: Arc<LlmExpertPod>,
    /// The fixed description of each finding code this Pod repairs.
    codes: BTreeMap<String, String>,
}

/// The expert Pods of one task, over [`LlmExpertPod`]s.
///
/// Hand it to the loop through [`Self::guarded`], which re-checks each Pod's eligibility at
/// call time. A call for a Pod this adapter does not hold, or for a code the Pod does not
/// repair, produces nothing and is not logged. The task is set per request with
/// [`Self::set_task`].
pub struct LlmExpertPods {
    task: ExpertTask,
    proposers: BTreeMap<PodId, Arc<LlmExpertPod>>,
    repairers: BTreeMap<PodId, RepairerEntry>,
    calls: VecDeque<ExpertCall>,
}

impl LlmExpertPods {
    #[must_use]
    pub fn new(task: ExpertTask) -> Self {
        Self {
            task,
            proposers: BTreeMap::new(),
            repairers: BTreeMap::new(),
            calls: VecDeque::new(),
        }
    }

    pub fn set_task(&mut self, task: ExpertTask) {
        self.task = task;
    }

    #[must_use]
    pub fn task(&self) -> &ExpertTask {
        &self.task
    }

    fn holds(&self, pod: &PodId) -> bool {
        self.proposers.contains_key(pod) || self.repairers.contains_key(pod)
    }

    /// Hold a proposer Pod.
    ///
    /// # Errors
    /// [`RosterError::DuplicatePod`] when a Pod with this id is already held.
    pub fn add_proposer(&mut self, pod: Arc<LlmExpertPod>) -> Result<(), RosterError> {
        let id = pod.manifest.id.clone();
        if self.holds(&id) {
            return Err(RosterError::DuplicatePod(id));
        }
        self.proposers.insert(id, pod);
        Ok(())
    }

    /// Hold a repair Pod for `codes`, each with the one-line description the Pod is shown.
    ///
    /// # Errors
    /// [`RosterError::DuplicatePod`] when a Pod with this id is already held.
    pub fn add_repairer(
        &mut self,
        pod: Arc<LlmExpertPod>,
        codes: &[(&str, &str)],
    ) -> Result<(), RosterError> {
        let id = pod.manifest.id.clone();
        if self.holds(&id) {
            return Err(RosterError::DuplicatePod(id));
        }
        let codes = codes
            .iter()
            .map(|(code, description)| ((*code).to_owned(), (*description).to_owned()))
            .collect();
        self.repairers.insert(id, RepairerEntry { pod, codes });
        Ok(())
    }

    /// The roster to hand to [`VcwParts`](crate::vcw::VcwParts).
    #[must_use]
    pub fn roster(&self) -> Roster {
        Roster {
            proposers: self.proposers.keys().cloned().collect(),
            repairers: self
                .repairers
                .iter()
                .map(|(pod, entry)| {
                    (
                        pod.clone(),
                        entry
                            .codes
                            .keys()
                            .map(|c| FindingCode::new(c.as_str()))
                            .collect(),
                    )
                })
                .collect(),
        }
    }

    /// The Pods to register in a [`PodRegistry`](ptr_pods::PodRegistry).
    #[must_use]
    pub fn pods(&self) -> Vec<Arc<dyn DynPod>> {
        let proposers = self.proposers.values().cloned();
        let repairers = self.repairers.values().map(|entry| entry.pod.clone());
        proposers
            .chain(repairers)
            .map(|pod| pod as Arc<dyn DynPod>)
            .collect()
    }

    /// The most recent calls (at most [`MAX_LOGGED_CALLS`]).
    #[must_use]
    pub fn calls(&self) -> &VecDeque<ExpertCall> {
        &self.calls
    }

    /// This adapter as the loop's [`RolePods`](crate::vcw::RolePods), checking each Pod's
    /// eligibility in `standing` at call time. `standing` must be the view passed to the loop's
    /// `run` for the same request.
    #[must_use]
    pub fn guarded<'a>(&'a mut self, standing: &'a dyn PodEligibility) -> Guarded<'a, Self> {
        Guarded::new(self, standing)
    }

    fn log(&mut self, call: ExpertCall) {
        if self.calls.len() >= MAX_LOGGED_CALLS {
            self.calls.pop_front();
        }
        self.calls.push_back(call);
    }

    /// Run `pod` on `message` unless `eligible` says no, and log what happened.
    fn run(
        &mut self,
        pod: &Arc<LlmExpertPod>,
        call: &PodRef,
        round: u32,
        code: Option<&str>,
        message: &str,
        eligible: &dyn Fn() -> bool,
    ) -> Option<TextPayload> {
        let (outcome, payload, new_tokens) = if eligible() {
            match pod.complete_while(message, eligible) {
                Ok(Some(output)) => {
                    let payload = TextPayload::new(
                        self.task.kind().payload_type(),
                        self.task.kind().extract(&output.text),
                    );
                    (CallOutcome::Produced, Some(payload), output.new_tokens)
                }
                Ok(None) => (CallOutcome::IneligibleAtCallTime, None, 0),
                Err(error) => (CallOutcome::Failed(error.kind()), None, 0),
            }
        } else {
            (CallOutcome::IneligibleAtCallTime, None, 0)
        };
        let record = CallRecord {
            role: pod.role,
            pod: call,
            round,
            code,
            outcome,
            new_tokens,
            answer: payload
                .as_ref()
                .map(|p| super::clipped(p.text()))
                .unwrap_or_default(),
            model: pod.model.clone(),
        };
        self.log(record.into_call());
        payload
    }
}

impl GuardedExperts<TextPayload> for LlmExpertPods {
    fn propose_while(
        &mut self,
        call: ProposeCall<'_>,
        eligible: &dyn Fn() -> bool,
    ) -> Option<TextPayload> {
        let pod = self.proposers.get(&call.pod.pod)?.clone();
        let message = self.task.proposer_message(call.round);
        self.run(&pod, call.pod, call.round, None, &message, eligible)
    }

    fn repair_while(
        &mut self,
        call: RepairCall<'_, TextPayload>,
        eligible: &dyn Fn() -> bool,
    ) -> Option<TextPayload> {
        let entry = self.repairers.get(&call.pod.pod)?;
        let description = entry.codes.get(call.code.as_str())?.clone();
        let pod = entry.pod.clone();
        let message =
            self.task
                .repairer_message(call.previous.text(), call.code.as_str(), &description);
        self.run(
            &pod,
            call.pod,
            call.round,
            Some(call.code.as_str()),
            &message,
            eligible,
        )
    }
}
