//! KV-reuse taint at the hard effect boundary (doc 36, phase P2a), persisted in the ledger.
//!
//! A model run produced under Approximate KV reuse may only yield candidates. The runtime
//! records the taint of every subject it ingests from such a run (the action by its revision-
//! and generation-independent identity digest, optionally the request) and `authorize_action`
//! refuses a recorded action with [`RuntimeError::ApproximateStateAtEffectBoundary`], so the
//! check sits before any `EffectAttempted` record for callers that authorize first (the HTTP
//! path does). Keying by digest means the bare `ActionIr` that callers hold stays enforced
//! although it carries no taint field.
//!
//! # Persistence
//!
//! The registry is a projection of [`LedgerEvent::KvTaintCommitted`] records, keyed by
//! `(KvTaintSubjectKind, digest)`. An Approximate ingest appends its records BEFORE the
//! [`ResumableRun`] is returned (taint first: a crash in between leaves a stray record, which
//! only refuses more) and only then updates the in-memory registry, through the very function
//! replay uses ([`PtrRuntime::apply_kv_taint_event`]), so live and replayed state cannot
//! diverge. Replay is fail closed: a `Record` must keep the live set within
//! [`MAX_LIVE_TAINT_SUBJECTS`]; a `Clear` must name only live subjects of the same recipe
//! digest. A forged `Record` therefore applies (taint only refuses more); a forged `Clear`
//! makes the log unopenable. The compacted snapshot carries the registry through the
//! `kv-taint:` entries of its materialized state and `restore_compacted` rebuilds it.
//!
//! # Bound
//!
//! At most [`MAX_LIVE_TAINT_SUBJECTS`] subjects are live. An ingest that would exceed it is
//! refused with [`RuntimeError::KvTaintRegistryFull`] before anything is appended; nothing
//! is ever evicted. A `Clear` frees the subject.
//!
//! # Digests
//!
//! * Subject, Action: [`taint_identity`] (domain `ptr.kv.taint.action-identity.v1`).
//! * Subject, Request: SHA-256 of `ptr.kv.taint.request.v1\0 || u64le(len) || request id`.
//! * Report digest: [`ptr_model_api::KvReuseReport::digest`], SHA-256 over the domain tag,
//!   the mode byte (Exact 0, Approximate 1) and `u64le(hit_tokens)`.
//! * Recipe digest ([`ingest_model_run_for`](PtrRuntime::ingest_model_run_for)): SHA-256 over
//!   `ptr.kv.taint.recipe.v1\0 || u64le(len) operation tag || lineage_digest (32 bytes) ||
//!   u64le(len) producer || u64le(len) generation parameters`. The report exposes only mode
//!   and hit count, which must not enter a recipe (a cold exact recompute has to reproduce
//!   it), so producer and generation parameters are named by the caller in
//!   [`KvTaintRequest`]. [`PtrRuntime::ingest_model_run`] uses the fixed recipe
//!   `SHA-256("ptr.kv.taint.recipe.unspecified-request.v1\0")`.
//!
//! Open: one live entry per subject. A second Approximate ingest of an already-live subject is
//! not recorded, so the entry (and the recipe a clear must reproduce) is the first one.

use crate::{PtrRuntime, ResumableRun, RuntimeError};
use ptr_ledger::{
    check_encodable, effect_code, integrity, KvTaintChange, KvTaintSubjectKind, LedgerEvent,
    MAX_KV_TAINT_SUBJECTS,
};
use ptr_model_api::{KvReuseTaint, ModelEvent, ModelRun, SegmentedModelRequest};
use ptr_types::{ActionIr, Digest, RequestId};
use std::collections::BTreeMap;

const TAINT_IDENTITY_DOMAIN: &[u8] = b"ptr.kv.taint.action-identity.v1\0";

/// What was asked for, without where the runtime stood when it was asked.
///
/// Covers target, operation, capability, input type, effect and payload, but not revision
/// or generation: re-stamping a stale action to the current revision (the normal flow after
/// any unrelated commit) must not shed the taint, which keying by `action_digest` would
/// allow. The payload stays in the key because it is the content the Approximate run
/// produced; a different payload is a different, separately derived action, and leaving it
/// out would refuse every action on the same target and capability. Domain-separated from
/// `action_digest`, so the two digests of one action never collide.
pub(crate) fn taint_identity(action: &ActionIr) -> [u8; 32] {
    let mut material = Vec::from(TAINT_IDENTITY_DOMAIN);
    for field in [
        action.target.as_str(),
        action.operation.as_str(),
        action.capability.0.as_str(),
        action.input_type.0.as_str(),
    ] {
        material.extend_from_slice(&(field.len() as u64).to_le_bytes());
        material.extend_from_slice(field.as_bytes());
    }
    material.extend_from_slice(&(action.payload.len() as u64).to_le_bytes());
    material.extend_from_slice(&action.payload);
    material.push(effect_code(action.effect));
    integrity::sha256(&material)
}

const REQUEST_DOMAIN: &[u8] = b"ptr.kv.taint.request.v1\0";
const RECIPE_DOMAIN: &[u8] = b"ptr.kv.taint.recipe.v1\0";
const UNSPECIFIED_RECIPE_DOMAIN: &[u8] = b"ptr.kv.taint.recipe.unspecified-request.v1\0";
/// The operation tag of a model run in a recipe digest.
const MODEL_RUN_OPERATION: &str = "model-run.v1";

/// Most live taint subjects the runtime holds (owner decision: 65,536).
pub const MAX_LIVE_TAINT_SUBJECTS: usize = 65_536;

/// One taint subject: what kind of thing and its digest.
pub(crate) type TaintSubject = (KvTaintSubjectKind, [u8; 32]);

/// What a live subject was recorded with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TaintEntry {
    pub(crate) report_digest: Digest,
    pub(crate) recipe_digest: Digest,
}

/// Why the taint registry refused a change or a restored state. Wrapped in
/// [`RuntimeError::KvTaint`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvTaintError {
    /// The record names no subject, a zero digest, or is otherwise not encodable.
    Unencodable,
    /// A clear names a subject that is not live.
    ClearSubjectNotLive,
    /// A clear names a live subject whose recorded recipe differs.
    ClearRecipeMismatch,
    /// A `kv-taint:` entry of a restored state is malformed.
    MalformedStateEntry { reason: &'static str },
}

/// The live taint registry: a projection of the committed `KvTaintCommitted` records.
#[derive(Debug)]
pub(crate) struct TaintRegistry {
    live: BTreeMap<TaintSubject, TaintEntry>,
    limit: usize,
    /// Derivations minted by this process (`ingest_derived_run`), by Derivation subject digest,
    /// with the content digest of the segment minted. Memory only: not history, empty after a
    /// restart. Bounded by `limit` like `live`.
    minted: BTreeMap<[u8; 32], Digest>,
}

impl Default for TaintRegistry {
    fn default() -> Self {
        Self {
            live: BTreeMap::new(),
            limit: MAX_LIVE_TAINT_SUBJECTS,
            minted: BTreeMap::new(),
        }
    }
}

impl TaintRegistry {
    fn get(&self, subject: &TaintSubject) -> Option<&TaintEntry> {
        self.live.get(subject)
    }

    /// The content digest of the segment minted under `derivation`, if this process minted it.
    pub(crate) fn minted_segment(&self, derivation: &[u8; 32]) -> Option<&Digest> {
        self.minted.get(derivation)
    }

    /// Whether one more derivation can be remembered (a re-mint of a known one always can).
    pub(crate) fn mint_room(&self, derivation: &[u8; 32]) -> Result<(), RuntimeError> {
        if self.minted.contains_key(derivation) || self.minted.len() < self.limit {
            return Ok(());
        }
        Err(RuntimeError::KvTaintRegistryFull { limit: self.limit })
    }

    /// Remember a minted derivation; the caller has checked [`TaintRegistry::mint_room`].
    pub(crate) fn remember_minted(&mut self, derivation: [u8; 32], segment: Digest) {
        self.minted.insert(derivation, segment);
    }

    /// Subjects of `subjects` not yet live, if adding them stays within the bound.
    fn room_for(&self, fresh: usize) -> Result<(), RuntimeError> {
        if self.live.len().saturating_add(fresh) > self.limit {
            return Err(RuntimeError::KvTaintRegistryFull { limit: self.limit });
        }
        Ok(())
    }

    /// Rebuild from the `kv-taint:` entries of a restored materialized state: `approx:` is
    /// live, `cleared:` is not. Anything malformed, or more live subjects than the bound,
    /// fails closed.
    pub(crate) fn from_materialized(
        values: &BTreeMap<String, String>,
    ) -> Result<Self, RuntimeError> {
        Self::from_materialized_within(values, MAX_LIVE_TAINT_SUBJECTS)
    }

    fn from_materialized_within(
        values: &BTreeMap<String, String>,
        limit: usize,
    ) -> Result<Self, RuntimeError> {
        let mut registry = Self {
            live: BTreeMap::new(),
            limit,
            minted: BTreeMap::new(),
        };
        let malformed =
            |reason| RuntimeError::KvTaint(KvTaintError::MalformedStateEntry { reason });
        for (key, value) in values.range(ptr_state::KV_TAINT_PREFIX.to_owned()..) {
            let Some(rest) = key.strip_prefix(ptr_state::KV_TAINT_PREFIX) else {
                break;
            };
            let (name, hex) = rest
                .split_once(':')
                .ok_or_else(|| malformed("key has no digest"))?;
            let kind = (0..=u8::MAX)
                .map_while(KvTaintSubjectKind::from_code)
                .find(|kind| kind.name() == name)
                .ok_or_else(|| malformed("unknown subject kind"))?;
            let digest = parse_digest(hex).ok_or_else(|| malformed("bad subject digest"))?;
            if let Some(cleared) = value.strip_prefix("cleared:") {
                parse_digest(cleared).ok_or_else(|| malformed("bad recompute digest"))?;
            } else if let Some(approx) = value.strip_prefix("approx:") {
                let (report, recipe) = approx
                    .split_once(':')
                    .ok_or_else(|| malformed("approx value has no recipe"))?;
                let entry = TaintEntry {
                    report_digest: parse_digest(report)
                        .ok_or_else(|| malformed("bad report digest"))?,
                    recipe_digest: parse_digest(recipe)
                        .ok_or_else(|| malformed("bad recipe digest"))?,
                };
                registry.room_for(1)?;
                registry.live.insert((kind, digest), entry);
            } else {
                return Err(malformed("unknown value form"));
            }
        }
        Ok(registry)
    }
}

/// A canonical (lowercase, 64 hex digits, non-zero) digest.
fn parse_digest(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 || !text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return None;
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().chunks(2)) {
        *slot = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    (out != [0; 32]).then_some(out)
}

/// What a model run is ingested for: the request it answered and the recipe parameters the
/// run report does not expose.
#[derive(Clone, Copy, Debug)]
pub struct KvTaintRequest<'a> {
    /// The segmented request the run answered; its lineage digest and id enter the taint.
    pub request: &'a SegmentedModelRequest,
    /// The producer identity (backend name and version) that ran it.
    pub producer: &'a str,
    /// Canonical bytes of the generation parameters (token budget, sampling, and so on).
    pub generation_parameters: &'a [u8],
}

pub(crate) fn put_len_prefixed(material: &mut Vec<u8>, bytes: &[u8]) {
    material.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    material.extend_from_slice(bytes);
}

pub(crate) fn request_subject(request_id: &RequestId) -> TaintSubject {
    let mut material = Vec::from(REQUEST_DOMAIN);
    put_len_prefixed(&mut material, request_id.0.as_bytes());
    (KvTaintSubjectKind::Request, integrity::sha256(&material))
}

fn unspecified_recipe() -> Digest {
    integrity::sha256(UNSPECIFIED_RECIPE_DOMAIN)
}

pub(crate) fn request_recipe(request: &KvTaintRequest<'_>) -> Digest {
    let mut material = Vec::from(RECIPE_DOMAIN);
    put_len_prefixed(&mut material, MODEL_RUN_OPERATION.as_bytes());
    material.extend_from_slice(&request.request.lineage_digest());
    put_len_prefixed(&mut material, request.producer.as_bytes());
    put_len_prefixed(&mut material, request.generation_parameters);
    integrity::sha256(&material)
}

const CONTENT_DOMAIN: &[u8] = b"ptr.kv.taint.content.v1\0";
const DERIVATION_DOMAIN: &[u8] = b"ptr.kv.taint.derivation.v1\0";
const POD_OUTPUT_DOMAIN: &[u8] = b"ptr.kv.taint.pod-output.v1\0";

/// The Content subject of some bytes (generated text, a segment's UTF-8 text, a semantic
/// value's text, a Pod payload): SHA-256 over a domain tag and the length-prefixed bytes.
/// Every taint check that matches content uses this one definition, so the derivation,
/// segment-admission, Pod and SemDB paths cannot disagree on what "the same content" is.
pub(crate) fn content_subject(bytes: &[u8]) -> TaintSubject {
    let mut material = Vec::from(CONTENT_DOMAIN);
    put_len_prefixed(&mut material, bytes);
    (KvTaintSubjectKind::Content, integrity::sha256(&material))
}

/// The Derivation subject of a derived value: SHA-256 over a domain tag and its recipe digest.
/// A derived value is identified by how it was made, so a cold recomputation of the same
/// recipe names the same Derivation subject.
pub(crate) fn derivation_subject(recipe_digest: &Digest) -> TaintSubject {
    let mut material = Vec::from(DERIVATION_DOMAIN);
    material.extend_from_slice(recipe_digest);
    (KvTaintSubjectKind::Derivation, integrity::sha256(&material))
}

/// The PodOutput subject of a Pod output: SHA-256 over a domain tag and the length-prefixed
/// output identity bytes the Pod admission path names it by.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn pod_output_subject(output_identity: &[u8]) -> TaintSubject {
    let mut material = Vec::from(POD_OUTPUT_DOMAIN);
    put_len_prefixed(&mut material, output_identity);
    (KvTaintSubjectKind::PodOutput, integrity::sha256(&material))
}

/// The single action a run carries, if any. A run with several control events (an action beside
/// a Pod request, or more than one) is refused before anything is recorded.
pub(crate) fn run_action(run: &ModelRun) -> Result<Option<ActionIr>, RuntimeError> {
    let actions: Vec<&ActionIr> = run
        .events
        .iter()
        .filter_map(|event| match event {
            ModelEvent::ActionReady(action) => Some(action),
            _ => None,
        })
        .collect();
    let pod_requests = run
        .events
        .iter()
        .filter(|event| matches!(event, ModelEvent::PodRequested { .. }))
        .count();
    if !actions.is_empty() && pod_requests > 0 {
        return Err(RuntimeError::MixedControlEvents);
    }
    if actions.len() + pod_requests > 1 {
        return Err(RuntimeError::MultipleControlEvents {
            count: actions.len() + pod_requests,
        });
    }
    Ok(actions.first().map(|action| (*action).clone()))
}

/// Sorted, de-duplicated subjects in the canonical ledger order.
fn canonical(subjects: &[TaintSubject]) -> Vec<TaintSubject> {
    let mut subjects = subjects.to_vec();
    subjects.sort();
    subjects.dedup();
    subjects
}

impl PtrRuntime {
    /// Ingest the output of a [`ModelRun`], recording its taint with the action it carries.
    ///
    /// Exact reuse and no reuse behave as a plain run and append nothing. For an Approximate
    /// run the action's taint record is appended to the ledger BEFORE the [`ResumableRun`]
    /// is returned; the returned run carries the taint, and `authorize_action` on its action
    /// fails, also after a restart. The recipe is the fixed "unspecified request" recipe; use
    /// [`PtrRuntime::ingest_model_run_for`] to bind the request. At most one action and no Pod
    /// request may appear; the whole run is refused otherwise, before anything is recorded.
    ///
    /// # Errors
    /// [`RuntimeError::KvTaintRegistryFull`] when recording would exceed
    /// [`MAX_LIVE_TAINT_SUBJECTS`] (nothing is appended), or a ledger error.
    pub fn ingest_model_run(&mut self, run: &ModelRun) -> Result<ResumableRun, RuntimeError> {
        self.ingest_run(run, KvReuseTaint::Exact, &[], unspecified_recipe())
    }

    /// Like [`PtrRuntime::ingest_model_run`], additionally recording the request (a Request
    /// subject) and binding the taint to a recipe digest over the request's lineage, the
    /// producer and the generation parameters, which a later clean recomputation must
    /// reproduce to clear it.
    ///
    /// # Errors
    /// As [`PtrRuntime::ingest_model_run`].
    pub fn ingest_model_run_for(
        &mut self,
        request: &KvTaintRequest<'_>,
        run: &ModelRun,
    ) -> Result<ResumableRun, RuntimeError> {
        let extra = [request_subject(request.request.request_id())];
        self.ingest_run(run, KvReuseTaint::Exact, &extra, request_recipe(request))
    }

    /// Like [`PtrRuntime::ingest_model_run`] for a run whose input carried `input` taint (the
    /// request's raw text matched a live tainted content subject): the run's taint is joined
    /// with it, so an Exact run over laundered input records its action as Approximate. The
    /// request is recorded with it when the joined taint is Approximate.
    pub(crate) fn ingest_model_run_with_input(
        &mut self,
        request_id: &RequestId,
        input: KvReuseTaint,
        run: &ModelRun,
    ) -> Result<ResumableRun, RuntimeError> {
        let extra = [request_subject(request_id)];
        self.ingest_run(run, input, &extra, unspecified_recipe())
    }

    fn ingest_run(
        &mut self,
        run: &ModelRun,
        input: KvReuseTaint,
        extra: &[TaintSubject],
        recipe: Digest,
    ) -> Result<ResumableRun, RuntimeError> {
        let taint = run.taint().join(input);
        let action = run_action(run)?;
        if let KvReuseTaint::Approximate { report_digest } = taint {
            let mut subjects = extra.to_vec();
            if let Some(action) = action.as_ref() {
                subjects.push((KvTaintSubjectKind::Action, taint_identity(action)));
            }
            self.record_taint(&subjects, report_digest, recipe)?;
        }
        Ok(ResumableRun {
            model_events: run.events.clone(),
            observations: Vec::new(),
            action,
            kv_taint: taint,
        })
    }

    /// The live entry of a subject, if tainted.
    pub(crate) fn subject_taint(
        &self,
        kind: KvTaintSubjectKind,
        digest: [u8; 32],
    ) -> Option<TaintEntry> {
        self.taint.get(&(kind, digest)).copied()
    }

    /// Record `subjects` as tainted by `report` under `recipe`: bound check for the whole batch
    /// first, then taint-first append (in canonical chunks of at most
    /// [`MAX_KV_TAINT_SUBJECTS`]), the registry updated from each committed event. Subjects
    /// that are already live are not recorded again.
    pub(crate) fn record_taint(
        &mut self,
        subjects: &[TaintSubject],
        report: Digest,
        recipe: Digest,
    ) -> Result<(), RuntimeError> {
        let fresh: Vec<TaintSubject> = canonical(subjects)
            .into_iter()
            .filter(|subject| self.taint.get(subject).is_none())
            .collect();
        if fresh.is_empty() {
            return Ok(());
        }
        self.taint.room_for(fresh.len())?;
        for chunk in fresh.chunks(MAX_KV_TAINT_SUBJECTS) {
            self.append_taint_event(LedgerEvent::KvTaintCommitted {
                subjects: chunk.to_vec(),
                change: KvTaintChange::Record {
                    report_digest: report,
                    recipe_digest: recipe,
                },
            })?;
        }
        Ok(())
    }

    /// Clear `subjects`, which must all be live with exactly `recipe`; `recompute` is the digest
    /// of the clean recomputation that justifies it. Checks the whole batch before appending.
    /// This is the checked mechanism only: deciding WHEN a recomputation may clear is the
    /// caller's rule.
    pub(crate) fn clear_taint(
        &mut self,
        subjects: &[TaintSubject],
        recipe: Digest,
        recompute: Digest,
    ) -> Result<(), RuntimeError> {
        let subjects = canonical(subjects);
        if subjects.is_empty() {
            return Err(RuntimeError::KvTaint(KvTaintError::Unencodable));
        }
        for subject in &subjects {
            self.check_clear(subject, recipe)?;
        }
        for chunk in subjects.chunks(MAX_KV_TAINT_SUBJECTS) {
            self.append_taint_event(LedgerEvent::KvTaintCommitted {
                subjects: chunk.to_vec(),
                change: KvTaintChange::Clear {
                    recipe_digest: recipe,
                    recompute_digest: recompute,
                },
            })?;
        }
        Ok(())
    }

    fn check_clear(&self, subject: &TaintSubject, recipe: Digest) -> Result<(), RuntimeError> {
        match self.taint.get(subject) {
            None => Err(RuntimeError::KvTaint(KvTaintError::ClearSubjectNotLive)),
            Some(entry) if entry.recipe_digest != recipe => {
                Err(RuntimeError::KvTaint(KvTaintError::ClearRecipeMismatch))
            }
            Some(_) => Ok(()),
        }
    }

    fn append_taint_event(&mut self, event: LedgerEvent) -> Result<(), RuntimeError> {
        self.validate_kv_taint_event(&event)?;
        self.append_prepared(event, None)?;
        Ok(())
    }

    /// Replay-time and commit-time check of a taint record against the registry as of now.
    pub(crate) fn validate_kv_taint_event(&self, event: &LedgerEvent) -> Result<(), RuntimeError> {
        let LedgerEvent::KvTaintCommitted { subjects, change } = event else {
            return Ok(());
        };
        check_encodable(event).map_err(|_| RuntimeError::KvTaint(KvTaintError::Unencodable))?;
        match change {
            KvTaintChange::Record { .. } => {
                let fresh = subjects
                    .iter()
                    .filter(|subject| self.taint.get(subject).is_none())
                    .count();
                self.taint.room_for(fresh)
            }
            KvTaintChange::Clear { recipe_digest, .. } => subjects
                .iter()
                .try_for_each(|subject| self.check_clear(subject, *recipe_digest)),
        }
    }

    /// Apply a committed taint record to the registry. Live and replayed state both go through
    /// here. A subject already live keeps its first entry.
    pub(crate) fn apply_kv_taint_event(&mut self, event: &LedgerEvent) {
        let LedgerEvent::KvTaintCommitted { subjects, change } = event else {
            return;
        };
        for subject in subjects {
            match change {
                KvTaintChange::Record {
                    report_digest,
                    recipe_digest,
                } => {
                    self.taint.live.entry(*subject).or_insert(TaintEntry {
                        report_digest: *report_digest,
                        recipe_digest: *recipe_digest,
                    });
                }
                KvTaintChange::Clear { .. } => {
                    self.taint.live.remove(subject);
                }
            }
        }
    }

    /// Taint recorded for `action` by [`PtrRuntime::ingest_model_run`]; Exact if none.
    pub fn action_kv_taint(&self, action: &ActionIr) -> KvReuseTaint {
        self.subject_taint(KvTaintSubjectKind::Action, taint_identity(action))
            .map_or(KvReuseTaint::Exact, |entry| KvReuseTaint::Approximate {
                report_digest: entry.report_digest,
            })
    }

    /// Refuse an action whose provenance is Approximate, whether `taint` is passed by the
    /// caller or was recorded at ingest.
    pub(crate) fn check_kv_taint(
        &self,
        action: &ActionIr,
        taint: KvReuseTaint,
    ) -> Result<(), RuntimeError> {
        match taint.join(self.action_kv_taint(action)) {
            KvReuseTaint::Exact => Ok(()),
            KvReuseTaint::Approximate { report_digest } => {
                Err(RuntimeError::ApproximateStateAtEffectBoundary { report_digest })
            }
        }
    }

    /// Like [`PtrRuntime::authorize_action`] for an action whose taint the caller holds,
    /// for example one taken from a [`ResumableRun`] or derived from tainted values.
    pub fn authorize_action_with_taint(
        &self,
        action: &ActionIr,
        taint: KvReuseTaint,
    ) -> Result<(), RuntimeError> {
        self.check_kv_taint(action, taint)?;
        self.authorize_action(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_config::PtrConfig;
    use ptr_ledger::CommittedEvent;
    use ptr_model_api::{KvReuseMode, KvReusePolicy, KvReuseReport};

    const REPORT: Digest = [7; 32];
    const RECIPE: Digest = [8; 32];
    const OTHER_RECIPE: Digest = [9; 32];
    const RECOMPUTE: Digest = [10; 32];

    fn subject(n: u8) -> TaintSubject {
        (KvTaintSubjectKind::Content, [n; 32])
    }

    fn taint_events(runtime: &PtrRuntime) -> Vec<CommittedEvent> {
        runtime
            .committed_events()
            .iter()
            .filter(|committed| matches!(committed.event, LedgerEvent::KvTaintCommitted { .. }))
            .cloned()
            .collect()
    }

    fn approximate_run(action: &ActionIr) -> ModelRun {
        ModelRun {
            events: vec![ModelEvent::ActionReady(action.clone())],
            kv_report: Some(
                KvReuseReport::new(
                    4,
                    KvReuseMode::Approximate,
                    8,
                    KvReusePolicy::AllowApproximate,
                )
                .unwrap(),
            ),
        }
    }

    fn action(payload: &[u8]) -> ActionIr {
        ActionIr {
            operation: "write".into(),
            target: "artifact:a".into(),
            capability: "file.write".into(),
            effect: ptr_types::Effect::Mutation,
            input_type: "Bytes".into(),
            generation: ptr_types::Generation(1),
            revision: ptr_types::Revision(0),
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn record_then_clear_updates_the_registry_through_the_ledger() {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        let subjects = [subject(1), subject(2)];
        runtime.record_taint(&subjects, REPORT, RECIPE).unwrap();
        for s in subjects {
            assert_eq!(
                runtime.subject_taint(s.0, s.1),
                Some(TaintEntry {
                    report_digest: REPORT,
                    recipe_digest: RECIPE
                })
            );
        }
        assert_eq!(taint_events(&runtime).len(), 1);
        runtime.clear_taint(&subjects, RECIPE, RECOMPUTE).unwrap();
        assert_eq!(runtime.subject_taint(subjects[0].0, subjects[0].1), None);
        assert_eq!(taint_events(&runtime).len(), 2);
    }

    #[test]
    fn already_live_subjects_are_not_recorded_again() {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime.record_taint(&[subject(1)], REPORT, RECIPE).unwrap();
        runtime.record_taint(&[subject(1)], REPORT, RECIPE).unwrap();
        assert_eq!(taint_events(&runtime).len(), 1);
        // A different entry for a live subject is not recorded either; the first one stays.
        runtime
            .record_taint(&[subject(1)], [1; 32], OTHER_RECIPE)
            .unwrap();
        assert_eq!(taint_events(&runtime).len(), 1);
        assert_eq!(
            runtime
                .subject_taint(subject(1).0, subject(1).1)
                .unwrap()
                .recipe_digest,
            RECIPE
        );
    }

    #[test]
    fn more_than_sixteen_subjects_split_into_canonical_records() {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        // Unsorted input with a duplicate.
        let mut subjects: Vec<TaintSubject> = (1..=40u8).rev().map(subject).collect();
        subjects.push(subject(5));
        runtime.record_taint(&subjects, REPORT, RECIPE).unwrap();
        let events = taint_events(&runtime);
        assert_eq!(events.len(), 3);
        let mut all = Vec::new();
        for committed in &events {
            let LedgerEvent::KvTaintCommitted { subjects, .. } = &committed.event else {
                unreachable!()
            };
            assert!(subjects.len() <= MAX_KV_TAINT_SUBJECTS);
            all.extend(subjects.iter().copied());
        }
        let mut sorted = all.clone();
        sorted.sort();
        assert_eq!(all, sorted);
        assert_eq!(all.len(), 40);
        assert_eq!(runtime.taint.live.len(), 40);
    }

    #[test]
    fn clear_refuses_a_non_live_subject_and_a_recipe_mismatch_and_appends_nothing() {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime
            .record_taint(&[subject(1), subject(2)], REPORT, RECIPE)
            .unwrap();
        let before = runtime.committed_events().len();
        assert_eq!(
            runtime.clear_taint(&[subject(3)], RECIPE, RECOMPUTE),
            Err(RuntimeError::KvTaint(KvTaintError::ClearSubjectNotLive))
        );
        assert_eq!(
            runtime.clear_taint(&[subject(1)], OTHER_RECIPE, RECOMPUTE),
            Err(RuntimeError::KvTaint(KvTaintError::ClearRecipeMismatch))
        );
        // One bad subject refuses the whole batch.
        assert_eq!(
            runtime.clear_taint(&[subject(1), subject(3)], RECIPE, RECOMPUTE),
            Err(RuntimeError::KvTaint(KvTaintError::ClearSubjectNotLive))
        );
        assert_eq!(
            runtime.clear_taint(&[], RECIPE, RECOMPUTE),
            Err(RuntimeError::KvTaint(KvTaintError::Unencodable))
        );
        assert_eq!(runtime.committed_events().len(), before);
        assert_eq!(runtime.taint.live.len(), 2);
    }

    #[test]
    fn a_full_registry_refuses_a_batch_before_appending_and_clear_frees_capacity() {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime.taint.limit = 3;
        runtime
            .record_taint(&[subject(1), subject(2), subject(3)], REPORT, RECIPE)
            .unwrap();
        let before = runtime.committed_events().len();
        // The whole batch is checked first: one new subject over the bound refuses all of it.
        assert_eq!(
            runtime.record_taint(&[subject(3), subject(4)], REPORT, RECIPE),
            Err(RuntimeError::KvTaintRegistryFull { limit: 3 })
        );
        let tainted = action(b"full");
        assert_eq!(
            runtime.ingest_model_run(&approximate_run(&tainted)),
            Err(RuntimeError::KvTaintRegistryFull { limit: 3 })
        );
        assert_eq!(runtime.committed_events().len(), before);
        assert_eq!(runtime.action_kv_taint(&tainted), KvReuseTaint::Exact);
        // Already-live subjects need no room.
        runtime.record_taint(&[subject(3)], REPORT, RECIPE).unwrap();
        runtime
            .clear_taint(&[subject(1)], RECIPE, RECOMPUTE)
            .unwrap();
        runtime.record_taint(&[subject(4)], REPORT, RECIPE).unwrap();
        assert_eq!(runtime.taint.live.len(), 3);
    }

    #[test]
    fn replay_refuses_a_clear_of_a_non_live_subject_or_another_recipe() {
        let clear = |digest: Digest, recipe: Digest| CommittedEvent {
            index: ptr_types::CommitIndex(2),
            event: LedgerEvent::KvTaintCommitted {
                subjects: vec![(KvTaintSubjectKind::Content, digest)],
                change: KvTaintChange::Clear {
                    recipe_digest: recipe,
                    recompute_digest: RECOMPUTE,
                },
            },
        };
        let record = CommittedEvent {
            index: ptr_types::CommitIndex(1),
            event: LedgerEvent::KvTaintCommitted {
                subjects: vec![subject(1)],
                change: KvTaintChange::Record {
                    report_digest: REPORT,
                    recipe_digest: RECIPE,
                },
            },
        };
        let replay = |events: &[CommittedEvent]| PtrRuntime::replay(PtrConfig::default(), events);
        assert!(replay(&[record.clone(), clear([1; 32], RECIPE)]).is_ok());
        assert_eq!(
            replay(&[clear([1; 32], RECIPE)]).err(),
            Some(RuntimeError::KvTaint(KvTaintError::ClearSubjectNotLive))
        );
        assert_eq!(
            replay(&[record.clone(), clear([2; 32], RECIPE)]).err(),
            Some(RuntimeError::KvTaint(KvTaintError::ClearSubjectNotLive))
        );
        assert_eq!(
            replay(&[record, clear([1; 32], OTHER_RECIPE)]).err(),
            Some(RuntimeError::KvTaint(KvTaintError::ClearRecipeMismatch))
        );
    }

    #[test]
    fn replay_refuses_a_record_beyond_the_bound() {
        // The bound is the registry's own; shrink it on a fresh runtime and feed the records
        // through the replay validation.
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime.taint.limit = 1;
        runtime.record_taint(&[subject(1)], REPORT, RECIPE).unwrap();
        let over = LedgerEvent::KvTaintCommitted {
            subjects: vec![subject(2)],
            change: KvTaintChange::Record {
                report_digest: REPORT,
                recipe_digest: RECIPE,
            },
        };
        assert_eq!(
            runtime.validate_lifecycle_event(&over),
            Err(RuntimeError::KvTaintRegistryFull { limit: 1 })
        );
    }

    fn state(entries: &[(&str, String)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }

    fn hex(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    #[test]
    fn the_registry_rebuilds_from_projection_entries_and_rejects_malformed_ones() {
        let key = |name: &str, byte: u8| format!("kv-taint:{name}:{}", hex(byte));
        let good = state(&[
            (&key("action", 1), format!("approx:{}:{}", hex(2), hex(3))),
            (&key("content", 4), format!("cleared:{}", hex(5))),
            (
                &key("pod-output", 6),
                format!("approx:{}:{}", hex(7), hex(8)),
            ),
            ("unrelated", "x".into()),
        ]);
        let registry = TaintRegistry::from_materialized(&good).unwrap();
        assert_eq!(registry.live.len(), 2);
        assert!(registry
            .live
            .contains_key(&(KvTaintSubjectKind::Action, [1; 32])));
        assert!(!registry
            .live
            .contains_key(&(KvTaintSubjectKind::Content, [4; 32])));

        let bad_cases: [(String, String); 9] = [
            (key("nonsense", 1), format!("approx:{}:{}", hex(2), hex(3))),
            ("kv-taint:action".into(), format!("cleared:{}", hex(5))),
            (
                format!("kv-taint:action:{}", "AB".repeat(32)),
                format!("cleared:{}", hex(5)),
            ),
            (key("action", 0), format!("cleared:{}", hex(5))),
            (key("action", 1), "approx:zz".into()),
            (key("action", 1), format!("approx:{}", hex(2))),
            (key("action", 1), format!("approx:{}:{}", hex(2), hex(0))),
            (key("action", 1), format!("cleared:{}x", hex(5))),
            (key("action", 1), "other".into()),
        ];
        for (k, v) in bad_cases {
            let result = TaintRegistry::from_materialized(&state(&[(&k, v.clone())]));
            assert!(
                matches!(
                    result,
                    Err(RuntimeError::KvTaint(
                        KvTaintError::MalformedStateEntry { .. }
                    ))
                ),
                "{k} = {v}"
            );
        }
    }

    #[test]
    fn a_restored_state_beyond_the_bound_fails_closed() {
        let mut entries = BTreeMap::new();
        for n in 1..=3u8 {
            entries.insert(
                format!("kv-taint:request:{}", hex(n)),
                format!("approx:{}:{}", hex(1), hex(2)),
            );
        }
        assert!(TaintRegistry::from_materialized_within(&entries, 3).is_ok());
        assert_eq!(
            TaintRegistry::from_materialized_within(&entries, 2).err(),
            Some(RuntimeError::KvTaintRegistryFull { limit: 2 })
        );
    }

    #[test]
    fn shared_subject_digests_are_kind_tagged_domain_separated_and_input_bound() {
        let content = content_subject(b"abc");
        assert_eq!(content.0, KvTaintSubjectKind::Content);
        assert_eq!(content, content_subject(b"abc"));
        assert_ne!(content.1, content_subject(b"abd").1);
        assert_ne!(content.1, integrity::sha256(b"abc"));
        let pod = pod_output_subject(b"abc");
        assert_eq!(pod.0, KvTaintSubjectKind::PodOutput);
        assert_ne!(pod.1, content.1);
        let derivation = derivation_subject(&[7; 32]);
        assert_eq!(derivation.0, KvTaintSubjectKind::Derivation);
        assert_eq!(derivation, derivation_subject(&[7; 32]));
        assert_ne!(derivation.1, derivation_subject(&[8; 32]).1);
        assert_ne!(derivation.1, [7; 32]);
    }

    #[test]
    fn request_and_recipe_digests_are_domain_separated_and_input_bound() {
        let a = request_subject(&RequestId::from("r1"));
        let b = request_subject(&RequestId::from("r2"));
        assert_ne!(a, b);
        assert_eq!(a.0, KvTaintSubjectKind::Request);
        assert_ne!(unspecified_recipe(), [0; 32]);
    }
}
