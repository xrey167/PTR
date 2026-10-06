//! KV-reuse taint at the hard effect boundary (doc 36, phase P2a).
//!
//! A model run produced under Approximate KV reuse may only yield candidates. The runtime
//! records the taint of every action it ingests from such a run, keyed by the action's
//! audit digest, and `authorize_action` refuses a recorded action with
//! [`RuntimeError::ApproximateStateAtEffectBoundary`], so the check sits before any
//! `EffectAttempted` record for callers that authorize first (the HTTP path does).
//! Keying by digest means the bare `ActionIr` that callers hold stays enforced although it
//! carries no taint field.
//!
//! The registry is in memory, bounded, and fails closed when full. It is not rebuilt after
//! a restart and an identical clean action stays refused while its tainted twin is
//! recorded; both are open (see component.toml).

use crate::execution::action_digest;
use crate::{PtrRuntime, ResumableRun, RuntimeError};
use ptr_model_api::{KvReuseTaint, ModelEvent, ModelRun};
use ptr_types::ActionIr;
use std::collections::BTreeMap;

/// Most distinct tainted actions the runtime remembers.
pub const MAX_TAINTED_ACTIONS: usize = 4096;

#[derive(Debug, Default)]
pub(crate) struct TaintedActions {
    by_digest: BTreeMap<[u8; 32], KvReuseTaint>,
}

impl TaintedActions {
    fn taint_of(&self, action: &ActionIr) -> KvReuseTaint {
        self.by_digest
            .get(&action_digest(action))
            .copied()
            .unwrap_or(KvReuseTaint::Exact)
    }

    fn record(&mut self, action: &ActionIr, taint: KvReuseTaint) -> Result<(), RuntimeError> {
        let digest = action_digest(action);
        if !self.by_digest.contains_key(&digest) && self.by_digest.len() >= MAX_TAINTED_ACTIONS {
            return Err(RuntimeError::KvTaintRegistryFull {
                limit: MAX_TAINTED_ACTIONS,
            });
        }
        let joined = self
            .by_digest
            .get(&digest)
            .copied()
            .unwrap_or_default()
            .join(taint);
        self.by_digest.insert(digest, joined);
        Ok(())
    }
}

impl PtrRuntime {
    /// Ingest the output of a [`ModelRun`], recording its taint with the action it carries.
    ///
    /// Exact reuse and no reuse behave as a plain run. For an Approximate run the returned
    /// [`ResumableRun`] carries the taint and the action is remembered as tainted; the run
    /// can still be inspected, but `authorize_action` on its action fails. At most one
    /// action and no Pod request may appear; the whole run is refused otherwise, before
    /// anything is recorded.
    pub fn ingest_model_run(&mut self, run: &ModelRun) -> Result<ResumableRun, RuntimeError> {
        let taint = run.taint();
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
        let action = actions.first().map(|action| (*action).clone());
        if let (Some(action), true) = (action.as_ref(), taint.is_approximate()) {
            self.tainted_actions.record(action, taint)?;
        }
        Ok(ResumableRun {
            model_events: run.events.clone(),
            observations: Vec::new(),
            action,
            kv_taint: taint,
        })
    }

    /// Taint recorded for `action` by [`PtrRuntime::ingest_model_run`]; Exact if none.
    pub fn action_kv_taint(&self, action: &ActionIr) -> KvReuseTaint {
        self.tainted_actions.taint_of(action)
    }

    /// Refuse an action whose provenance is Approximate, whether `taint` is passed by the
    /// caller or was recorded at ingest.
    pub(crate) fn check_kv_taint(
        &self,
        action: &ActionIr,
        taint: KvReuseTaint,
    ) -> Result<(), RuntimeError> {
        match taint.join(self.tainted_actions.taint_of(action)) {
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
