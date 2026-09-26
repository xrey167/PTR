//! Durable semantic publication uses the same ordered log as lifecycle changes.
//! No acknowledgments, observations or model continuation precede a successful
//! append. This reconstructs semantic state, not neural or execution checkpoints.
use super::{PtrRuntime, RuntimeError};
use ptr_events::RuntimeEvent;
use ptr_ledger::LedgerEvent;
use ptr_protocol::TypedPayload;
use ptr_semdb::{
    PreparedDelta, PreparedView, SemanticDelta, SemanticError, SemanticPayload, SemanticSnapshot,
};
use ptr_verifier::{VerificationReport, VerificationStatus};

use crate::execution::RequiredVerification;
use ptr_types::{CommitIndex, Generation, PodId, RequestId, Revision, Validity};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticCommit {
    pub revision: Revision,
    pub affected: BTreeSet<String>,
    /// None means a validated no-op; it did not write a new journal record.
    pub commit_index: Option<CommitIndex>,
}

/// Lifecycle generations a commit relied on that are no longer live when it
/// would append; the commit was refused and nothing was appended.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaleReliance {
    /// Every relied-on target that is not live at the generation relied on.
    pub targets: BTreeMap<String, StaleTarget>,
}

/// One relied-on target that is no longer live at the relied generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaleTarget {
    /// The generation the commit relied on.
    pub relied: Generation,
    /// What [`PtrRuntime::generation_validity`] answers for it now:
    /// `Revoked`, `Superseded`, or `None` for a generation the lifecycle
    /// authority does not know.
    pub validity: Option<Validity>,
}

impl StaleReliance {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        "PTR_RUNTIME_STALE_RELIANCE"
    }
}

impl fmt::Display for StaleReliance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: relied-on generations no longer live: {:?}",
            self.code(),
            self.targets.keys().collect::<Vec<_>>()
        )
    }
}

impl std::error::Error for StaleReliance {}

pub fn request_raw_key(request: &RequestId) -> String {
    format!("request:{request}:raw")
}

/// Length-delimited components prevent request/Pod separator collisions.
pub fn pod_output_key(request: &RequestId, pod: &PodId) -> String {
    format!(
        "pod-output:{}:{request}:{}:{pod}",
        request.0.len(),
        pod.0.len()
    )
}

impl PtrRuntime {
    /// Optimistic base-revision check; source updates and dependency metadata
    /// enter a single durable record. Invalid input never fences a healthy writer.
    pub fn apply_semantic_delta(
        &mut self,
        expected: Revision,
        delta: SemanticDelta,
    ) -> Result<SemanticCommit, RuntimeError> {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        if expected != self.revision() {
            return Err(RuntimeError::Semantic(SemanticError::RevisionMismatch {
                expected: self.revision(),
                actual: expected,
            }));
        }
        let encoded_delta = delta.encode().map_err(RuntimeError::Semantic)?;
        let prepared = self
            .semdb
            .prepare_delta(delta)
            .map_err(RuntimeError::Semantic)?;
        let revision = prepared.revision();
        let affected = prepared.affected().clone();
        if revision == expected {
            return Ok(SemanticCommit {
                revision,
                affected,
                commit_index: None,
            });
        }
        let event = LedgerEvent::SemanticDeltaCommitted {
            base_revision: expected,
            revision,
            encoded_delta,
        };
        let index = self.append_prepared(event, Some(prepared))?;
        Ok(SemanticCommit {
            revision,
            affected,
            commit_index: Some(index),
        })
    }

    /// Prepare `delta`, let `verify` judge the exact state it would publish,
    /// and append that same prepared state only if the report passes at the
    /// required level with no hard finding.
    ///
    /// This is the commit path for work produced elsewhere, such as a
    /// human-approved plan, where no score may stand in for verification: a
    /// delta whose report is not `Pass`, is below `required`, or carries a
    /// hard finding is refused before any byte is appended. The
    /// verifier sees a [`PreparedView`] of the values and the dependency sets
    /// the delta would publish, which cannot be mistaken for a published
    /// snapshot, so it can refuse an undeclared or inappropriate dependency as
    /// well as a wrong value. A validated no-op is returned without appending,
    /// but only after it too has passed verification.
    ///
    /// It checks no lifecycle generation. Work that relied on some, such as a
    /// certified agent branch, must be committed through
    /// [`Self::apply_certified_semantic_delta`] with the generations it
    /// relied on; this is that call with none.
    ///
    /// Returns the published revision and affected keys, with no commit index
    /// for a verified no-op. A changing delta also updates materialized state
    /// and emits a commit-applied event.
    ///
    /// # Errors
    /// Refuses a fenced runtime, a stale `expected` revision, semantic encoding
    /// or preparation errors, and a rejected verification report before append.
    /// Ledger append and state-application errors propagate; an error after
    /// append begins leaves execution fenced because the commit is uncertain.
    pub fn apply_verified_semantic_delta<F>(
        &mut self,
        expected: Revision,
        delta: SemanticDelta,
        required: RequiredVerification,
        verify: F,
    ) -> Result<SemanticCommit, RuntimeError>
    where
        F: FnOnce(&PreparedView<'_>) -> VerificationReport,
    {
        self.apply_certified_semantic_delta(expected, delta, &BTreeMap::new(), required, verify)
    }

    /// [`Self::apply_verified_semantic_delta`] for work that relied on
    /// lifecycle generations: after verification passes and immediately
    /// before it would append, it asks [`Self::generation_validity`] about
    /// every `(target, generation)` in `relied` and refuses the delta unless
    /// each is `Live`.
    ///
    /// This is the commit path for a certified agent branch: a
    /// `ptr_branch::MergePlan` yields exactly its `expected`, `delta` and
    /// `relied` arguments, and certification checked those generations
    /// against the lifecycle state it was given, which a revocation or
    /// supersession committed since does not reach. The check and the append
    /// happen in this one `&mut self` call, so no lifecycle change can come
    /// between them. It checks only the generations passed: a caller that
    /// passes fewer than its work relied on is not stopped, and nothing here
    /// can tell.
    ///
    /// # Errors
    /// Everything [`Self::apply_verified_semantic_delta`] refuses, and
    /// [`RuntimeError::StaleReliance`] naming every relied-on target that is
    /// revoked, superseded or unknown at the generation relied on. A stale
    /// reliance refuses a no-op too, and nothing is appended.
    pub fn apply_certified_semantic_delta<F>(
        &mut self,
        expected: Revision,
        delta: SemanticDelta,
        relied: &BTreeMap<String, Generation>,
        required: RequiredVerification,
        verify: F,
    ) -> Result<SemanticCommit, RuntimeError>
    where
        F: FnOnce(&PreparedView<'_>) -> VerificationReport,
    {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        if expected != self.revision() {
            return Err(RuntimeError::Semantic(SemanticError::RevisionMismatch {
                expected: self.revision(),
                actual: expected,
            }));
        }
        let encoded_delta = delta.encode().map_err(RuntimeError::Semantic)?;
        let prepared = self
            .semdb
            .prepare_delta(delta)
            .map_err(RuntimeError::Semantic)?;
        let report = verify(&prepared.view());
        let hard_findings = report.findings.iter().filter(|f| f.hard).count();
        if report.status != VerificationStatus::Pass
            || !required.accepts(report.level)
            || hard_findings > 0
        {
            return Err(RuntimeError::DeltaVerificationRejected {
                status: report.status,
                level: report.level,
                hard_findings,
            });
        }
        self.check_reliance(relied)?;
        let revision = prepared.revision();
        let affected = prepared.affected().clone();
        if revision == expected {
            return Ok(SemanticCommit {
                revision,
                affected,
                commit_index: None,
            });
        }
        let event = LedgerEvent::SemanticDeltaCommitted {
            base_revision: expected,
            revision,
            encoded_delta,
        };
        let index = self.append_prepared(event, Some(prepared))?;
        Ok(SemanticCommit {
            revision,
            affected,
            commit_index: Some(index),
        })
    }

    /// Refuse unless every relied-on generation is live now.
    fn check_reliance(&self, relied: &BTreeMap<String, Generation>) -> Result<(), RuntimeError> {
        let targets: BTreeMap<String, StaleTarget> = relied
            .iter()
            .filter_map(|(target, &generation)| {
                let validity = self.generation_validity(target, generation);
                (validity != Some(Validity::Live)).then(|| {
                    (
                        target.clone(),
                        StaleTarget {
                            relied: generation,
                            validity,
                        },
                    )
                })
            })
            .collect();
        if targets.is_empty() {
            Ok(())
        } else {
            Err(RuntimeError::StaleReliance(StaleReliance { targets }))
        }
    }

    pub fn ingest_text(
        &mut self,
        request: RequestId,
        text: impl Into<String>,
    ) -> Result<Revision, RuntimeError> {
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert(request_raw_key(&request), text.into().into());
        let committed = self.apply_semantic_delta(self.revision(), delta)?;
        self.emit(RuntimeEvent::RequestStarted(request));
        self.emit(RuntimeEvent::SnapshotOpened(committed.revision));
        Ok(committed.revision)
    }

    /// Semantic identity only. Lifecycle/authorization checks are independent;
    /// this must not be treated as a neural-checkpoint or execution permit.
    pub fn snapshot_is_current(&self, snapshot: &SemanticSnapshot) -> bool {
        !self.execution.is_fenced() && !self.semdb.is_stale(snapshot)
    }

    pub(super) fn promote_pod_output(
        &mut self,
        request: &RequestId,
        pod: &PodId,
        output: &TypedPayload,
    ) -> Result<Revision, RuntimeError> {
        let key = pod_output_key(request, pod);
        let mut delta = SemanticDelta::default();
        delta.upserts.insert(
            key.clone(),
            SemanticPayload {
                type_id: output.type_id.clone(),
                source: pod.to_string(),
                bytes: output.bytes.clone(),
            }
            .into(),
        );
        delta
            .dependencies
            .insert(key, [request_raw_key(request)].into());
        self.apply_semantic_delta(self.revision(), delta)
            .map(|commit| commit.revision)
    }

    pub(super) fn prepare_semantic_event(
        &self,
        event: &LedgerEvent,
    ) -> Result<Option<PreparedDelta>, RuntimeError> {
        let LedgerEvent::SemanticDeltaCommitted {
            base_revision,
            revision,
            encoded_delta,
        } = event
        else {
            return Ok(None);
        };
        if *base_revision != self.revision() {
            return Err(RuntimeError::Semantic(SemanticError::RevisionMismatch {
                expected: self.revision(),
                actual: *base_revision,
            }));
        }
        let delta = SemanticDelta::decode(encoded_delta).map_err(RuntimeError::Semantic)?;
        let prepared = self
            .semdb
            .prepare_delta(delta)
            .map_err(RuntimeError::Semantic)?;
        if prepared.revision() == self.revision() {
            return Err(RuntimeError::Semantic(SemanticError::NoChangeRecord));
        }
        if prepared.revision() != *revision {
            return Err(RuntimeError::Semantic(SemanticError::RevisionMismatch {
                expected: prepared.revision(),
                actual: *revision,
            }));
        }
        Ok(Some(prepared))
    }
}
