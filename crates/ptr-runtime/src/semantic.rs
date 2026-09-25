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
use ptr_types::{CommitIndex, PodId, RequestId, Revision};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticCommit {
    pub revision: Revision,
    pub affected: BTreeSet<String>,
    /// None means a validated no-op; it did not write a new journal record.
    pub commit_index: Option<CommitIndex>,
}

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
    /// This is the commit path for work produced elsewhere — a certified agent
    /// branch, a human-approved plan — where no score may stand in for
    /// verification: a delta whose report is not `Pass`, is below `required`,
    /// or carries a hard finding is refused before any byte is appended. The
    /// verifier sees a [`PreparedView`], which cannot be mistaken for a
    /// published snapshot. A validated no-op is returned without appending,
    /// but only after it too has passed verification.
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
