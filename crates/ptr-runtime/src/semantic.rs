//! Durable semantic publication uses the same ordered log as lifecycle changes.
//! No acknowledgments, observations or model continuation precede a successful
//! append. This reconstructs semantic state, not neural or execution checkpoints.
use super::{PtrRuntime, RuntimeError};
use crate::merge::{self, ChangeOrigin, SemanticChange};
use ptr_events::RuntimeEvent;
use ptr_ledger::{Attestation, LedgerEvent, SemanticOrigin};
use ptr_protocol::TypedPayload;
use ptr_semdb::{
    is_ingress_key, PreparedDelta, SemanticDelta, SemanticError, SemanticPayload, SemanticSnapshot,
    SemanticValue,
};
use ptr_verifier::VerificationStatus;

use ptr_types::{
    CommitIndex, Generation, PodId, PrincipalId, RequestId, Revision, Validity, VerificationLevel,
};
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
    /// Ingress: `delta`, in the exact shape `origin` fixes (rule R2 of
    /// [`Self::validate_semantic_origin`]), committed at the current revision
    /// with no verifier, since raw input is not something an interpreter may
    /// refuse. Only [`Self::ingest_text`] and Pod promotion reach it.
    ///
    /// Optimistic base-revision check; source updates and dependency metadata
    /// enter a single durable record. Invalid input never fences a healthy
    /// writer.
    fn apply_ingress(
        &mut self,
        origin: SemanticOrigin,
        delta: SemanticDelta,
    ) -> Result<SemanticCommit, RuntimeError> {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        let expected = self.revision();
        self.validate_semantic_origin(None, &delta, &origin)?;
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
            origin,
        };
        let index = self.append_semantic(event, prepared)?;
        Ok(SemanticCommit {
            revision,
            affected,
            commit_index: Some(index),
        })
    }

    /// A host write: `delta`, prepared against revision `expected`, judged
    /// by every verifier of the installed [`SemanticGrant`](crate::SemanticGrant)
    /// and appended only if they admit it, recorded as written by
    /// `principal` and admitted by those verifiers.
    ///
    /// In order, it refuses:
    /// 1. a fenced runtime ([`RuntimeError::ExecutionFenced`]);
    /// 2. a runtime with no grant ([`RuntimeError::NoSemanticGrant`]);
    /// 3. a grant that does not allow host writes
    ///    ([`RuntimeError::HostWritesNotGranted`]);
    /// 4. a principal that is not an identifier of at most
    ///    [`MAX_PROVENANCE_TEXT`](crate::merge::MAX_PROVENANCE_TEXT) bytes
    ///    ([`RuntimeError::InvalidProvenanceText`]);
    /// 5. a stale `expected` revision;
    /// 6. an ingress key as an upsert, a removal or a derived (dependency)
    ///    key ([`RuntimeError::ReservedSemanticNamespace`]); an ingress key
    ///    may be a dependency input;
    /// 7. a delta that does not encode or prepare;
    /// 8. a change the grant's verifiers do not admit
    ///    ([`RuntimeError::SemanticVerificationRejected`], naming the hard
    ///    findings by code, never by message), or a report that does not fit
    ///    a record ([`RuntimeError::InvalidVerificationReport`]).
    ///
    /// The verifiers see a [`SemanticChange`] of the state before and after,
    /// the delta, the keys it invalidates and the principal. A validated
    /// no-op is returned without appending, after it too has been admitted.
    /// Each verifier's result is emitted as a
    /// [`RuntimeEvent::VerifierResult`], passed when it reported `Pass` at a
    /// level the grant accepts with no hard finding.
    ///
    /// It checks no lifecycle generation; work that relied on some is
    /// committed through [`Self::apply_certified_semantic_delta`].
    ///
    /// The earlier form, which took a requirement and a closure instead of
    /// a principal and ran no installed verifier, no longer compiles:
    ///
    /// ```compile_fail
    /// use ptr_runtime::{execution::RequiredVerification, PtrRuntime};
    /// fn write(runtime: &mut PtrRuntime, delta: ptr_semdb::SemanticDelta, report: ptr_verifier::VerificationReport) {
    ///     let principal = ptr_types::PrincipalId::from("operator");
    ///     let _ = runtime.apply_verified_semantic_delta(runtime.revision(), delta, RequiredVerification::Deterministic, |_| report);
    /// }
    /// ```
    ///
    /// ```
    /// use ptr_runtime::{execution::RequiredVerification, PtrRuntime};
    /// fn write(runtime: &mut PtrRuntime, delta: ptr_semdb::SemanticDelta, report: ptr_verifier::VerificationReport) {
    ///     let principal = ptr_types::PrincipalId::from("operator");
    ///     let _ = runtime.apply_verified_semantic_delta(runtime.revision(), delta, &principal);
    /// }
    /// ```
    ///
    /// # Errors
    /// The refusals above, before anything is appended. Ledger append and
    /// state-application errors propagate; an error after append begins
    /// leaves execution fenced because the commit is uncertain.
    pub fn apply_verified_semantic_delta(
        &mut self,
        expected: Revision,
        delta: SemanticDelta,
        principal: &PrincipalId,
    ) -> Result<SemanticCommit, RuntimeError> {
        self.host_write(expected, delta, &BTreeMap::new(), principal)
    }

    /// [`Self::apply_verified_semantic_delta`] for work that relied on
    /// lifecycle generations: after the grant admits the change and
    /// immediately before it would append, it asks
    /// [`Self::generation_validity`] about every `(target, generation)` in
    /// `relied` and refuses the delta unless each is `Live`.
    ///
    /// A certified agent branch's plan is committed this way today, as a
    /// host write: `ptr_branch::MergePlan`'s accessors give exactly its
    /// `expected`, `delta` and `relied` arguments, and certification checked
    /// those generations against the lifecycle state it was given, which a
    /// revocation or supersession committed since does not reach. The check
    /// and the append happen in this one `&mut self` call, so no lifecycle
    /// change can come between them. It checks only the generations passed:
    /// a caller that passes fewer than its work relied on is not stopped, and
    /// nothing here can tell.
    ///
    /// # Errors
    /// Everything [`Self::apply_verified_semantic_delta`] refuses, and
    /// [`RuntimeError::StaleReliance`] naming every relied-on target that is
    /// revoked, superseded or unknown at the generation relied on. A stale
    /// reliance refuses a no-op too, and nothing is appended.
    pub fn apply_certified_semantic_delta(
        &mut self,
        expected: Revision,
        delta: SemanticDelta,
        relied: &BTreeMap<String, Generation>,
        principal: &PrincipalId,
    ) -> Result<SemanticCommit, RuntimeError> {
        self.host_write(expected, delta, relied, principal)
    }

    fn host_write(
        &mut self,
        expected: Revision,
        delta: SemanticDelta,
        relied: &BTreeMap<String, Generation>,
        principal: &PrincipalId,
    ) -> Result<SemanticCommit, RuntimeError> {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        let grant = self
            .semantic_grant
            .as_ref()
            .ok_or(RuntimeError::NoSemanticGrant)?;
        if !grant.host_writes() {
            return Err(RuntimeError::HostWritesNotGranted);
        }
        if !merge::valid_provenance(&principal.0) {
            return Err(RuntimeError::InvalidProvenanceText { field: "principal" });
        }
        if expected != self.revision() {
            return Err(RuntimeError::Semantic(SemanticError::RevisionMismatch {
                expected: self.revision(),
                actual: expected,
            }));
        }
        if let Some(key) = written_ingress_key(&delta) {
            return Err(RuntimeError::ReservedSemanticNamespace {
                key: key.to_owned(),
            });
        }
        let encoded_delta = delta.encode().map_err(RuntimeError::Semantic)?;
        let prepared = self
            .semdb
            .prepare_delta(delta)
            .map_err(RuntimeError::Semantic)?;
        let verdict = {
            let change = SemanticChange {
                base: expected,
                next: prepared.revision(),
                before: self
                    .semdb
                    .base_view(&prepared)
                    .map_err(RuntimeError::Semantic)?,
                after: prepared.view(),
                delta: prepared.delta(),
                affected: prepared.affected(),
                origin: ChangeOrigin::Host { principal },
            };
            grant.judge(&change)?
        };
        for (name, report) in &verdict.reports {
            let passed = report.status == VerificationStatus::Pass
                && verdict.required.accepts(report.level)
                && !report.findings.iter().any(|finding| finding.hard);
            self.emit(RuntimeEvent::VerifierResult {
                verifier: (*name).to_owned(),
                passed,
            });
        }
        if !verdict.admitted() {
            return Err(RuntimeError::SemanticVerificationRejected(
                verdict.refusal(),
            ));
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
        let origin = SemanticOrigin::Host {
            principal: principal.0.clone(),
            verification: verdict.attestation(),
        };
        // What replay will check, checked before the record exists.
        self.validate_semantic_origin(None, prepared.delta(), &origin)?;
        let event = LedgerEvent::SemanticDeltaCommitted {
            base_revision: expected,
            revision,
            encoded_delta,
            origin,
        };
        let index = self.append_semantic(event, prepared)?;
        Ok(SemanticCommit {
            revision,
            affected,
            commit_index: Some(index),
        })
    }

    /// Append a semantic record a writer built, refusing before anything is
    /// written one whose encoding the ledger would refuse (a record too large
    /// to frame, say), so such a refusal does not fence the runtime.
    fn append_semantic(
        &mut self,
        event: LedgerEvent,
        prepared: PreparedDelta,
    ) -> Result<CommitIndex, RuntimeError> {
        ptr_ledger::check_encodable(&event)
            .map_err(|error| RuntimeError::Ledger(error.to_string()))?;
        self.append_prepared(event, Some(prepared))
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

    /// Record the raw text of a request as ingress: exactly one text value at
    /// [`request_raw_key`], recorded with a `Request` origin, with no
    /// verifier.
    ///
    /// Committing any other delta without the grant is not possible: the
    /// runtime has no public commit path for a semantic delta that no
    /// verifier judges.
    ///
    /// ```compile_fail
    /// fn write(runtime: &mut ptr_runtime::PtrRuntime, delta: ptr_semdb::SemanticDelta) {
    ///     let _ = runtime.apply_semantic_delta(runtime.revision(), delta);
    /// }
    /// ```
    ///
    /// ```
    /// fn write(runtime: &mut ptr_runtime::PtrRuntime, delta: ptr_semdb::SemanticDelta) {
    ///     let _ = runtime.ingest_text("request-1".into(), "text");
    /// }
    /// ```
    pub fn ingest_text(
        &mut self,
        request: RequestId,
        text: impl Into<String>,
    ) -> Result<Revision, RuntimeError> {
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert(request_raw_key(&request), text.into().into());
        let origin = SemanticOrigin::Request {
            request: request.0.clone(),
        };
        let committed = self.apply_ingress(origin, delta)?;
        self.emit(RuntimeEvent::RequestStarted(request));
        self.emit(RuntimeEvent::SnapshotOpened(committed.revision));
        Ok(committed.revision)
    }

    /// Semantic identity only. Lifecycle/authorization checks are independent;
    /// this must not be treated as a neural-checkpoint or execution permit.
    pub fn snapshot_is_current(&self, snapshot: &SemanticSnapshot) -> bool {
        !self.execution.is_fenced() && !self.semdb.is_stale(snapshot)
    }

    /// Promote a Pod's output that its verifier passed with no hard finding,
    /// as ingress at the level the verifier reported.
    pub(super) fn promote_pod_output(
        &mut self,
        request: &RequestId,
        pod: &PodId,
        output: &TypedPayload,
        level: VerificationLevel,
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
        let origin = SemanticOrigin::PodOutput {
            request: request.0.clone(),
            pod: pod.0.clone(),
            level,
        };
        self.apply_ingress(origin, delta)
            .map(|commit| commit.revision)
    }

    /// Check a semantic record against the current state and prepare its
    /// delta. `index` is where the record was committed, when it is being
    /// replayed, and `None` when it is about to be written.
    pub(super) fn prepare_semantic_event(
        &self,
        index: Option<CommitIndex>,
        event: &LedgerEvent,
    ) -> Result<Option<PreparedDelta>, RuntimeError> {
        let LedgerEvent::SemanticDeltaCommitted {
            base_revision,
            revision,
            encoded_delta,
            origin,
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
        self.validate_semantic_origin(index, &delta, origin)?;
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

    /// Whether a semantic record's origin may stand where it is: the rules a
    /// record is replayed under, which every writer also checks before it
    /// appends. `index` is where the record was committed, on replay, and
    /// `None` for a record about to be written.
    ///
    /// - **R1, `Legacy`.** Replayed only while no attributed record precedes
    ///   it, which the materialized [`ptr_state::ATTESTED_MARKER`] records;
    ///   after one, [`RuntimeError::LegacySemanticRecord`]. A writer never
    ///   records it.
    /// - **R2, `Request` and `PodOutput`.** Exactly the shape ingress writes.
    ///   A request: one text value at [`request_raw_key`] and nothing else.
    ///   A Pod's output: one payload at [`pod_output_key`] whose source is
    ///   the Pod, depending on exactly the request's raw text, and nothing
    ///   else.
    /// - **R3, `Host`.** No ingress key written, removed or derived; a
    ///   principal that is an identifier of at most
    ///   [`MAX_PROVENANCE_TEXT`](crate::merge::MAX_PROVENANCE_TEXT) bytes; and
    ///   an attestation whose weakest level meets its requirement, whose
    ///   verifiers are distinct valid names, and whose findings are sorted,
    ///   distinct, and each a recorded verifier's name, `/` and a valid code.
    /// - **`Merge`** is refused: this build does not merge.
    ///
    /// # Errors
    /// [`RuntimeError::LegacySemanticRecord`] for R1 on replay, and
    /// [`RuntimeError::InvalidSemanticOrigin`] with the rule's reason for
    /// every other failure.
    pub(super) fn validate_semantic_origin(
        &self,
        index: Option<CommitIndex>,
        delta: &SemanticDelta,
        origin: &SemanticOrigin,
    ) -> Result<(), RuntimeError> {
        let invalid = |reason| RuntimeError::InvalidSemanticOrigin { index, reason };
        match origin {
            SemanticOrigin::Legacy => {
                let Some(index) = index else {
                    return Err(invalid("a writer never records a legacy origin"));
                };
                if self.state.values.contains_key(ptr_state::ATTESTED_MARKER) {
                    return Err(RuntimeError::LegacySemanticRecord { index });
                }
                Ok(())
            }
            SemanticOrigin::Request { request } => {
                let key = request_raw_key(&RequestId(request.clone()));
                let shaped = delta.removals.is_empty()
                    && delta.dependencies.is_empty()
                    && delta.upserts.len() == 1
                    && matches!(delta.upserts.get(&key), Some(SemanticValue::Text(_)));
                if shaped {
                    Ok(())
                } else {
                    Err(invalid("a request record writes exactly its own raw text"))
                }
            }
            SemanticOrigin::PodOutput { request, pod, .. } => {
                let request = RequestId(request.clone());
                let key = pod_output_key(&request, &PodId(pod.clone()));
                let shaped = delta.removals.is_empty()
                    && delta.upserts.len() == 1
                    && matches!(
                        delta.upserts.get(&key),
                        Some(SemanticValue::Payload(payload)) if payload.source == *pod
                    )
                    && delta.dependencies.len() == 1
                    && delta.dependencies.get(&key)
                        == Some(&BTreeSet::from([request_raw_key(&request)]));
                if shaped {
                    Ok(())
                } else {
                    Err(invalid(
                        "a Pod output record writes exactly the Pod's output for its request",
                    ))
                }
            }
            SemanticOrigin::Host {
                principal,
                verification,
            } => {
                if written_ingress_key(delta).is_some() {
                    return Err(invalid(
                        "a host write writes, removes or derives an ingress key",
                    ));
                }
                if !merge::valid_provenance(principal) {
                    return Err(invalid(
                        "a host write's principal is not valid provenance text",
                    ));
                }
                check_attestation(verification).map_err(invalid)
            }
            SemanticOrigin::Merge(_) => Err(invalid("merge records need a build that merges")),
        }
    }
}

/// The first ingress key `delta` writes, removes or derives, if any. An
/// ingress key may still be a dependency input.
fn written_ingress_key(delta: &SemanticDelta) -> Option<&str> {
    delta
        .upserts
        .keys()
        .chain(delta.removals.iter())
        .chain(delta.dependencies.keys())
        .map(String::as_str)
        .find(|key| is_ingress_key(key))
}

/// The attestation rules replay checks beyond the codec's: the weakest level
/// meets the requirement, the verifier names are distinct valid names, and
/// the findings are sorted, distinct, and each a recorded verifier's name,
/// `/` and a valid finding code.
fn check_attestation(attestation: &Attestation) -> Result<(), &'static str> {
    let required = merge::requirement_of(attestation.required)
        .ok_or("an attestation requires a level no grant requires")?;
    if !required.accepts(attestation.level) {
        return Err("an attestation's weakest level does not meet its requirement");
    }
    let mut names = BTreeSet::new();
    for name in &attestation.verifiers {
        if !merge::valid_verifier_name(name) || !names.insert(name.as_str()) {
            return Err("an attestation names an invalid or repeated verifier");
        }
    }
    let mut previous: Option<&str> = None;
    for finding in &attestation.findings {
        if previous.is_some_and(|previous| finding.as_str() <= previous) {
            return Err("an attestation's findings are not sorted and distinct");
        }
        previous = Some(finding);
        let recorded = finding
            .split_once('/')
            .is_some_and(|(name, code)| names.contains(name) && merge::valid_finding_code(code));
        if !recorded {
            return Err("an attestation finding is not a recorded verifier's valid code");
        }
    }
    Ok(())
}
