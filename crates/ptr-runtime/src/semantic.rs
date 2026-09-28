//! Durable semantic publication uses the same ordered log as lifecycle changes.
//! No acknowledgments, observations or model continuation precede a successful
//! append. This reconstructs semantic state, not neural or execution checkpoints.
use super::{PtrRuntime, RuntimeError};
use crate::merge::{self, ChangeOrigin, SemanticChange};
use ptr_events::RuntimeEvent;
use ptr_ledger::{Attestation, LedgerEvent, MergeAuthorityRecord, SemanticOrigin};
use ptr_protocol::TypedPayload;
use ptr_semdb::{
    is_ingress_key, PreparedDelta, SemanticDelta, SemanticError, SemanticPayload, SemanticSnapshot,
    SemanticValue,
};

use ptr_types::{CommitIndex, PodId, PrincipalId, RequestId, Revision, VerificationLevel};
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
        let encoded_delta = delta.encode().map_err(RuntimeError::Semantic)?;
        self.validate_semantic_origin(None, expected, &encoded_delta, &delta, &origin)?;
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
    /// 7. a delta that does not encode or prepare, and one that would evict
    ///    an ingress key a record written before origins existed derived from
    ///    a key it changes ([`RuntimeError::ReservedSemanticNamespace`]); no
    ///    write since can remove such a dependency, so the key it hangs on can
    ///    no longer be written by a host;
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
    /// level the grant accepts with no hard finding; a verifier whose report
    /// fails closed is emitted as not passed, and none after it judges.
    ///
    /// It checks no lifecycle generation and certifies nothing, and its
    /// record names the principal, not a branch: an agent branch is merged
    /// through [`Self::merge_branch`], which certifies it against this
    /// runtime's state in the same call. A plan's delta written here is a
    /// host write like any other.
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
        if let Some(key) = evicted_ingress_key(&prepared) {
            return Err(RuntimeError::ReservedSemanticNamespace {
                key: key.to_owned(),
            });
        }
        let judgement = {
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
            grant.judge(&change)
        };
        self.emit_verifier_results(&judgement.results);
        let verdict = judgement.verdict?;
        if !verdict.admitted() {
            return Err(RuntimeError::SemanticVerificationRejected(
                verdict.refusal(),
            ));
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
        let origin = SemanticOrigin::Host {
            principal: principal.0.clone(),
            verification: verdict.attestation(),
        };
        // What replay will check, checked before the record exists.
        self.validate_semantic_origin(None, expected, &encoded_delta, prepared.delta(), &origin)?;
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
        self.validate_semantic_origin(index, *base_revision, encoded_delta, &delta, origin)?;
        // A record replayed from memory never passed through the codec; one
        // the ledger could not frame would leave a runtime that cannot export
        // its own history.
        ptr_ledger::check_encodable(event).map_err(|_| RuntimeError::InvalidSemanticOrigin {
            index,
            reason: "a record the ledger cannot frame",
        })?;
        let prepared = self
            .semdb
            .prepare_delta(delta)
            .map_err(RuntimeError::Semantic)?;
        if matches!(
            origin,
            SemanticOrigin::Host { .. } | SemanticOrigin::Merge(_)
        ) && evicted_ingress_key(&prepared).is_some()
        {
            return Err(RuntimeError::InvalidSemanticOrigin {
                index,
                reason: "a host write or merge evicts an ingress key",
            });
        }
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
    /// `None` for a record about to be written; `base` and `encoded` are the
    /// record's base revision and its delta's encoding, from which `delta`
    /// was decoded.
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
    /// - **R3, `Host` and `Merge`.** No ingress key written, removed, derived
    ///   or evicted (a key only ingress writes that a record without an
    ///   origin derived from one the write changes, checked once the delta is
    ///   prepared); provenance text that is an identifier of at most
    ///   [`MAX_PROVENANCE_TEXT`](crate::merge::MAX_PROVENANCE_TEXT) bytes (a
    ///   host write's principal; a merge's branch, author, and reviewer or
    ///   policy version); a merge's triage score a finite `f32` in `[0, 1]`;
    ///   and an attestation whose weakest level meets its requirement, whose
    ///   verifiers are distinct valid names, and whose findings are sorted,
    ///   distinct, and each a recorded verifier's name, `/` and a valid code.
    /// - **R4, `Merge`.** No dependency entry, since certification never
    ///   rewires one; no rebased key removed or reserved to ingress; a plan
    ///   digest that is `ptr_branch::merge_plan_digest` of the branch, `base`,
    ///   `encoded`, the dependency digest and the rebased keys, so the record
    ///   carries exactly the delta of the plan it names; and a branch not
    ///   merged before ([`ptr_state::merged_branch_key`] absent).
    ///
    /// # Errors
    /// [`RuntimeError::LegacySemanticRecord`] for R1 on replay, and
    /// [`RuntimeError::InvalidSemanticOrigin`] with the rule's reason for
    /// every other failure. Replay also refuses, with the same error, a
    /// record the ledger could not frame, which only a history held in memory
    /// can carry.
    pub(super) fn validate_semantic_origin(
        &self,
        index: Option<CommitIndex>,
        base: Revision,
        encoded: &[u8],
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
            SemanticOrigin::Merge(merge) => {
                if written_ingress_key(delta).is_some() {
                    return Err(invalid("a merge writes, removes or derives an ingress key"));
                }
                if !merge::valid_provenance(&merge.branch)
                    || !merge::valid_provenance(&merge.author)
                {
                    return Err(invalid(
                        "a merge's branch or author is not valid provenance text",
                    ));
                }
                match &merge.authority {
                    MergeAuthorityRecord::Triage {
                        policy_version,
                        score_bits,
                    } => {
                        if !merge::valid_provenance(policy_version) {
                            return Err(invalid(
                                "a merge's policy version is not valid provenance text",
                            ));
                        }
                        // `contains` is false for NaN and both infinities.
                        if !(0.0..=1.0).contains(&f32::from_bits(*score_bits)) {
                            return Err(invalid("a merge's triage score is not a probability"));
                        }
                    }
                    MergeAuthorityRecord::Reviewed { reviewer } => {
                        if !merge::valid_provenance(reviewer) {
                            return Err(invalid("a merge's reviewer is not valid provenance text"));
                        }
                    }
                }
                check_attestation(&merge.verification).map_err(invalid)?;
                if !delta.dependencies.is_empty() {
                    return Err(invalid("a merge carries a dependency entry"));
                }
                if merge
                    .rebased
                    .iter()
                    .any(|key| delta.removals.contains(key) || is_ingress_key(key))
                {
                    return Err(invalid(
                        "a merge's rebased key is removed or reserved to ingress",
                    ));
                }
                let plan = ptr_branch::merge_plan_digest(
                    &merge.branch,
                    base,
                    encoded,
                    &merge.dependencies,
                    &merge.rebased,
                );
                if plan != merge.plan {
                    return Err(invalid(
                        "a merge's plan digest is not the digest of its branch, base, delta, \
                         dependencies and rebased keys",
                    ));
                }
                if self
                    .state
                    .values
                    .contains_key(&ptr_state::merged_branch_key(&merge.branch))
                {
                    return Err(invalid("a merge of a branch that is already merged"));
                }
                Ok(())
            }
        }
    }
}

/// The first ingress key `delta` writes, removes or derives, if any. An
/// ingress key may still be a dependency input.
pub(crate) fn written_ingress_key(delta: &SemanticDelta) -> Option<&str> {
    delta
        .upserts
        .keys()
        .chain(delta.removals.iter())
        .chain(delta.dependencies.keys())
        .map(String::as_str)
        .find(|key| is_ingress_key(key))
}

/// The key only ingress writes that a prepared change evicts, if any. A host
/// write or merge never writes or derives one, so such a key is derived from
/// a key the change writes, which only a record written before origins
/// existed can have set up.
pub(crate) fn evicted_ingress_key(prepared: &PreparedDelta) -> Option<&str> {
    prepared
        .affected()
        .iter()
        .map(String::as_str)
        .find(|key| is_ingress_key(key))
}

/// The attestation rules replay checks: the requirement is one a grant
/// records, the weakest level meets it, there are 1 to
/// [`MAX_SEMANTIC_VERIFIERS`](merge::MAX_SEMANTIC_VERIFIERS) verifiers with
/// distinct valid names, and at most
/// [`MAX_ATTESTED_FINDINGS`](merge::MAX_ATTESTED_FINDINGS) findings, sorted,
/// distinct, and each a recorded verifier's name, `/` and a valid finding
/// code. The counts are the codec's too, but a history replayed from memory
/// never passed through the codec.
fn check_attestation(attestation: &Attestation) -> Result<(), &'static str> {
    let required = merge::requirement_of(attestation.required)
        .ok_or("an attestation requires a level no grant requires")?;
    if !required.accepts(attestation.level) {
        return Err("an attestation's weakest level does not meet its requirement");
    }
    if attestation.verifiers.is_empty()
        || attestation.verifiers.len() > merge::MAX_SEMANTIC_VERIFIERS
    {
        return Err("an attestation names no verifier, or more than a grant installs");
    }
    if attestation.findings.len() > merge::MAX_ATTESTED_FINDINGS {
        return Err("an attestation carries more soft findings than a record holds");
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
