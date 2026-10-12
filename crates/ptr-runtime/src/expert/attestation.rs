//! A minimal consumed-input attestation for expert admission (R013).
//!
//! An attestation is what a trusted training attestor signs about one training run: which run
//! it was, which proposal it served, and which artifact it emitted. Admission with an
//! attestation ([`PtrRuntime::admit_attested_expert_proposal`]) refuses, typed and with nothing
//! appended, an attestation that names another proposal or another artifact than the one being
//! committed ([`ExpertAdmissionError::AttestationProposalMismatch`],
//! [`ExpertAdmissionError::AttestationArtifactMismatch`]), and a grant that requires
//! attestation ([`ExpertAdmissionGrant::with_required_attestation`]) refuses an admission that
//! carries none ([`ExpertAdmissionError::AttestationRequired`]).
//!
//! # No new ledger machinery
//!
//! The attestor and the run are *inputs* of the admitted expert, recorded in its version 3
//! dependency block under the identities `attestor:{name}` (at the attestor's epoch, the
//! generation of that identity) and `run:{hex of the run id}` (generation 1). Everything else
//! follows from the dependency index that already exists:
//!
//! * a revoked attestor is a revoked input: an admission naming it is refused
//!   ([`ExpertAdmissionError::InputRevoked`]), and [`PtrRuntime::invalidate_input`] on it fences
//!   every expert admitted under it with cause `InputRevoked`, which is terminal (never
//!   readmitted, never a `Reactivate` seed, retired only), and so are their descendants;
//! * a new epoch of the attestor is a new generation of the identity, so a retrained expert
//!   attested at the new epoch is admitted while the old epoch stays revoked;
//! * the run id is single-use: any record whose block names a `run:` input that an earlier
//!   record already named is refused ([`ExpertAdmissionError::RunIdReused`]). The rule lives in
//!   the shared dependency check, so replay re-checks it and the plain admission path cannot
//!   bypass it.
//!
//! # Limits
//!
//! The attestation itself is not recorded and carries no signature: it is the host's assertion,
//! like the evaluator identity of a grant, and replay cannot tell whether a record was admitted
//! with one. What replay does re-check is everything that lives in the dependency block: that
//! the attestor and the run are among the inputs, that the attestor is live, that the run id is
//! unused. The artifact binding and the proposal binding are admission-time checks only. The
//! artifact digest an attestation binds is the digest the operation determines
//! ([`super::expert_evidence_artifact_digest`]), not the weights: nothing here reads weights.

use super::{
    expert_evidence_artifact_digest, Attested, ExpertAdmissionError, ExpertAdmissionGrant,
};
use crate::{PtrRuntime, RuntimeError};
use ptr_lineage::BoundGateReport;
use ptr_pods::{ExpertInputRef, ExpertInputs, ExpertProposal};
use ptr_types::{CommitIndex, Generation};

/// The prefix of an attestor's input identity: `attestor:{name}`.
pub const ATTESTOR_IDENTITY_PREFIX: &str = "attestor:";

/// The prefix of a training run's input identity: `run:{lowercase hex of the run id}`.
pub const RUN_IDENTITY_PREFIX: &str = "run:";

/// The generation every run identity is recorded at.
pub const RUN_GENERATION: Generation = Generation(1);

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// What one attestor asserts about one training run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertAttestation {
    attestor: String,
    epoch: Generation,
    attestor_digest: [u8; 32],
    run_id: [u8; 32],
    proposal_id: [u8; 32],
    artifact_digest: [u8; 32],
}

impl ExpertAttestation {
    /// An attestation by `attestor` at `epoch` (the attestor's key epoch; its identity digest is
    /// `attestor_digest`) that the run `run_id` served the proposal `proposal_id` and emitted
    /// the artifact `artifact_digest`.
    ///
    /// # Errors
    /// [`ExpertAdmissionError::InvalidRecord`] for a blank attestor name, a zero epoch or a
    /// zero digest or run id (the dependency block refuses those as inputs).
    pub fn new(
        attestor: impl Into<String>,
        epoch: Generation,
        attestor_digest: [u8; 32],
        run_id: [u8; 32],
        proposal_id: [u8; 32],
        artifact_digest: [u8; 32],
    ) -> Result<Self, ExpertAdmissionError> {
        let attestor = attestor.into();
        let invalid = |what: &str| ExpertAdmissionError::InvalidRecord(what.to_owned());
        if attestor.is_empty() {
            return Err(invalid("an attestation names no attestor"));
        }
        if epoch == Generation::default() {
            return Err(invalid("an attestation has the attestor epoch zero"));
        }
        if attestor_digest == [0; 32] || run_id == [0; 32] {
            return Err(invalid("an attestation has an all-zero attestor or run id"));
        }
        Ok(Self {
            attestor,
            epoch,
            attestor_digest,
            run_id,
            proposal_id,
            artifact_digest,
        })
    }

    /// The input identity of the attestor: `attestor:{name}`.
    pub fn attestor_identity(&self) -> String {
        format!("{ATTESTOR_IDENTITY_PREFIX}{}", self.attestor)
    }

    /// The attestor's epoch, the generation of its input identity.
    pub fn epoch(&self) -> Generation {
        self.epoch
    }

    /// The digest the attestor's input identity carries.
    pub fn attestor_digest(&self) -> &[u8; 32] {
        &self.attestor_digest
    }

    /// The input identity of the run: `run:{hex}`.
    pub fn run_identity(&self) -> String {
        format!("{RUN_IDENTITY_PREFIX}{}", hex(&self.run_id))
    }

    /// The run id.
    pub fn run_id(&self) -> &[u8; 32] {
        &self.run_id
    }

    /// The proposal the run served.
    pub fn proposal_id(&self) -> &[u8; 32] {
        &self.proposal_id
    }

    /// The artifact the run emitted.
    pub fn artifact_digest(&self) -> &[u8; 32] {
        &self.artifact_digest
    }

    /// `inputs` plus the attestor and the run as training inputs: the block an attested
    /// admission commits, and the block the gate report's artifact digest must be computed over
    /// ([`super::expert_evidence_artifact_digest_with_inputs`]).
    ///
    /// # Errors
    /// [`ExpertAdmissionError::InvalidRecord`] when the block already names an `attestor:` or
    /// `run:` identity (those come from the attestation alone) or cannot hold two more inputs.
    pub fn attested_inputs(
        &self,
        inputs: &ExpertInputs,
    ) -> Result<ExpertInputs, ExpertAdmissionError> {
        let invalid = |what: String| ExpertAdmissionError::InvalidRecord(what);
        let reserved = |identity: &str| {
            identity.starts_with(ATTESTOR_IDENTITY_PREFIX)
                || identity.starts_with(RUN_IDENTITY_PREFIX)
        };
        if inputs
            .training()
            .iter()
            .chain(inputs.evaluation())
            .any(|input| reserved(input.identity()))
        {
            return Err(invalid(
                "a dependency block names an attestor or a run of its own".to_owned(),
            ));
        }
        let attestor =
            ExpertInputRef::new(self.attestor_identity(), self.epoch, self.attestor_digest)
                .map_err(|error| invalid(error.to_string()))?;
        let run = ExpertInputRef::new(self.run_identity(), RUN_GENERATION, self.run_id)
            .map_err(|error| invalid(error.to_string()))?;
        let mut training = inputs.training().to_vec();
        training.push(attestor);
        training.push(run);
        ExpertInputs::new(training, inputs.evaluation().to_vec())
            .map_err(|error| invalid(error.to_string()))
    }
}

impl PtrRuntime {
    /// [`PtrRuntime::admit_expert_proposal_with_inputs`] with an attestation: the attestor and
    /// the run join the dependency block ([`ExpertAttestation::attested_inputs`]), and the
    /// attestation must name this proposal and this proposal's artifact. The gate `report` must
    /// be bound to the attested block.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check: the two bindings above, then every
    /// check of the plain path (a revoked attestor and a reused run id among them); nothing is
    /// appended on error.
    pub fn admit_attested_expert_proposal(
        &mut self,
        proposal: &ExpertProposal,
        report: &BoundGateReport,
        grant: &ExpertAdmissionGrant,
        inputs: ExpertInputs,
        attestation: &ExpertAttestation,
    ) -> Result<CommitIndex, RuntimeError> {
        if attestation.proposal_id != *proposal.proposal_id().as_bytes() {
            return Err(RuntimeError::Expert(
                ExpertAdmissionError::AttestationProposalMismatch,
            ));
        }
        if attestation.artifact_digest != expert_evidence_artifact_digest(proposal.op()) {
            return Err(RuntimeError::Expert(
                ExpertAdmissionError::AttestationArtifactMismatch,
            ));
        }
        let block = attestation
            .attested_inputs(&inputs)
            .map_err(RuntimeError::Expert)?;
        self.admit_expert_record(proposal, report, grant, Some(block), Attested::Yes)
    }
}
