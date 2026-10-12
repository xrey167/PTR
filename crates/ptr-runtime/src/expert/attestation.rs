//! A signed consumed-input attestation for expert admission (R013).
//!
//! An attestation is a record signed by a registered attestor key about one training run: which
//! run it was, which proposal it served, which artifact it emitted, on which base model and as
//! which expert generation(s). The signing side builds an [`ExpertAttestationClaim`] and
//! [`ExpertAttestationClaim::sign`]s it; the runtime never sees a signing key.
//!
//! # Trust
//!
//! Which keys are trusted is host configuration, not something the caller of an admission
//! supplies: an [`ExpertAttestorRegistry`] maps `(attestor name, epoch)` to an Ed25519 verifying
//! key, and the trusted set is fixed at construction. The host builds the registry from its own
//! configuration and hands it to the constructor ([`PtrRuntime::new_with_attestors`],
//! [`PtrRuntime::open_durable_with_attestors`], [`PtrRuntime::replay_with_attestors`],
//! [`PtrRuntime::restore_compacted_with_attestors`]); a live runtime has no registration call. A
//! holder of grants cannot make a key trusted: a grant (even a gate grant a caller assembles
//! from `ExpertAdmissionGrant::binding()`, which the types allow) authorises an operation on
//! experts, and no API takes a grant to change the registry (`PtrRuntime::expert_attestors`
//! returns a shared reference). Verification is strict (`verify_strict`, which
//! also refuses a weak key and a non-canonical signature). An attestor or an epoch the registry
//! does not hold is refused as [`ExpertAdmissionError::AttestorUnknown`]; a history replayed
//! against an empty registry therefore refuses every attested record (it fails closed).
//!
//! # What the signature covers
//!
//! The signed payload is domain separated and covers the attestor name, its epoch, the key id
//! (SHA-256 of the verifying key under its own domain), the run id, the proposal id, the artifact
//! digest ([`super::expert_evidence_artifact_digest`], the digest the operation determines, not
//! the weights), the base model (id, digest, generation) and the ordered result experts (pod
//! and generation). Admission refuses, typed and with nothing appended, a claim that names
//! another proposal ([`ExpertAdmissionError::AttestationProposalMismatch`]), artifact
//! ([`ExpertAdmissionError::AttestationArtifactMismatch`]), base model
//! ([`ExpertAdmissionError::AttestationBaseModelMismatch`]) or result set
//! ([`ExpertAdmissionError::AttestationResultMismatch`]) than the operation being committed.
//!
//! # What is recorded
//!
//! The attestation lives in the version 3 dependency block of the record, as training inputs:
//!
//! * `attestor:{name}` at the attestor's epoch (the generation of the identity), digest = key id;
//! * `run:{hex of the run id}` at generation 1, digest = the run id;
//! * `attestation:{hex of the 64 signature bytes}` at generation 1, digest = SHA-256 of the signed
//!   payload;
//! * `policy:attestation-required` at generation 1 when the admitting grant required
//!   attestation ([`ExpertAdmissionGrant::with_required_attestation`]).
//!
//! Replay rebuilds the payload from the record itself (proposal, operation, base model, results,
//! inputs), checks the signature against the registry's key for that attestor and epoch and
//! checks the recorded key id and payload digest, so a record whose artifact, proposal, base
//! model or generation was changed after signing does not replay.
//!
//! The requirement is durable: the marker input registers `policy:attestation-required` in the
//! dependency index, which is replayed and compacted with the rest of it. From the first record
//! that carries it, every later artifact-producing record, admitted or replayed, must carry a
//! complete attestation ([`ExpertAdmissionError::AttestationRequired`]), whatever grant a
//! restarted host builds. Operations that produce no artifact (Freeze, Retire, Prune, unfreeze)
//! are not attested and not affected.
//!
//! # Revocation and single use
//!
//! The attestor and the run are inputs, so the existing dependency machinery applies:
//! revoking an attestor epoch ([`PtrRuntime::invalidate_input`] on `attestor:{name}`) refuses
//! new admissions under it ([`ExpertAdmissionError::InputRevoked`]) and fences every expert
//! admitted under it, descendants included, with the terminal cause `InputRevoked`; a new epoch
//! is a new generation of the identity. A run id is single-use: any record naming `run:{hex}` in
//! any role at any generation is refused when an earlier record named that run
//! ([`ExpertAdmissionError::RunIdReused`]), at admission and in replay. The identities
//! `attestor:`, `run:` and `attestation:` and the policy marker are reserved: only an attested
//! admission sets them ([`ExpertAdmissionError::ReservedInputIdentity`]).
//!
//! # Limits
//!
//! The registry is host memory, not history: it is not ledgered, and a registry that is
//! rebuilt after a restart is the host's assertion again. The runtime checks that the registered
//! key is the one the record names, not that the host registered the right key. What the
//! attestor actually read is not checked here: the signature binds claims, not weights.

use super::{
    expert_evidence_artifact_digest, Attested, ExpertAdmissionError, ExpertAdmissionGrant,
};
use crate::{PtrRuntime, RuntimeError};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use ptr_ledger::integrity::sha256;
use ptr_lineage::BoundGateReport;
use ptr_pods::{BaseModelRef, ExpertCommitRecord, ExpertInputRef, ExpertInputs, ExpertProposal};
use ptr_types::{CommitIndex, Generation, PodId};
use std::collections::BTreeMap;

/// The prefix of an attestor's input identity: `attestor:{name}`.
pub const ATTESTOR_IDENTITY_PREFIX: &str = "attestor:";

/// The prefix of a training run's input identity: `run:{lowercase hex of the run id}`.
pub const RUN_IDENTITY_PREFIX: &str = "run:";

/// The prefix of the recorded signature's input identity: `attestation:{hex of the signature}`.
pub const ATTESTATION_IDENTITY_PREFIX: &str = "attestation:";

/// The identity of the marker input a record admitted under a grant that requires attestation
/// carries.
pub const ATTESTATION_POLICY_IDENTITY: &str = "policy:attestation-required";

/// The generation every run, signature and marker identity is recorded at.
pub const RUN_GENERATION: Generation = Generation(1);

/// The longest attestor name in bytes.
pub const MAX_ATTESTOR_NAME_BYTES: usize = 64;

const KEY_ID_DOMAIN: &[u8] = b"ptr.runtime.expert-attestor-key.v1";
const PAYLOAD_DOMAIN: &str = "ptr.runtime.expert-attestation.v1";
const POLICY_DIGEST_DOMAIN: &[u8] = b"ptr.runtime.expert-attestation-required.v1";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex<const N: usize>(text: &str) -> Option<[u8; N]> {
    if text.len() != N * 2 || !text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return None;
    }
    let mut out = [0u8; N];
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().chunks(2)) {
        let digits = std::str::from_utf8(pair).ok()?;
        *slot = u8::from_str_radix(digits, 16).ok()?;
    }
    Some(out)
}

fn put_text(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn policy_digest() -> [u8; 32] {
    sha256(POLICY_DIGEST_DOMAIN)
}

/// The id of an attestor key: SHA-256 over the domain `ptr.runtime.expert-attestor-key.v1` and
/// the 32 bytes of the verifying key.
pub fn expert_attestor_key_id(key: &VerifyingKey) -> [u8; 32] {
    let mut bytes = KEY_ID_DOMAIN.to_vec();
    bytes.extend_from_slice(key.as_bytes());
    sha256(&bytes)
}

/// Whether `identity` is one of the input identities only an attested admission may set.
pub(super) fn is_reserved_identity(identity: &str) -> bool {
    identity.starts_with(ATTESTOR_IDENTITY_PREFIX)
        || identity.starts_with(RUN_IDENTITY_PREFIX)
        || identity.starts_with(ATTESTATION_IDENTITY_PREFIX)
        || identity == ATTESTATION_POLICY_IDENTITY
}

fn check_attestor_name(name: &str) -> Result<(), ExpertAdmissionError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_ATTESTOR_NAME_BYTES
        && name
            .bytes()
            .all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(ExpertAdmissionError::InvalidAttestorName {
            name: name.chars().take(MAX_ATTESTOR_NAME_BYTES).collect(),
        })
    }
}

/// The attestor keys the host trusts, by `(name, epoch)`.
///
/// Host configuration: nothing in a proposal, a report or an attestation can add to it. A name is
/// 1 to [`MAX_ATTESTOR_NAME_BYTES`] bytes of `a-z`, `0-9`, `.`, `_` and `-`.
#[derive(Clone, Debug, Default)]
pub struct ExpertAttestorRegistry {
    keys: BTreeMap<(String, Generation), VerifyingKey>,
}

impl ExpertAttestorRegistry {
    /// An empty registry: nothing is trusted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Trust `key` for `attestor` at `epoch`.
    ///
    /// # Errors
    /// [`ExpertAdmissionError::InvalidAttestorName`] for a name outside the rules above,
    /// [`ExpertAdmissionError::InvalidRecord`] for epoch zero or for a `(name, epoch)` already
    /// registered with another key (an epoch's key never changes; rotate by a new epoch).
    pub fn register(
        &mut self,
        attestor: &str,
        epoch: Generation,
        key: VerifyingKey,
    ) -> Result<(), ExpertAdmissionError> {
        check_attestor_name(attestor)?;
        if epoch == Generation::default() {
            return Err(ExpertAdmissionError::InvalidRecord(
                "an attestor is registered at epoch zero".to_owned(),
            ));
        }
        match self.keys.get(&(attestor.to_owned(), epoch)) {
            Some(known) if *known != key => Err(ExpertAdmissionError::InvalidRecord(
                "the attestor epoch is already registered with another key".to_owned(),
            )),
            Some(_) => Ok(()),
            None => {
                self.keys.insert((attestor.to_owned(), epoch), key);
                Ok(())
            }
        }
    }

    /// The key trusted for `attestor` at `epoch`.
    pub fn key(&self, attestor: &str, epoch: Generation) -> Option<&VerifyingKey> {
        self.keys.get(&(attestor.to_owned(), epoch))
    }
}

/// What an attestor asserts about one training run, before it is signed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertAttestationClaim {
    attestor: String,
    epoch: Generation,
    run_id: [u8; 32],
    proposal_id: [u8; 32],
    artifact_digest: [u8; 32],
    base_model: BaseModelRef,
    results: Vec<(PodId, Generation)>,
}

impl ExpertAttestationClaim {
    /// A claim by `attestor` at `epoch` that the run `run_id` served `proposal_id`, emitted
    /// `artifact_digest` on `base_model` and produced the ordered `results`.
    ///
    /// # Errors
    /// [`ExpertAdmissionError::InvalidAttestorName`], and
    /// [`ExpertAdmissionError::InvalidRecord`] for epoch zero, an all-zero run id, or no result.
    pub fn new(
        attestor: impl Into<String>,
        epoch: Generation,
        run_id: [u8; 32],
        proposal_id: [u8; 32],
        artifact_digest: [u8; 32],
        base_model: BaseModelRef,
        results: Vec<(PodId, Generation)>,
    ) -> Result<Self, ExpertAdmissionError> {
        let attestor = attestor.into();
        check_attestor_name(&attestor)?;
        let invalid = |what: &str| ExpertAdmissionError::InvalidRecord(what.to_owned());
        if epoch == Generation::default() {
            return Err(invalid("an attestation has the attestor epoch zero"));
        }
        if run_id == [0; 32] {
            return Err(invalid("an attestation has an all-zero run id"));
        }
        if results.is_empty() {
            return Err(invalid("an attestation names no result expert"));
        }
        Ok(Self {
            attestor,
            epoch,
            run_id,
            proposal_id,
            artifact_digest,
            base_model,
            results,
        })
    }

    /// The claim an honest attestor makes about `proposal`: its id, its artifact digest, its base
    /// model and its results.
    ///
    /// # Errors
    /// As [`ExpertAttestationClaim::new`]; an operation that produces no result expert has no
    /// claim.
    pub fn for_proposal(
        attestor: impl Into<String>,
        epoch: Generation,
        run_id: [u8; 32],
        proposal: &ExpertProposal,
    ) -> Result<Self, ExpertAdmissionError> {
        let results = proposal.op().results();
        let Some(first) = results.first() else {
            return Err(ExpertAdmissionError::InvalidRecord(
                "an operation without a result expert has nothing to attest".to_owned(),
            ));
        };
        Self::new(
            attestor,
            epoch,
            run_id,
            *proposal.proposal_id().as_bytes(),
            expert_evidence_artifact_digest(proposal.op()),
            first.base_model().clone(),
            results
                .iter()
                .map(|result| (result.pod().clone(), result.generation()))
                .collect(),
        )
    }

    /// Sign the claim with the attestor's key.
    pub fn sign(self, key: &SigningKey) -> ExpertAttestation {
        let verifying = key.verifying_key();
        let key_id = expert_attestor_key_id(&verifying);
        let signature = key.sign(&self.payload(&key_id)).to_bytes();
        ExpertAttestation {
            claim: self,
            key_id,
            signature,
        }
    }

    pub fn attestor(&self) -> &str {
        &self.attestor
    }

    pub fn epoch(&self) -> Generation {
        self.epoch
    }

    pub fn run_id(&self) -> &[u8; 32] {
        &self.run_id
    }

    pub fn proposal_id(&self) -> &[u8; 32] {
        &self.proposal_id
    }

    pub fn artifact_digest(&self) -> &[u8; 32] {
        &self.artifact_digest
    }

    pub fn base_model(&self) -> &BaseModelRef {
        &self.base_model
    }

    pub fn results(&self) -> &[(PodId, Generation)] {
        &self.results
    }

    /// The bytes that are signed: see the module documentation for what they cover.
    fn payload(&self, key_id: &[u8; 32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_text(&mut bytes, PAYLOAD_DOMAIN);
        put_text(&mut bytes, &self.attestor);
        bytes.extend_from_slice(&self.epoch.0.to_le_bytes());
        bytes.extend_from_slice(key_id);
        bytes.extend_from_slice(&self.run_id);
        bytes.extend_from_slice(&self.proposal_id);
        bytes.extend_from_slice(&self.artifact_digest);
        put_text(&mut bytes, &self.base_model.id().to_string());
        bytes.extend_from_slice(self.base_model.digest().as_bytes());
        bytes.extend_from_slice(&self.base_model.generation().to_le_bytes());
        bytes.extend_from_slice(&(self.results.len() as u64).to_le_bytes());
        for (pod, generation) in &self.results {
            put_text(&mut bytes, &pod.0);
            bytes.extend_from_slice(&generation.0.to_le_bytes());
        }
        bytes
    }
}

/// A claim and the signature of one attestor key over it. Holding one proves nothing until the
/// runtime verifies it against its registry at admission and in replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertAttestation {
    claim: ExpertAttestationClaim,
    key_id: [u8; 32],
    signature: [u8; 64],
}

impl ExpertAttestation {
    /// An attestation from its parts, as received from an attestor. The signature is not checked
    /// here.
    pub fn from_parts(
        claim: ExpertAttestationClaim,
        key_id: [u8; 32],
        signature: [u8; 64],
    ) -> Self {
        Self {
            claim,
            key_id,
            signature,
        }
    }

    pub fn claim(&self) -> &ExpertAttestationClaim {
        &self.claim
    }

    /// The id of the key that signed ([`expert_attestor_key_id`]).
    pub fn key_id(&self) -> &[u8; 32] {
        &self.key_id
    }

    pub fn signature(&self) -> &[u8; 64] {
        &self.signature
    }

    /// The input identity of the attestor: `attestor:{name}`.
    pub fn attestor_identity(&self) -> String {
        format!("{ATTESTOR_IDENTITY_PREFIX}{}", self.claim.attestor)
    }

    /// The input identity of the run: `run:{hex}`.
    pub fn run_identity(&self) -> String {
        format!("{RUN_IDENTITY_PREFIX}{}", hex(&self.claim.run_id))
    }

    /// The input identity that records the signature: `attestation:{hex}`.
    pub fn signature_identity(&self) -> String {
        format!("{ATTESTATION_IDENTITY_PREFIX}{}", hex(&self.signature))
    }

    /// `inputs` plus the attestor, the run, the signature and, when `grant` requires
    /// attestation, the policy marker, as training inputs: the block an attested admission
    /// commits, and the block the gate report's artifact digest must be computed over
    /// ([`super::expert_evidence_artifact_digest_with_inputs`]).
    ///
    /// # Errors
    /// [`ExpertAdmissionError::ReservedInputIdentity`] when the block already names a reserved
    /// identity (those come from the attestation alone), and
    /// [`ExpertAdmissionError::InvalidRecord`] when the block cannot hold the additions.
    pub fn attested_inputs(
        &self,
        inputs: &ExpertInputs,
        grant: &ExpertAdmissionGrant,
    ) -> Result<ExpertInputs, ExpertAdmissionError> {
        let invalid = |what: String| ExpertAdmissionError::InvalidRecord(what);
        if let Some(reserved) = inputs
            .training()
            .iter()
            .chain(inputs.evaluation())
            .find(|input| is_reserved_identity(input.identity()))
        {
            return Err(ExpertAdmissionError::ReservedInputIdentity {
                identity: reserved.identity().to_owned(),
            });
        }
        let payload_digest = sha256(&self.claim.payload(&self.key_id));
        let make = |identity: String, generation: Generation, digest: [u8; 32]| {
            ExpertInputRef::new(identity, generation, digest)
                .map_err(|error| invalid(error.to_string()))
        };
        let mut training = inputs.training().to_vec();
        training.push(make(
            self.attestor_identity(),
            self.claim.epoch,
            self.key_id,
        )?);
        training.push(make(
            self.run_identity(),
            RUN_GENERATION,
            self.claim.run_id,
        )?);
        training.push(make(
            self.signature_identity(),
            RUN_GENERATION,
            payload_digest,
        )?);
        if grant.requires_attestation() {
            training.push(make(
                ATTESTATION_POLICY_IDENTITY.to_owned(),
                RUN_GENERATION,
                policy_digest(),
            )?);
        }
        ExpertInputs::new(training, inputs.evaluation().to_vec())
            .map_err(|error| invalid(error.to_string()))
    }
}

/// Strict verification of `signature` over `claim` by `key`, with the key checked against
/// `key_id`.
fn verify(
    claim: &ExpertAttestationClaim,
    key: &VerifyingKey,
    key_id: &[u8; 32],
    signature: &[u8; 64],
) -> Result<(), ExpertAdmissionError> {
    if expert_attestor_key_id(key) != *key_id {
        return Err(ExpertAdmissionError::AttestorKeyMismatch {
            attestor: claim.attestor.clone(),
            epoch: claim.epoch,
        });
    }
    key.verify_strict(&claim.payload(key_id), &Signature::from_bytes(signature))
        .map_err(|_| ExpertAdmissionError::AttestationSignatureInvalid)
}

fn unknown(claim: &ExpertAttestationClaim) -> ExpertAdmissionError {
    ExpertAdmissionError::AttestorUnknown {
        attestor: claim.attestor.clone(),
        epoch: claim.epoch,
    }
}

/// What the dependency block of a record says about attestation, found by identity.
#[derive(Default)]
struct Found<'a> {
    attestors: Vec<&'a ExpertInputRef>,
    runs: Vec<&'a ExpertInputRef>,
    signatures: Vec<&'a ExpertInputRef>,
    markers: Vec<&'a ExpertInputRef>,
}

/// The attestation checks of a record, shared by admission and replay: they run in
/// `plan_dependencies`, after the generic per-input checks.
///
/// `policy_recorded` is whether an earlier record registered the policy marker. An
/// artifact-producing record must carry a complete, verifying attestation when it carries the
/// marker itself or when the policy is recorded; a record that carries any reserved identity
/// carries all of them or is refused.
pub(super) fn check_record(
    registry: &ExpertAttestorRegistry,
    record: &ExpertCommitRecord,
    policy_recorded: bool,
) -> Result<(), ExpertAdmissionError> {
    let op = record.proposal().op();
    let results = op.results();
    let Some(first) = results.first() else {
        return Ok(());
    };
    let mut found = Found::default();
    if let Some(block) = record.inputs() {
        if let Some(reserved) = block
            .evaluation()
            .iter()
            .find(|input| is_reserved_identity(input.identity()))
        {
            return Err(ExpertAdmissionError::ReservedInputIdentity {
                identity: reserved.identity().to_owned(),
            });
        }
        for input in block.training() {
            let identity = input.identity();
            if identity.starts_with(ATTESTOR_IDENTITY_PREFIX) {
                found.attestors.push(input);
            } else if identity.starts_with(RUN_IDENTITY_PREFIX) {
                found.runs.push(input);
            } else if identity.starts_with(ATTESTATION_IDENTITY_PREFIX) {
                found.signatures.push(input);
            } else if identity == ATTESTATION_POLICY_IDENTITY {
                found.markers.push(input);
            }
        }
    }
    if found.attestors.is_empty() && found.signatures.is_empty() {
        if let Some(run) = found.runs.first() {
            return Err(ExpertAdmissionError::ReservedInputIdentity {
                identity: run.identity().to_owned(),
            });
        }
        if !found.markers.is_empty() || policy_recorded {
            return Err(ExpertAdmissionError::AttestationRequired);
        }
        return Ok(());
    }
    let malformed = |what: &str| ExpertAdmissionError::AttestationMalformed(what.to_owned());
    let ([attestor], [run], [signature], markers) = (
        found.attestors.as_slice(),
        found.runs.as_slice(),
        found.signatures.as_slice(),
        found.markers.as_slice(),
    ) else {
        return Err(malformed(
            "an attested record names exactly one attestor, one run and one signature",
        ));
    };
    let marker_ok = match markers {
        [] => true,
        [marker] => marker.generation() == RUN_GENERATION && *marker.digest() == policy_digest(),
        _ => false,
    };
    if !marker_ok {
        return Err(malformed("the attestation policy marker is malformed"));
    }
    let name = attestor
        .identity()
        .strip_prefix(ATTESTOR_IDENTITY_PREFIX)
        .unwrap_or_default();
    check_attestor_name(name)?;
    let run_id: [u8; 32] = run
        .identity()
        .strip_prefix(RUN_IDENTITY_PREFIX)
        .and_then(unhex)
        .filter(|id| id == run.digest() && run.generation() == RUN_GENERATION)
        .ok_or_else(|| malformed("the run input does not name its own run id at generation 1"))?;
    let signature_bytes: [u8; 64] = signature
        .identity()
        .strip_prefix(ATTESTATION_IDENTITY_PREFIX)
        .and_then(unhex)
        .filter(|_| signature.generation() == RUN_GENERATION)
        .ok_or_else(|| malformed("the signature input is not 64 bytes of hex at generation 1"))?;
    let claim = ExpertAttestationClaim::new(
        name,
        attestor.generation(),
        run_id,
        *record.proposal().proposal_id().as_bytes(),
        expert_evidence_artifact_digest(op),
        first.base_model().clone(),
        results
            .iter()
            .map(|result| (result.pod().clone(), result.generation()))
            .collect(),
    )
    .map_err(|error| malformed(&format!("{error:?}")))?;
    let key = registry
        .key(name, attestor.generation())
        .ok_or_else(|| unknown(&claim))?;
    verify(&claim, key, attestor.digest(), &signature_bytes)?;
    if sha256(&claim.payload(attestor.digest())) != *signature.digest() {
        return Err(malformed(
            "the recorded payload digest is not the digest of the signed payload",
        ));
    }
    Ok(())
}

impl PtrRuntime {
    /// The attestor keys this runtime trusts.
    pub fn expert_attestors(&self) -> &ExpertAttestorRegistry {
        &self.attestors
    }

    /// [`PtrRuntime::admit_expert_proposal_with_inputs`] with an attestation: the attestor, the
    /// run and the signature join the dependency block ([`ExpertAttestation::attested_inputs`]),
    /// and the attestation must name this proposal, this proposal's artifact, base model and
    /// results and verify, strictly, under the key the registry holds for its attestor and
    /// epoch. The gate `report` must be bound to the attested block.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check: the operation produces an artifact,
    /// the four bindings, the attestor is registered, the key is the one the attestation names,
    /// the signature verifies, then every check of the plain path (a revoked attestor and a
    /// reused run id among them); nothing is appended on error.
    pub fn admit_attested_expert_proposal(
        &mut self,
        proposal: &ExpertProposal,
        report: &BoundGateReport,
        grant: &ExpertAdmissionGrant,
        inputs: ExpertInputs,
        attestation: &ExpertAttestation,
    ) -> Result<CommitIndex, RuntimeError> {
        let refuse = |error| Err(RuntimeError::Expert(error));
        let claim = &attestation.claim;
        let results = proposal.op().results();
        let Some(first) = results.first() else {
            return refuse(ExpertAdmissionError::InvalidRecord(
                "an attestation on an operation that creates no artifact".to_owned(),
            ));
        };
        if claim.proposal_id != *proposal.proposal_id().as_bytes() {
            return refuse(ExpertAdmissionError::AttestationProposalMismatch);
        }
        if claim.artifact_digest != expert_evidence_artifact_digest(proposal.op()) {
            return refuse(ExpertAdmissionError::AttestationArtifactMismatch);
        }
        if claim.base_model != *first.base_model() {
            return refuse(ExpertAdmissionError::AttestationBaseModelMismatch);
        }
        let named: Vec<(PodId, Generation)> = results
            .iter()
            .map(|result| (result.pod().clone(), result.generation()))
            .collect();
        if claim.results != named {
            return refuse(ExpertAdmissionError::AttestationResultMismatch);
        }
        let key = self
            .attestors
            .key(&claim.attestor, claim.epoch)
            .ok_or_else(|| RuntimeError::Expert(unknown(claim)))?;
        verify(claim, key, &attestation.key_id, &attestation.signature)
            .map_err(RuntimeError::Expert)?;
        let block = attestation
            .attested_inputs(&inputs, grant)
            .map_err(RuntimeError::Expert)?;
        self.admit_expert_record(proposal, report, grant, Some(block), Attested::Yes)
    }
}
