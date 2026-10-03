use crate::{PodOutput, PodTurnEvent};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use ptr_types::{
    Digest, EvidenceId, Generation, NamespaceId, PodAddress, PodId, PodRevisionAddress, ProjectId,
    ProvenanceRef, RequestId, Revision, ScopeId, SessionId, StateId, TraceId, TypeId,
};
use sha2::{Digest as ShaDigest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodEvidenceBundle {
    pub version: u8,
    pub session_id: SessionId,
    pub trace_id: TraceId,
    pub manifest_digest: Digest,
    pub artifact_digest: Digest,
    pub generation: Generation,
    pub revision: Revision,
    pub events: Vec<PodTurnEvent>,
    pub outputs: Vec<PodOutput>,
    pub chain_head: Digest,
    pub bundle_digest: Digest,
    pub signer_public_key: Option<[u8; 32]>,
    pub signature: Option<[u8; 64]>,
    record_order: Vec<EvidenceRecordRef>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvidenceRecordRef {
    Event(usize),
    Output(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceError {
    InvalidHeader,
    InvalidEventBinding,
    InvalidSequence,
    InvalidOutputBinding,
    Unsealed,
    DigestMismatch,
    MissingSignature,
    InvalidSignature,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayedEvidence {
    pub events: Vec<PodTurnEvent>,
    pub outputs: Vec<PodOutput>,
}

pub trait EvidenceSigner: Send + Sync {
    fn public_key(&self) -> [u8; 32];
    fn sign_digest(&self, digest: &Digest) -> [u8; 64];
}

pub trait EvidenceVerifier: Send + Sync {
    fn verify_digest(&self, digest: &Digest, signature: &[u8; 64]) -> Result<(), EvidenceError>;
}

pub struct Ed25519EvidenceSigner {
    key: SigningKey,
}

impl Ed25519EvidenceSigner {
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        Self {
            key: SigningKey::from_bytes(bytes),
        }
    }
}

impl EvidenceSigner for Ed25519EvidenceSigner {
    fn public_key(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    fn sign_digest(&self, digest: &Digest) -> [u8; 64] {
        self.key.sign(digest).to_bytes()
    }
}

pub struct Ed25519EvidenceVerifier {
    key: VerifyingKey,
}

impl Ed25519EvidenceVerifier {
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, EvidenceError> {
        Ok(Self {
            key: VerifyingKey::from_bytes(bytes).map_err(|_| EvidenceError::InvalidSignature)?,
        })
    }
}

impl EvidenceVerifier for Ed25519EvidenceVerifier {
    fn verify_digest(&self, digest: &Digest, signature: &[u8; 64]) -> Result<(), EvidenceError> {
        self.key
            .verify(digest, &Signature::from_bytes(signature))
            .map_err(|_| EvidenceError::InvalidSignature)
    }
}

impl PodEvidenceBundle {
    pub fn new(
        session_id: SessionId,
        trace_id: TraceId,
        manifest_digest: Digest,
        artifact_digest: Digest,
        generation: Generation,
        revision: Revision,
    ) -> Result<Self, EvidenceError> {
        if session_id.0.is_empty()
            || trace_id.0.is_empty()
            || manifest_digest == [0; 32]
            || artifact_digest == [0; 32]
            || generation.0 == 0
        {
            return Err(EvidenceError::InvalidHeader);
        }
        Ok(Self {
            version: 1,
            session_id,
            trace_id,
            manifest_digest,
            artifact_digest,
            generation,
            revision,
            events: Vec::new(),
            outputs: Vec::new(),
            chain_head: [0; 32],
            bundle_digest: [0; 32],
            signer_public_key: None,
            signature: None,
            record_order: Vec::new(),
        })
    }

    pub fn append_event(&mut self, event: PodTurnEvent) -> Result<(), EvidenceError> {
        if event.session_id != self.session_id
            || event.generation != self.generation
            || event.manifest_digest != self.manifest_digest
            || event.artifact_digest != self.artifact_digest
        {
            return Err(EvidenceError::InvalidEventBinding);
        }
        let expected_sequence = self.events.len() as u64 + 1;
        if event.sequence != expected_sequence {
            return Err(EvidenceError::InvalidSequence);
        }
        self.chain_head = chain_digest(self.chain_head, &canonical_event(&event));
        self.events.push(event);
        self.record_order
            .push(EvidenceRecordRef::Event(self.events.len() - 1));
        self.bundle_digest = [0; 32];
        self.signature = None;
        Ok(())
    }

    pub fn append_output(&mut self, output: PodOutput) -> Result<(), EvidenceError> {
        if output.generation != self.generation
            || output.manifest_digest != self.manifest_digest
            || output.artifact_digest != self.artifact_digest
            || output.revision != self.revision
        {
            return Err(EvidenceError::InvalidOutputBinding);
        }
        self.chain_head = chain_digest(self.chain_head, &canonical_output(&output));
        self.outputs.push(output);
        self.record_order
            .push(EvidenceRecordRef::Output(self.outputs.len() - 1));
        self.bundle_digest = [0; 32];
        self.signature = None;
        Ok(())
    }

    pub fn seal(&mut self) -> Result<Digest, EvidenceError> {
        self.verify_chain()?;
        self.signer_public_key = None;
        self.signature = None;
        self.bundle_digest = Sha256::digest(canonical_bundle(self, false)).into();
        Ok(self.bundle_digest)
    }

    pub fn verify(&self) -> Result<(), EvidenceError> {
        self.verify_chain()?;
        if self.bundle_digest == [0; 32] {
            return Err(EvidenceError::Unsealed);
        }
        let expected = Sha256::digest(canonical_bundle(self, false));
        if expected[..] != self.bundle_digest[..] {
            return Err(EvidenceError::DigestMismatch);
        }
        Ok(())
    }

    pub fn replay(&self) -> Result<ReplayedEvidence, EvidenceError> {
        self.verify()?;
        Ok(ReplayedEvidence {
            events: self.events.clone(),
            outputs: self.outputs.clone(),
        })
    }

    pub fn sign(&mut self, key: &SigningKey) -> Result<[u8; 64], EvidenceError> {
        if self.bundle_digest == [0; 32] {
            self.seal()?;
        } else {
            self.verify()?;
        }
        let signature = key.sign(&self.bundle_digest).to_bytes();
        self.signer_public_key = Some(key.verifying_key().to_bytes());
        self.signature = Some(signature);
        Ok(signature)
    }

    pub fn sign_with<S: EvidenceSigner>(&mut self, signer: &S) -> Result<[u8; 64], EvidenceError> {
        if self.bundle_digest == [0; 32] {
            self.seal()?;
        } else {
            self.verify()?;
        }
        let signature = signer.sign_digest(&self.bundle_digest);
        self.signer_public_key = Some(signer.public_key());
        self.signature = Some(signature);
        Ok(signature)
    }

    pub fn verify_signature(&self) -> Result<(), EvidenceError> {
        self.verify()?;
        let public_key = self
            .signer_public_key
            .ok_or(EvidenceError::MissingSignature)?;
        let signature = self.signature.ok_or(EvidenceError::MissingSignature)?;
        let key =
            VerifyingKey::from_bytes(&public_key).map_err(|_| EvidenceError::InvalidSignature)?;
        key.verify(&self.bundle_digest, &Signature::from_bytes(&signature))
            .map_err(|_| EvidenceError::InvalidSignature)
    }

    pub fn verify_with<V: EvidenceVerifier>(&self, verifier: &V) -> Result<(), EvidenceError> {
        self.verify()?;
        let signature = self.signature.ok_or(EvidenceError::MissingSignature)?;
        verifier.verify_digest(&self.bundle_digest, &signature)
    }

    /// Versioned opaque representation for the durable ledger. The ledger
    /// stores these bytes but does not interpret their Pod-specific schema.
    pub fn encode_canonical(&self) -> Result<Vec<u8>, EvidenceError> {
        self.verify()?;
        let mut bytes = canonical_bundle(self, true);
        bytes.extend_from_slice(&(self.record_order.len() as u64).to_le_bytes());
        for record in &self.record_order {
            match record {
                EvidenceRecordRef::Event(index) => {
                    bytes.push(0);
                    put_bytes(&mut bytes, &canonical_event(&self.events[*index]));
                }
                EvidenceRecordRef::Output(index) => {
                    bytes.push(1);
                    put_bytes(&mut bytes, &canonical_output(&self.outputs[*index]));
                }
            }
        }
        Ok(bytes)
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, EvidenceError> {
        let mut cursor = EvidenceCursor::new(bytes);
        let version = cursor.u8()?;
        if version != 1 {
            return Err(EvidenceError::InvalidHeader);
        }
        let session_text = cursor.string()?;
        let trace_text = cursor.string()?;
        let session_id = SessionId::from(session_text.as_str());
        let trace_id = TraceId::from(trace_text.as_str());
        let manifest_digest = cursor.digest()?;
        let artifact_digest = cursor.digest()?;
        let generation = Generation(cursor.u64()?);
        let revision = Revision(cursor.u64()?);
        let chain_head = cursor.digest()?;
        let event_count = cursor.u64()? as usize;
        let output_count = cursor.u64()? as usize;
        let bundle_digest = cursor.digest()?;
        let (signer_public_key, signature) = match cursor.u8()? {
            0 => (None, None),
            1 => {
                let public_key = cursor.digest()?;
                let signature = cursor.signature_bytes()?;
                (Some(public_key), Some(signature))
            }
            _ => return Err(EvidenceError::InvalidHeader),
        };
        let mut bundle = Self::new(
            session_id,
            trace_id,
            manifest_digest,
            artifact_digest,
            generation,
            revision,
        )?;
        let record_count = cursor.u64()? as usize;
        for _ in 0..record_count {
            match cursor.u8()? {
                0 => {
                    let record = cursor.bytes()?;
                    bundle.append_event(decode_event(&record)?)?;
                }
                1 => {
                    let record = cursor.bytes()?;
                    bundle.append_output(decode_output(&record)?)?;
                }
                _ => return Err(EvidenceError::InvalidHeader),
            }
        }
        if bundle.events.len() != event_count || bundle.outputs.len() != output_count {
            return Err(EvidenceError::DigestMismatch);
        }
        if !cursor.finished() {
            return Err(EvidenceError::InvalidHeader);
        }
        if bundle.chain_head != chain_head {
            return Err(EvidenceError::DigestMismatch);
        }
        bundle.bundle_digest = bundle_digest;
        bundle.signer_public_key = signer_public_key;
        bundle.signature = signature;
        bundle.verify()?;
        Ok(bundle)
    }

    fn verify_chain(&self) -> Result<(), EvidenceError> {
        let mut head = [0; 32];
        for record in &self.record_order {
            let bytes = match record {
                EvidenceRecordRef::Event(index) => self
                    .events
                    .get(*index)
                    .map(canonical_event)
                    .ok_or(EvidenceError::DigestMismatch)?,
                EvidenceRecordRef::Output(index) => self
                    .outputs
                    .get(*index)
                    .map(canonical_output)
                    .ok_or(EvidenceError::DigestMismatch)?,
            };
            head = chain_digest(head, &bytes);
        }
        if head != self.chain_head {
            return Err(EvidenceError::DigestMismatch);
        }
        Ok(())
    }
}

fn chain_digest(previous: Digest, record: &[u8]) -> Digest {
    let mut bytes = Vec::with_capacity(32 + record.len());
    bytes.extend_from_slice(&previous);
    bytes.extend_from_slice(record);
    Sha256::digest(bytes).into()
}

fn canonical_bundle(bundle: &PodEvidenceBundle, include_digest: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(bundle.version);
    put_string(&mut bytes, &bundle.session_id.0);
    put_string(&mut bytes, &bundle.trace_id.0);
    bytes.extend_from_slice(&bundle.manifest_digest);
    bytes.extend_from_slice(&bundle.artifact_digest);
    bytes.extend_from_slice(&bundle.generation.0.to_le_bytes());
    bytes.extend_from_slice(&bundle.revision.0.to_le_bytes());
    bytes.extend_from_slice(&bundle.chain_head);
    bytes.extend_from_slice(&(bundle.events.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(bundle.outputs.len() as u64).to_le_bytes());
    if include_digest {
        bytes.extend_from_slice(&bundle.bundle_digest);
        match (bundle.signer_public_key, bundle.signature) {
            (Some(public_key), Some(signature)) => {
                bytes.push(1);
                bytes.extend_from_slice(&public_key);
                bytes.extend_from_slice(&signature);
            }
            (None, None) => bytes.push(0),
            _ => bytes.push(2),
        }
    }
    bytes
}

fn canonical_event(event: &PodTurnEvent) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_string(&mut bytes, &format!("{:?}", event.kind));
    bytes.extend_from_slice(&event.sequence.to_le_bytes());
    put_string(&mut bytes, &event.session_id.0);
    put_string(&mut bytes, &event.scope_id.0);
    bytes.extend_from_slice(&event.turn_id.to_le_bytes());
    put_string(&mut bytes, &event.request_id.0);
    put_string(&mut bytes, &event.pod.address.project.0);
    put_string(&mut bytes, &event.pod.address.namespace.0);
    put_string(&mut bytes, &event.pod.address.pod_id.0);
    bytes.extend_from_slice(&event.pod.semantic_revision);
    bytes.extend_from_slice(&event.pod.generation.0.to_le_bytes());
    bytes.extend_from_slice(&event.generation.0.to_le_bytes());
    bytes.extend_from_slice(&event.manifest_digest);
    bytes.extend_from_slice(&event.artifact_digest);
    put_optional_type(&mut bytes, event.input_type.as_ref());
    put_optional_type(&mut bytes, event.output_type.as_ref());
    put_optional_string(&mut bytes, event.state_before.as_ref().map(|id| &id.0));
    put_optional_string(&mut bytes, event.state_after.as_ref().map(|id| &id.0));
    bytes.extend_from_slice(&event.revision.0.to_le_bytes());
    bytes
}

fn canonical_output(output: &PodOutput) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_string(&mut bytes, &format!("{:?}", output.kind));
    put_string(&mut bytes, &output.payload.type_id.0);
    bytes.extend_from_slice(&(output.payload.bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&output.payload.bytes);
    bytes.extend_from_slice(&output.generation.0.to_le_bytes());
    bytes.extend_from_slice(&output.manifest_digest);
    bytes.extend_from_slice(&output.artifact_digest);
    bytes.extend_from_slice(&output.revision.0.to_le_bytes());
    bytes.push(u8::from(output.verified));
    bytes.extend_from_slice(&(output.dependencies.len() as u64).to_le_bytes());
    for dependency in &output.dependencies {
        bytes.extend_from_slice(dependency);
    }
    bytes.extend_from_slice(&(output.provenance.len() as u64).to_le_bytes());
    for provenance in &output.provenance {
        put_string(&mut bytes, &provenance.source.0);
        put_optional_string(&mut bytes, provenance.note.as_ref());
    }
    bytes
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn put_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

struct EvidenceCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> EvidenceCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, length: usize) -> Result<&'a [u8], EvidenceError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(EvidenceError::InvalidHeader)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(EvidenceError::InvalidHeader)?;
        self.offset = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, EvidenceError> {
        Ok(self.take(1)?[0])
    }
    fn u64(&mut self) -> Result<u64, EvidenceError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| EvidenceError::InvalidHeader)?,
        ))
    }
    fn digest(&mut self) -> Result<Digest, EvidenceError> {
        self.take(32)?
            .try_into()
            .map_err(|_| EvidenceError::InvalidHeader)
    }
    fn string(&mut self) -> Result<String, EvidenceError> {
        String::from_utf8(self.bytes()?).map_err(|_| EvidenceError::InvalidHeader)
    }
    fn bytes(&mut self) -> Result<Vec<u8>, EvidenceError> {
        let length = usize::try_from(self.u64()?).map_err(|_| EvidenceError::InvalidHeader)?;
        Ok(self.take(length)?.to_vec())
    }
    fn optional_string(&mut self) -> Result<Option<String>, EvidenceError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.string()?)),
            _ => Err(EvidenceError::InvalidHeader),
        }
    }
    fn optional_type(&mut self) -> Result<Option<TypeId>, EvidenceError> {
        Ok(self.optional_string()?.map(TypeId))
    }
    fn signature_bytes(&mut self) -> Result<[u8; 64], EvidenceError> {
        self.take(64)?
            .try_into()
            .map_err(|_| EvidenceError::InvalidHeader)
    }
    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn put_optional_string(bytes: &mut Vec<u8>, value: Option<&String>) {
    match value {
        Some(value) => {
            bytes.push(1);
            put_string(bytes, value);
        }
        None => bytes.push(0),
    }
}

fn put_optional_type(bytes: &mut Vec<u8>, value: Option<&TypeId>) {
    put_optional_string(bytes, value.map(|value| &value.0));
}

fn decode_event(bytes: &[u8]) -> Result<PodTurnEvent, EvidenceError> {
    let mut cursor = EvidenceCursor::new(bytes);
    let kind = match cursor.string()?.as_str() {
        "SessionCreated" => crate::PodTurnKind::SessionCreated,
        "RequestStarted" => crate::PodTurnKind::RequestStarted,
        "InputReceived" => crate::PodTurnKind::InputReceived,
        "PartialOutput" => crate::PodTurnKind::PartialOutput,
        "ToolCall" => crate::PodTurnKind::ToolCall,
        "Observation" => crate::PodTurnKind::Observation,
        "Hypothesis" => crate::PodTurnKind::Hypothesis,
        "TurnCommitted" => crate::PodTurnKind::TurnCommitted,
        "TurnInterrupted" => crate::PodTurnKind::TurnInterrupted,
        "TurnResumed" => crate::PodTurnKind::TurnResumed,
        "RequestCompleted" => crate::PodTurnKind::RequestCompleted,
        "RequestFailed" => crate::PodTurnKind::RequestFailed,
        "RequestUncertain" => crate::PodTurnKind::RequestUncertain,
        "SessionClosed" => crate::PodTurnKind::SessionClosed,
        _ => return Err(EvidenceError::InvalidHeader),
    };
    let sequence = cursor.u64()?;
    let session_text = cursor.string()?;
    let scope_text = cursor.string()?;
    let turn_id = cursor.u64()?;
    let request_text = cursor.string()?;
    let project_text = cursor.string()?;
    let namespace_text = cursor.string()?;
    let pod_text = cursor.string()?;
    let semantic_revision = cursor.digest()?;
    let pod_generation = Generation(cursor.u64()?);
    let generation = Generation(cursor.u64()?);
    let manifest_digest = cursor.digest()?;
    let artifact_digest = cursor.digest()?;
    let input_type = cursor.optional_type()?;
    let output_type = cursor.optional_type()?;
    let state_before = cursor.optional_string()?.map(StateId);
    let state_after = cursor.optional_string()?.map(StateId);
    let revision = Revision(cursor.u64()?);
    if !cursor.finished() {
        return Err(EvidenceError::InvalidHeader);
    }
    let address = PodAddress::new(
        ProjectId(project_text),
        NamespaceId(namespace_text),
        PodId(pod_text),
    )
    .map_err(|_| EvidenceError::InvalidHeader)?;
    let pod = PodRevisionAddress {
        address,
        semantic_revision,
        generation: pod_generation,
    };
    pod.validate().map_err(|_| EvidenceError::InvalidHeader)?;
    Ok(PodTurnEvent {
        kind,
        sequence,
        session_id: SessionId(session_text),
        scope_id: ScopeId(scope_text),
        turn_id,
        request_id: RequestId(request_text),
        pod,
        generation,
        manifest_digest,
        artifact_digest,
        input_type,
        output_type,
        state_before,
        state_after,
        revision,
    })
}

fn decode_output(bytes: &[u8]) -> Result<PodOutput, EvidenceError> {
    let mut cursor = EvidenceCursor::new(bytes);
    let kind = match cursor.string()?.as_str() {
        "Observation" => crate::PodOutputKind::Observation,
        "Candidate" => crate::PodOutputKind::Candidate,
        "Hypothesis" => crate::PodOutputKind::Hypothesis,
        "ToolResult" => crate::PodOutputKind::ToolResult,
        "EnvironmentObservation" => crate::PodOutputKind::EnvironmentObservation,
        "StateDelta" => crate::PodOutputKind::StateDelta,
        "ActionProposal" => crate::PodOutputKind::ActionProposal,
        "VerifiedResult" => crate::PodOutputKind::VerifiedResult,
        _ => return Err(EvidenceError::InvalidHeader),
    };
    let type_id = TypeId(cursor.string()?);
    let payload = ptr_protocol::TypedPayload {
        type_id,
        bytes: cursor.bytes()?,
    };
    let generation = Generation(cursor.u64()?);
    let manifest_digest = cursor.digest()?;
    let artifact_digest = cursor.digest()?;
    let revision = Revision(cursor.u64()?);
    let verified = match cursor.u8()? {
        0 => false,
        1 => true,
        _ => return Err(EvidenceError::InvalidHeader),
    };
    let dependency_count =
        usize::try_from(cursor.u64()?).map_err(|_| EvidenceError::InvalidHeader)?;
    let mut dependencies = Vec::with_capacity(dependency_count);
    for _ in 0..dependency_count {
        dependencies.push(cursor.digest()?);
    }
    let provenance_count =
        usize::try_from(cursor.u64()?).map_err(|_| EvidenceError::InvalidHeader)?;
    let mut provenance = Vec::with_capacity(provenance_count);
    for _ in 0..provenance_count {
        let source = EvidenceId(cursor.string()?);
        let note = cursor.optional_string()?;
        provenance.push(ProvenanceRef { source, note });
    }
    if !cursor.finished() {
        return Err(EvidenceError::InvalidHeader);
    }
    Ok(PodOutput {
        kind,
        payload,
        generation,
        manifest_digest,
        artifact_digest,
        provenance,
        dependencies,
        revision,
        verified,
    })
}
