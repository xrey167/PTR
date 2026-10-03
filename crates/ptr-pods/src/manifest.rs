use ptr_types::{Digest, Generation, PrincipalId, Revision};
use sha2::{Digest as ShaDigest, Sha256};
use std::io;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LineageBinding {
    pub key: String,
    pub generation: Generation,
    pub digest: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionManifest {
    pub generation: Generation,
    pub knowledge: Vec<LineageBinding>,
    pub artifacts: Vec<LineageBinding>,
    pub origins: Vec<String>,
    pub reader: Option<LineageBinding>,
    /// Optional model/reader adapter artifact. This was added in canonical
    /// schema v2; v1 manifests decode with `None` for compatibility.
    pub adapter: Option<LineageBinding>,
    pub snapshot_revision: Revision,
    pub snapshot_digest: Digest,
    pub principal: PrincipalId,
    pub policy_revision: Revision,
    pub manifest_digest: Digest,
}

impl ExecutionManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        generation: Generation,
        knowledge: Vec<LineageBinding>,
        artifacts: Vec<LineageBinding>,
        origins: Vec<String>,
        reader: Option<LineageBinding>,
        snapshot_revision: Revision,
        snapshot_digest: Digest,
        principal: PrincipalId,
        policy_revision: Revision,
    ) -> Result<Self, ManifestError> {
        Self::build_with_adapter(
            generation,
            knowledge,
            artifacts,
            origins,
            reader,
            None,
            snapshot_revision,
            snapshot_digest,
            principal,
            policy_revision,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_with_adapter(
        generation: Generation,
        mut knowledge: Vec<LineageBinding>,
        mut artifacts: Vec<LineageBinding>,
        mut origins: Vec<String>,
        reader: Option<LineageBinding>,
        adapter: Option<LineageBinding>,
        snapshot_revision: Revision,
        snapshot_digest: Digest,
        principal: PrincipalId,
        policy_revision: Revision,
    ) -> Result<Self, ManifestError> {
        knowledge.sort_by(|left, right| {
            left.key
                .cmp(&right.key)
                .then(left.generation.cmp(&right.generation))
                .then(left.digest.cmp(&right.digest))
        });
        artifacts.sort_by(|left, right| {
            left.key
                .cmp(&right.key)
                .then(left.generation.cmp(&right.generation))
                .then(left.digest.cmp(&right.digest))
        });
        origins.sort();
        let manifest = Self {
            generation,
            knowledge,
            artifacts,
            origins,
            reader,
            adapter,
            snapshot_revision,
            snapshot_digest,
            principal,
            policy_revision,
            manifest_digest: [0; 32],
        };
        manifest.validate_structure()?;
        let digest = manifest.calculate_digest();
        Ok(Self {
            manifest_digest: digest,
            ..manifest
        })
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        self.validate_structure()?;
        if self.manifest_digest == [0; 32] {
            return Err(ManifestError::MissingDigest);
        }
        if self.calculate_digest() != self.manifest_digest {
            return Err(ManifestError::DigestMismatch);
        }
        Ok(())
    }

    /// Versioned canonical representation used inside runtime journal events.
    /// The ledger stores these bytes opaquely; this crate owns their schema and
    /// validates the embedded digest again on decode.
    pub fn encode_canonical(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        let mut out = vec![2u8];
        put_u64(&mut out, self.generation.0);
        put_bindings(&mut out, &self.knowledge);
        put_bindings(&mut out, &self.artifacts);
        put_strings(&mut out, &self.origins);
        match &self.reader {
            Some(reader) => {
                out.push(1);
                put_binding(&mut out, reader);
            }
            None => out.push(0),
        }
        match &self.adapter {
            Some(adapter) => {
                out.push(1);
                put_binding(&mut out, adapter);
            }
            None => out.push(0),
        }
        put_u64(&mut out, self.snapshot_revision.0);
        out.extend_from_slice(&self.snapshot_digest);
        put_string(&mut out, &self.principal.0);
        put_u64(&mut out, self.policy_revision.0);
        out.extend_from_slice(&self.manifest_digest);
        Ok(out)
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, ManifestError> {
        let mut cursor = ManifestCursor { bytes, offset: 0 };
        let version = cursor.u8()?;
        if !matches!(version, 1 | 2) {
            return Err(ManifestError::InvalidEncoding);
        }
        let manifest = Self {
            generation: Generation(cursor.u64()?),
            knowledge: cursor.bindings()?,
            artifacts: cursor.bindings()?,
            origins: cursor.strings()?,
            reader: match cursor.u8()? {
                0 => None,
                1 => Some(cursor.binding()?),
                _ => return Err(ManifestError::InvalidEncoding),
            },
            adapter: if version >= 2 {
                match cursor.u8()? {
                    0 => None,
                    1 => Some(cursor.binding()?),
                    _ => return Err(ManifestError::InvalidEncoding),
                }
            } else {
                None
            },
            snapshot_revision: Revision(cursor.u64()?),
            snapshot_digest: cursor.digest()?,
            principal: PrincipalId(cursor.string()?),
            policy_revision: Revision(cursor.u64()?),
            manifest_digest: cursor.digest()?,
        };
        if cursor.offset != bytes.len() {
            return Err(ManifestError::InvalidEncoding);
        }
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate_structure(&self) -> Result<(), ManifestError> {
        if self.generation.0 == 0 || self.knowledge.is_empty() || self.artifacts.is_empty() {
            return Err(ManifestError::MissingLineage);
        }
        if self.origins.is_empty() || self.principal.0.is_empty() {
            return Err(ManifestError::MissingProvenance);
        }
        if self.snapshot_digest == [0; 32] {
            return Err(ManifestError::MissingDigest);
        }
        validate_unique_bindings(&self.knowledge)?;
        validate_unique_bindings(&self.artifacts)?;
        for binding in self
            .knowledge
            .iter()
            .chain(self.artifacts.iter())
            .chain(self.reader.iter())
            .chain(self.adapter.iter())
        {
            if binding.key.is_empty() || binding.generation.0 == 0 || binding.digest == [0; 32] {
                return Err(ManifestError::InvalidBinding);
            }
        }
        if let Some(reader) = &self.reader {
            if reader.generation != self.generation {
                return Err(ManifestError::ReaderGenerationMismatch);
            }
        }
        if let Some(adapter) = &self.adapter {
            if adapter.generation != self.generation {
                return Err(ManifestError::AdapterGenerationMismatch);
            }
        }
        Ok(())
    }

    fn calculate_digest(&self) -> Digest {
        let mut bytes = Vec::new();
        macro_rules! put {
            ($value:expr) => {{
                let value = $value.to_string();
                bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                bytes.extend_from_slice(value.as_bytes());
            }};
        }
        put!(self.generation.0);
        for binding in self
            .knowledge
            .iter()
            .chain(self.artifacts.iter())
            .chain(self.reader.iter())
            .chain(self.adapter.iter())
        {
            put!(&binding.key);
            put!(binding.generation.0);
            bytes.extend_from_slice(&binding.digest);
        }
        for origin in &self.origins {
            put!(origin);
        }
        put!(self.snapshot_revision.0);
        bytes.extend_from_slice(&self.snapshot_digest);
        put!(&self.principal.0);
        put!(self.policy_revision.0);
        Sha256::digest(bytes).into()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactLifecycle {
    Candidate,
    Trained,
    Evaluated,
    Approved,
    Active,
    Superseded,
    Retired,
    Revoked,
}

pub struct LifecycleGate;

impl LifecycleGate {
    pub fn transition(
        current: ArtifactLifecycle,
        next: ArtifactLifecycle,
        manifest_valid: bool,
        provenance_valid: bool,
        base_identity_valid: bool,
        verifier_passed: bool,
    ) -> Result<ArtifactLifecycle, ManifestError> {
        let allowed = matches!(
            (current, next),
            (ArtifactLifecycle::Candidate, ArtifactLifecycle::Trained)
                | (ArtifactLifecycle::Trained, ArtifactLifecycle::Evaluated)
                | (ArtifactLifecycle::Evaluated, ArtifactLifecycle::Approved)
                | (ArtifactLifecycle::Approved, ArtifactLifecycle::Active)
                | (ArtifactLifecycle::Active, ArtifactLifecycle::Superseded)
                | (ArtifactLifecycle::Active, ArtifactLifecycle::Retired)
                | (ArtifactLifecycle::Active, ArtifactLifecycle::Revoked)
        );
        if !allowed {
            return Err(ManifestError::InvalidLifecycleTransition);
        }
        if matches!(
            next,
            ArtifactLifecycle::Evaluated | ArtifactLifecycle::Approved | ArtifactLifecycle::Active
        ) && !(manifest_valid && provenance_valid && base_identity_valid && verifier_passed)
        {
            return Err(ManifestError::AdmissionRequired);
        }
        Ok(next)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestError {
    MissingLineage,
    MissingProvenance,
    MissingDigest,
    InvalidBinding,
    DuplicateBinding,
    ReaderGenerationMismatch,
    AdapterGenerationMismatch,
    DigestMismatch,
    InvalidLifecycleTransition,
    AdmissionRequired,
    InvalidEncoding,
}

fn validate_unique_bindings(bindings: &[LineageBinding]) -> Result<(), ManifestError> {
    for pair in bindings.windows(2) {
        if pair[0].key == pair[1].key {
            return Err(ManifestError::DuplicateBinding);
        }
    }
    Ok(())
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    let length = u32::try_from(value.len()).expect("manifest string fits u32");
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_binding(out: &mut Vec<u8>, binding: &LineageBinding) {
    put_string(out, &binding.key);
    put_u64(out, binding.generation.0);
    out.extend_from_slice(&binding.digest);
}

fn put_bindings(out: &mut Vec<u8>, bindings: &[LineageBinding]) {
    let count = u32::try_from(bindings.len()).expect("manifest bindings fit u32");
    out.extend_from_slice(&count.to_le_bytes());
    for binding in bindings {
        put_binding(out, binding);
    }
}

fn put_strings(out: &mut Vec<u8>, values: &[String]) {
    let count = u32::try_from(values.len()).expect("manifest strings fit u32");
    out.extend_from_slice(&count.to_le_bytes());
    for value in values {
        put_string(out, value);
    }
}

struct ManifestCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ManifestCursor<'a> {
    fn u8(&mut self) -> Result<u8, ManifestError> {
        let byte = *self
            .bytes
            .get(self.offset)
            .ok_or(ManifestError::InvalidEncoding)?;
        self.offset += 1;
        Ok(byte)
    }

    fn u32(&mut self) -> Result<u32, ManifestError> {
        let end = self
            .offset
            .checked_add(4)
            .ok_or(ManifestError::InvalidEncoding)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(ManifestError::InvalidEncoding)?;
        self.offset = end;
        Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
    }

    fn u64(&mut self) -> Result<u64, ManifestError> {
        let end = self
            .offset
            .checked_add(8)
            .ok_or(ManifestError::InvalidEncoding)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(ManifestError::InvalidEncoding)?;
        self.offset = end;
        Ok(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
    }

    fn bytes(&mut self, length: usize) -> Result<&'a [u8], ManifestError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ManifestError::InvalidEncoding)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(ManifestError::InvalidEncoding)?;
        self.offset = end;
        Ok(bytes)
    }

    fn digest(&mut self) -> Result<Digest, ManifestError> {
        Ok(self.bytes(32)?.try_into().expect("digest length"))
    }

    fn string(&mut self) -> Result<String, ManifestError> {
        let length = self.u32()? as usize;
        String::from_utf8(self.bytes(length)?.to_vec()).map_err(|_| ManifestError::InvalidEncoding)
    }

    fn binding(&mut self) -> Result<LineageBinding, ManifestError> {
        Ok(LineageBinding {
            key: self.string()?,
            generation: Generation(self.u64()?),
            digest: self.digest()?,
        })
    }

    fn bindings(&mut self) -> Result<Vec<LineageBinding>, ManifestError> {
        let count = self.u32()? as usize;
        if count > 4096 {
            return Err(ManifestError::InvalidEncoding);
        }
        (0..count).map(|_| self.binding()).collect()
    }

    fn strings(&mut self) -> Result<Vec<String>, ManifestError> {
        let count = self.u32()? as usize;
        if count > 4096 {
            return Err(ManifestError::InvalidEncoding);
        }
        (0..count).map(|_| self.string()).collect()
    }
}

impl From<io::Error> for ManifestError {
    fn from(_: io::Error) -> Self {
        ManifestError::InvalidEncoding
    }
}
