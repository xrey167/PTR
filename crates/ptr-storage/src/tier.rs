use ptr_types::{Digest, Generation, Revision};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(1);

pub type TierFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, TierError>> + Send + 'a>>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TierObjectDomain {
    KvSnapshot,
    ModelWeight,
    Adapter,
    Artifact,
}

impl TierObjectDomain {
    pub fn code(self) -> u8 {
        match self {
            Self::KvSnapshot => 0,
            Self::ModelWeight => 1,
            Self::Adapter => 2,
            Self::Artifact => 3,
        }
    }

    pub fn from_code(code: u8) -> Result<Self, TierError> {
        match code {
            0 => Ok(Self::KvSnapshot),
            1 => Ok(Self::ModelWeight),
            2 => Ok(Self::Adapter),
            3 => Ok(Self::Artifact),
            _ => Err(TierError::InvalidManifest("unknown object domain".into())),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum StorageTier {
    Gpu,
    Cpu,
    Nvme,
    ObjectStore,
}

impl StorageTier {
    pub fn code(self) -> u8 {
        match self {
            Self::Gpu => 0,
            Self::Cpu => 1,
            Self::Nvme => 2,
            Self::ObjectStore => 3,
        }
    }

    pub fn from_code(code: u8) -> Result<Self, TierError> {
        match code {
            0 => Ok(Self::Gpu),
            1 => Ok(Self::Cpu),
            2 => Ok(Self::Nvme),
            3 => Ok(Self::ObjectStore),
            _ => Err(TierError::InvalidManifest("unknown storage tier".into())),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TierBackendId(pub String);

impl From<&str> for TierBackendId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkDescriptor {
    pub index: u32,
    pub offset: u64,
    pub length: u64,
    pub digest: Digest,
}

impl ChunkDescriptor {
    pub fn for_bytes(index: u32, offset: u64, bytes: &[u8]) -> Self {
        Self {
            index,
            offset,
            length: bytes.len() as u64,
            digest: digest(bytes),
        }
    }

    pub fn validate_bytes(&self, bytes: &[u8]) -> Result<(), TierError> {
        if self.length != bytes.len() as u64 {
            return Err(TierError::LengthMismatch);
        }
        if self.digest != digest(bytes) {
            return Err(TierError::DigestMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierObjectManifest {
    pub domain: TierObjectDomain,
    pub logical_id: String,
    pub generation: Generation,
    pub revision: Revision,
    pub schema_digest: Digest,
    pub root_digest: Digest,
    pub chunks: Vec<ChunkDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTierObject {
    pub manifest: TierObjectManifest,
    pub chunks: Vec<Vec<u8>>,
}

impl PreparedTierObject {
    pub fn from_bytes(
        domain: TierObjectDomain,
        logical_id: impl Into<String>,
        generation: Generation,
        revision: Revision,
        schema_digest: Digest,
        bytes: &[u8],
        chunk_size: usize,
    ) -> Result<Self, TierError> {
        if bytes.is_empty() || chunk_size == 0 {
            return Err(TierError::InvalidManifest(
                "tier objects and chunk size must be non-empty".into(),
            ));
        }
        let mut offset = 0u64;
        let mut chunks = Vec::new();
        let mut descriptors = Vec::new();
        for (index, value) in bytes.chunks(chunk_size).enumerate() {
            let descriptor = ChunkDescriptor::for_bytes(index as u32, offset, value);
            offset = offset
                .checked_add(descriptor.length)
                .ok_or_else(|| TierError::InvalidManifest("chunk range overflow".into()))?;
            descriptors.push(descriptor);
            chunks.push(value.to_vec());
        }
        Ok(Self {
            manifest: TierObjectManifest::new(
                domain,
                logical_id,
                generation,
                revision,
                schema_digest,
                descriptors,
            )?,
            chunks,
        })
    }

    pub fn verify(&self) -> Result<(), TierError> {
        self.manifest.validate()?;
        if self.chunks.len() != self.manifest.chunks.len() {
            return Err(TierError::InvalidManifest("chunk count mismatch".into()));
        }
        for (descriptor, bytes) in self.manifest.chunks.iter().zip(&self.chunks) {
            descriptor.validate_bytes(bytes)?;
        }
        Ok(())
    }

    pub fn reassemble(&self) -> Result<Vec<u8>, TierError> {
        self.verify()?;
        let capacity = self
            .manifest
            .chunks
            .iter()
            .try_fold(0usize, |total, chunk| {
                total.checked_add(chunk.length as usize)
            })
            .ok_or_else(|| TierError::InvalidManifest("object length overflow".into()))?;
        let mut bytes = Vec::with_capacity(capacity);
        for chunk in &self.chunks {
            bytes.extend_from_slice(chunk);
        }
        Ok(bytes)
    }
}

impl TierObjectManifest {
    pub fn new(
        domain: TierObjectDomain,
        logical_id: impl Into<String>,
        generation: Generation,
        revision: Revision,
        schema_digest: Digest,
        chunks: Vec<ChunkDescriptor>,
    ) -> Result<Self, TierError> {
        let mut manifest = Self {
            domain,
            logical_id: logical_id.into(),
            generation,
            revision,
            schema_digest,
            root_digest: [0; 32],
            chunks,
        };
        manifest.validate_structure()?;
        manifest.root_digest = manifest.canonical_digest();
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), TierError> {
        self.validate_structure()?;
        if self.root_digest != self.canonical_digest() {
            return Err(TierError::RootDigestMismatch);
        }
        Ok(())
    }

    pub fn canonical_digest(&self) -> Digest {
        let mut value = Vec::new();
        value.extend_from_slice(b"ptr-tier-object-v1\0");
        value.push(self.domain.code());
        put_text(&mut value, &self.logical_id);
        value.extend_from_slice(&self.generation.0.to_le_bytes());
        value.extend_from_slice(&self.revision.0.to_le_bytes());
        value.extend_from_slice(&self.schema_digest);
        value.extend_from_slice(&(self.chunks.len() as u64).to_le_bytes());
        for chunk in &self.chunks {
            value.extend_from_slice(&chunk.index.to_le_bytes());
            value.extend_from_slice(&chunk.offset.to_le_bytes());
            value.extend_from_slice(&chunk.length.to_le_bytes());
            value.extend_from_slice(&chunk.digest);
        }
        digest(&value)
    }

    pub fn encode_canonical(&self) -> Result<Vec<u8>, TierError> {
        self.validate()?;
        let mut out = b"PTRTIER1".to_vec();
        out.push(self.domain.code());
        put_text(&mut out, &self.logical_id);
        out.extend_from_slice(&self.generation.0.to_le_bytes());
        out.extend_from_slice(&self.revision.0.to_le_bytes());
        out.extend_from_slice(&self.schema_digest);
        out.extend_from_slice(&self.root_digest);
        out.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());
        for chunk in &self.chunks {
            out.extend_from_slice(&chunk.index.to_le_bytes());
            out.extend_from_slice(&chunk.offset.to_le_bytes());
            out.extend_from_slice(&chunk.length.to_le_bytes());
            out.extend_from_slice(&chunk.digest);
        }
        Ok(out)
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, TierError> {
        let mut cursor = TierCursor::new(bytes);
        if cursor.take(8)? != b"PTRTIER1" {
            return Err(TierError::InvalidManifest("invalid manifest magic".into()));
        }
        let domain = TierObjectDomain::from_code(cursor.u8()?)?;
        let logical_id = cursor.text()?;
        let generation = Generation(cursor.u64()?);
        let revision = Revision(cursor.u64()?);
        let schema_digest = cursor.digest()?;
        let root_digest = cursor.digest()?;
        let count = cursor.u32()? as usize;
        if count > 1_048_576 {
            return Err(TierError::InvalidManifest("too many chunks".into()));
        }
        let mut chunks = Vec::with_capacity(count);
        for _ in 0..count {
            chunks.push(ChunkDescriptor {
                index: cursor.u32()?,
                offset: cursor.u64()?,
                length: cursor.u64()?,
                digest: cursor.digest()?,
            });
        }
        if !cursor.finished() {
            return Err(TierError::InvalidManifest("trailing manifest bytes".into()));
        }
        let manifest = Self {
            domain,
            logical_id,
            generation,
            revision,
            schema_digest,
            root_digest,
            chunks,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate_structure(&self) -> Result<(), TierError> {
        if self.logical_id.trim().is_empty() || self.logical_id != self.logical_id.trim() {
            return Err(TierError::InvalidManifest("invalid logical id".into()));
        }
        if self.chunks.is_empty() {
            return Err(TierError::InvalidManifest("empty tier object".into()));
        }
        let mut next_offset = 0u64;
        for (expected, chunk) in self.chunks.iter().enumerate() {
            if chunk.index != expected as u32 || chunk.offset != next_offset || chunk.length == 0 {
                return Err(TierError::InvalidManifest(
                    "chunks must be contiguous and canonically ordered".into(),
                ));
            }
            next_offset = next_offset
                .checked_add(chunk.length)
                .ok_or_else(|| TierError::InvalidManifest("chunk range overflow".into()))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierCapabilities {
    pub tier: StorageTier,
    pub persistent: bool,
    pub range_reads: bool,
    pub conditional_create: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TierHealth {
    Healthy,
    Degraded,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkReceipt {
    pub backend: TierBackendId,
    pub digest: Digest,
    pub length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierIntegrityReport {
    pub digest: Digest,
    pub length: u64,
    pub valid: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TierError {
    InvalidBackendId,
    InvalidManifest(String),
    LengthMismatch,
    DigestMismatch,
    RootDigestMismatch,
    UnknownChunk,
    ImmutableConflict,
    CorruptChunk,
    Io(String),
    Backend(String),
}

pub trait TierBackend: Send + Sync {
    fn identity(&self) -> TierBackendId;
    fn capabilities(&self) -> TierCapabilities;
    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt>;
    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>>;
    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport>;
    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()>;
    fn health(&self) -> TierFuture<'_, TierHealth>;
}

pub struct CpuTierBackend {
    id: TierBackendId,
    values: Mutex<BTreeMap<Digest, Vec<u8>>>,
}

impl CpuTierBackend {
    pub fn new(id: TierBackendId) -> Self {
        Self {
            id,
            values: Mutex::new(BTreeMap::new()),
        }
    }
}

impl TierBackend for CpuTierBackend {
    fn identity(&self) -> TierBackendId {
        self.id.clone()
    }

    fn capabilities(&self) -> TierCapabilities {
        TierCapabilities {
            tier: StorageTier::Cpu,
            persistent: false,
            range_reads: false,
            conditional_create: true,
        }
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        Box::pin(async move {
            chunk.validate_bytes(&bytes)?;
            let mut values = self
                .values
                .lock()
                .map_err(|_| TierError::Backend("CPU tier lock poisoned".into()))?;
            if let Some(existing) = values.get(&chunk.digest) {
                if existing != &bytes {
                    return Err(TierError::ImmutableConflict);
                }
            } else {
                values.insert(chunk.digest, bytes);
            }
            Ok(ChunkReceipt {
                backend: self.id.clone(),
                digest: chunk.digest,
                length: chunk.length,
            })
        })
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        Box::pin(async move {
            let values = self
                .values
                .lock()
                .map_err(|_| TierError::Backend("CPU tier lock poisoned".into()))?;
            let bytes = values
                .get(&chunk.digest)
                .cloned()
                .ok_or(TierError::UnknownChunk)?;
            chunk
                .validate_bytes(&bytes)
                .map_err(|_| TierError::CorruptChunk)?;
            Ok(bytes)
        })
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        Box::pin(async move {
            let values = self
                .values
                .lock()
                .map_err(|_| TierError::Backend("CPU tier lock poisoned".into()))?;
            let Some(bytes) = values.get(&chunk.digest) else {
                return Err(TierError::UnknownChunk);
            };
            Ok(TierIntegrityReport {
                digest: digest(bytes),
                length: bytes.len() as u64,
                valid: chunk.validate_bytes(bytes).is_ok(),
            })
        })
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        Box::pin(async move {
            self.values
                .lock()
                .map_err(|_| TierError::Backend("CPU tier lock poisoned".into()))?
                .remove(&chunk.digest);
            Ok(())
        })
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        Box::pin(async { Ok(TierHealth::Healthy) })
    }
}

pub struct FileTierBackend {
    id: TierBackendId,
    root: PathBuf,
}

impl FileTierBackend {
    pub fn new(id: TierBackendId, root: impl AsRef<Path>) -> Result<Self, TierError> {
        if id.0.trim().is_empty() || id.0 != id.0.trim() {
            return Err(TierError::InvalidBackendId);
        }
        let root = root.as_ref().to_owned();
        fs::create_dir_all(root.join("chunks")).map_err(io_error)?;
        Ok(Self { id, root })
    }

    pub fn chunk_path(&self, chunk: &ChunkDescriptor) -> PathBuf {
        self.root.join("chunks").join(hex(&chunk.digest))
    }

    fn put_sync(&self, chunk: &ChunkDescriptor, bytes: &[u8]) -> Result<(), TierError> {
        chunk.validate_bytes(bytes)?;
        let destination = self.chunk_path(chunk);
        if destination.exists() {
            let existing = fs::read(&destination).map_err(io_error)?;
            if existing == bytes && chunk.validate_bytes(&existing).is_ok() {
                return Ok(());
            }
            return Err(TierError::ImmutableConflict);
        }
        let temp = destination.with_extension(format!(
            "{}.{}.tmp",
            std::process::id(),
            NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)
            .map_err(io_error)?;
        if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&temp);
            return Err(io_error(error));
        }
        drop(file);
        match fs::rename(&temp, &destination) {
            Ok(()) => {
                sync_parent(&destination)?;
                Ok(())
            }
            Err(_) if destination.exists() => {
                let _ = fs::remove_file(&temp);
                let existing = fs::read(&destination).map_err(io_error)?;
                if existing == bytes && chunk.validate_bytes(&existing).is_ok() {
                    Ok(())
                } else {
                    Err(TierError::ImmutableConflict)
                }
            }
            Err(error) => {
                let _ = fs::remove_file(&temp);
                Err(io_error(error))
            }
        }
    }
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> Result<(), TierError> {
    let parent = path.parent().ok_or_else(|| {
        TierError::Backend("content-addressed chunk has no parent directory".into())
    })?;
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(io_error)
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> Result<(), TierError> {
    Ok(())
}

impl TierBackend for FileTierBackend {
    fn identity(&self) -> TierBackendId {
        self.id.clone()
    }

    fn capabilities(&self) -> TierCapabilities {
        TierCapabilities {
            tier: StorageTier::Nvme,
            persistent: true,
            range_reads: true,
            conditional_create: true,
        }
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        Box::pin(async move {
            self.put_sync(&chunk, &bytes)?;
            Ok(ChunkReceipt {
                backend: self.id.clone(),
                digest: chunk.digest,
                length: chunk.length,
            })
        })
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        Box::pin(async move {
            let bytes = fs::read(self.chunk_path(chunk)).map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    TierError::UnknownChunk
                } else {
                    io_error(error)
                }
            })?;
            chunk
                .validate_bytes(&bytes)
                .map_err(|_| TierError::CorruptChunk)?;
            Ok(bytes)
        })
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        Box::pin(async move {
            let bytes = fs::read(self.chunk_path(chunk)).map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    TierError::UnknownChunk
                } else {
                    io_error(error)
                }
            })?;
            Ok(TierIntegrityReport {
                digest: digest(&bytes),
                length: bytes.len() as u64,
                valid: chunk.validate_bytes(&bytes).is_ok(),
            })
        })
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        Box::pin(async move {
            match fs::remove_file(self.chunk_path(chunk)) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(io_error(error)),
            }
        })
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        Box::pin(async move {
            if self.root.join("chunks").is_dir() {
                Ok(TierHealth::Healthy)
            } else {
                Ok(TierHealth::Unavailable)
            }
        })
    }
}

fn put_text(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

pub fn digest(bytes: &[u8]) -> Digest {
    Sha256::digest(bytes).into()
}

fn hex(value: &Digest) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn io_error(error: std::io::Error) -> TierError {
    TierError::Io(error.to_string())
}

struct TierCursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> TierCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], TierError> {
        let end = self
            .at
            .checked_add(len)
            .ok_or_else(|| TierError::InvalidManifest("manifest length overflow".into()))?;
        let value = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| TierError::InvalidManifest("truncated manifest".into()))?;
        self.at = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, TierError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, TierError> {
        self.take(4)?
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| TierError::InvalidManifest("invalid u32".into()))
    }

    fn u64(&mut self) -> Result<u64, TierError> {
        self.take(8)?
            .try_into()
            .map(u64::from_le_bytes)
            .map_err(|_| TierError::InvalidManifest("invalid u64".into()))
    }

    fn digest(&mut self) -> Result<Digest, TierError> {
        self.take(32)?
            .try_into()
            .map_err(|_| TierError::InvalidManifest("invalid digest".into()))
    }

    fn text(&mut self) -> Result<String, TierError> {
        let len = self.u64()? as usize;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| TierError::InvalidManifest("invalid text".into()))
    }

    fn finished(&self) -> bool {
        self.at == self.bytes.len()
    }
}
