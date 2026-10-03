use aes_gcm_siv::{
    aead::{Aead, KeyInit, Payload},
    Aes256GcmSiv, Nonce,
};
use ptr_types::{ArtifactId, Digest, Generation, KeyId, Revision};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use zeroize::Zeroizing;

#[cfg(feature = "tier-opendal")]
mod opendal_tier;
mod tier;
mod tier_protected;
#[cfg(feature = "tier-opendal")]
pub use opendal_tier::OpenDalTierBackend;
pub use tier::{
    digest as tier_digest, ChunkDescriptor, ChunkReceipt, CpuTierBackend, FileTierBackend,
    PreparedTierObject, StorageTier, TierBackend, TierBackendId, TierCapabilities, TierError,
    TierFuture, TierHealth, TierIntegrityReport, TierObjectDomain, TierObjectManifest,
};
pub use tier_protected::{
    open_protected_tier_chunk, seal_protected_tier_chunk, ProtectedTierChunk,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum StateDomain {
    Ledger,
    Knowledge,
    Artifact,
    KvSnapshot,
    RuntimeJournal,
}

impl StateDomain {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::Ledger => b"ledger",
            Self::Knowledge => b"knowledge",
            Self::Artifact => b"artifact",
            Self::KvSnapshot => b"kv-snapshot",
            Self::RuntimeJournal => b"runtime-journal",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationAnchor {
    pub logical_id: String,
    pub generation: Generation,
    pub previous_digest: Option<Digest>,
    pub anchor_digest: Digest,
}

impl GenerationAnchor {
    pub fn new(
        logical_id: impl Into<String>,
        generation: Generation,
        previous_digest: Option<Digest>,
    ) -> Self {
        let mut anchor = Self {
            logical_id: logical_id.into(),
            generation,
            previous_digest,
            anchor_digest: [0; 32],
        };
        anchor.anchor_digest = anchor_digest(&anchor);
        anchor
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtectedRecord {
    pub domain: StateDomain,
    pub logical_id: String,
    pub generation: Generation,
    pub revision: Revision,
    pub plaintext_digest: Digest,
    pub payload: Vec<u8>,
    pub key_id: KeyId,
    pub anchor: GenerationAnchor,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ProtectedHandle {
    pub domain: StateDomain,
    pub logical_id: String,
    pub generation: Generation,
    pub revision: Revision,
    pub ciphertext_digest: Digest,
    pub anchor_digest: Digest,
    pub key_id: KeyId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntegrityReport {
    pub plaintext_digest: Digest,
    pub ciphertext_digest: Digest,
    pub generation: Generation,
    pub valid: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProtectedStateError {
    UnknownHandle,
    UnknownKey(KeyId),
    RevokedKey(KeyId),
    PlaintextDigestMismatch,
    CiphertextDigestMismatch,
    AnchorMismatch,
    Rollback,
    InvalidKeyId,
    Crypto,
    Io,
}

pub type ZeroizingKey = Zeroizing<[u8; 32]>;

pub trait KeyProvider: Send + Sync {
    fn unwrap_key(&self, key_id: &KeyId) -> Result<ZeroizingKey, ProtectedStateError>;
    fn rotate(&mut self, key_id: KeyId, key: [u8; 32]) -> Result<(), ProtectedStateError>;
    fn revoke(&mut self, key_id: &KeyId) -> Result<(), ProtectedStateError>;
}

pub trait ProtectedStateStore: Send + Sync {
    fn seal(&mut self, record: ProtectedRecord) -> Result<ProtectedHandle, ProtectedStateError>;
    fn open(&self, handle: &ProtectedHandle) -> Result<Vec<u8>, ProtectedStateError>;
    fn verify(&self, handle: &ProtectedHandle) -> Result<IntegrityReport, ProtectedStateError>;
    fn rotate_key(&mut self, key_id: KeyId, key: [u8; 32]) -> Result<(), ProtectedStateError>;
}

#[derive(Default)]
pub struct SoftwareKeyProvider {
    keys: BTreeMap<KeyId, ZeroizingKey>,
    revoked: std::collections::BTreeSet<KeyId>,
}

impl SoftwareKeyProvider {
    pub fn with_key(mut self, key_id: KeyId, key: [u8; 32]) -> Self {
        self.keys.insert(key_id, Zeroizing::new(key));
        self
    }
}

impl KeyProvider for SoftwareKeyProvider {
    fn unwrap_key(&self, key_id: &KeyId) -> Result<ZeroizingKey, ProtectedStateError> {
        if self.revoked.contains(key_id) {
            return Err(ProtectedStateError::RevokedKey(key_id.clone()));
        }
        self.keys
            .get(key_id)
            .cloned()
            .ok_or_else(|| ProtectedStateError::UnknownKey(key_id.clone()))
    }

    fn rotate(&mut self, key_id: KeyId, key: [u8; 32]) -> Result<(), ProtectedStateError> {
        self.revoked.remove(&key_id);
        self.keys.insert(key_id, Zeroizing::new(key));
        Ok(())
    }

    fn revoke(&mut self, key_id: &KeyId) -> Result<(), ProtectedStateError> {
        if !self.keys.contains_key(key_id) {
            return Err(ProtectedStateError::UnknownKey(key_id.clone()));
        }
        self.revoked.insert(key_id.clone());
        Ok(())
    }
}

struct SealedRecord {
    handle: ProtectedHandle,
    anchor: GenerationAnchor,
    aad: Vec<u8>,
    nonce: [u8; 12],
    ciphertext: Vec<u8>,
    plaintext_digest: Digest,
}

pub struct InMemoryProtectedStateStore<K: KeyProvider> {
    keys: K,
    records: BTreeMap<Digest, SealedRecord>,
    anchors: BTreeMap<(StateDomain, String), GenerationAnchor>,
}

impl<K: KeyProvider> InMemoryProtectedStateStore<K> {
    pub fn new(keys: K) -> Self {
        Self {
            keys,
            records: BTreeMap::new(),
            anchors: BTreeMap::new(),
        }
    }

    pub fn keys_mut(&mut self) -> &mut K {
        &mut self.keys
    }

    pub fn revoke_key(&mut self, key_id: &KeyId) -> Result<(), ProtectedStateError> {
        self.keys.revoke(key_id)
    }
}

impl<K: KeyProvider> ProtectedStateStore for InMemoryProtectedStateStore<K> {
    fn seal(&mut self, record: ProtectedRecord) -> Result<ProtectedHandle, ProtectedStateError> {
        if record.logical_id.is_empty() || record.key_id.0.is_empty() {
            return Err(ProtectedStateError::InvalidKeyId);
        }
        let plaintext_digest = digest(&record.payload);
        if plaintext_digest != record.plaintext_digest {
            return Err(ProtectedStateError::PlaintextDigestMismatch);
        }
        if record.anchor.logical_id != record.logical_id
            || record.anchor.generation != record.generation
            || record.anchor.anchor_digest != anchor_digest(&record.anchor)
        {
            return Err(ProtectedStateError::AnchorMismatch);
        }
        let anchor_key = (record.domain, record.logical_id.clone());
        let key = self.keys.unwrap_key(&record.key_id)?;
        let aad = aad(&record);
        let nonce = nonce(&aad);
        let cipher =
            Aes256GcmSiv::new_from_slice(key.as_ref()).map_err(|_| ProtectedStateError::Crypto)?;
        let ciphertext = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &record.payload,
                    aad: &aad,
                },
            )
            .map_err(|_| ProtectedStateError::Crypto)?;
        let ciphertext_digest = digest(&ciphertext);
        let handle = ProtectedHandle {
            domain: record.domain,
            logical_id: record.logical_id.clone(),
            generation: record.generation,
            revision: record.revision,
            ciphertext_digest,
            anchor_digest: record.anchor.anchor_digest,
            key_id: record.key_id.clone(),
        };
        if let Some(existing) = self.records.get(&ciphertext_digest) {
            if existing.handle == handle && existing.anchor == record.anchor {
                return Ok(handle);
            }
            return Err(ProtectedStateError::CiphertextDigestMismatch);
        }
        if let Some(previous) = self.anchors.get(&anchor_key) {
            if record.generation.0 <= previous.generation.0
                || record.anchor.previous_digest != Some(previous.anchor_digest)
            {
                return Err(ProtectedStateError::Rollback);
            }
        } else if record.anchor.previous_digest.is_some() {
            return Err(ProtectedStateError::AnchorMismatch);
        }
        self.records.insert(
            ciphertext_digest,
            SealedRecord {
                handle: handle.clone(),
                anchor: record.anchor.clone(),
                aad,
                nonce,
                ciphertext,
                plaintext_digest,
            },
        );
        self.anchors.insert(anchor_key, record.anchor);
        Ok(handle)
    }

    fn open(&self, handle: &ProtectedHandle) -> Result<Vec<u8>, ProtectedStateError> {
        let sealed = self
            .records
            .get(&handle.ciphertext_digest)
            .ok_or(ProtectedStateError::UnknownHandle)?;
        if &sealed.handle != handle || digest(&sealed.ciphertext) != handle.ciphertext_digest {
            return Err(ProtectedStateError::CiphertextDigestMismatch);
        }
        if sealed.anchor.anchor_digest != handle.anchor_digest {
            return Err(ProtectedStateError::AnchorMismatch);
        }
        let key = self.keys.unwrap_key(&handle.key_id)?;
        let cipher =
            Aes256GcmSiv::new_from_slice(key.as_ref()).map_err(|_| ProtectedStateError::Crypto)?;
        let plaintext = cipher
            .decrypt(
                Nonce::from_slice(&sealed.nonce),
                Payload {
                    msg: &sealed.ciphertext,
                    aad: &sealed.aad,
                },
            )
            .map_err(|_| ProtectedStateError::Crypto)?;
        if digest(&plaintext) != sealed.plaintext_digest {
            return Err(ProtectedStateError::PlaintextDigestMismatch);
        }
        Ok(plaintext)
    }

    fn verify(&self, handle: &ProtectedHandle) -> Result<IntegrityReport, ProtectedStateError> {
        let plaintext = self.open(handle)?;
        Ok(IntegrityReport {
            plaintext_digest: digest(&plaintext),
            ciphertext_digest: handle.ciphertext_digest,
            generation: handle.generation,
            valid: true,
        })
    }

    fn rotate_key(&mut self, key_id: KeyId, key: [u8; 32]) -> Result<(), ProtectedStateError> {
        self.keys.rotate(key_id, key)
    }
}

/// File-backed protected state. The payload remains encrypted at rest; the
/// supplied key provider is deliberately required again when reopening.
pub struct FileProtectedStateStore<K: KeyProvider> {
    path: PathBuf,
    inner: InMemoryProtectedStateStore<K>,
}

impl<K: KeyProvider> FileProtectedStateStore<K> {
    pub fn open(path: impl AsRef<Path>, keys: K) -> Result<Self, ProtectedStateError> {
        let path = path.as_ref().to_owned();
        let inner = if path.exists() {
            decode_store(&fs::read(&path).map_err(|_| ProtectedStateError::Io)?, keys)?
        } else {
            InMemoryProtectedStateStore::new(keys)
        };
        Ok(Self { path, inner })
    }

    fn persist(&self) -> Result<(), ProtectedStateError> {
        static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);
        let bytes = encode_store(&self.inner);
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let file_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or(ProtectedStateError::Io)?;
        let temp = self.path.with_file_name(format!(
            ".{file_name}.{}.{}.ptr.tmp",
            std::process::id(),
            suffix
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|_| ProtectedStateError::Io)?;
        if file.write_all(&bytes).is_err() || file.sync_all().is_err() {
            let _ = fs::remove_file(&temp);
            return Err(ProtectedStateError::Io);
        }
        drop(file);
        if atomic_replace(&temp, &self.path).is_err() {
            let _ = fs::remove_file(&temp);
            return Err(ProtectedStateError::Io);
        }
        Ok(())
    }
}

#[cfg(not(windows))]
fn atomic_replace(temp: &Path, destination: &Path) -> std::io::Result<()> {
    fs::rename(temp, destination)
}

#[cfg(windows)]
fn atomic_replace(temp: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = temp
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let target = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            target.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

impl<K: KeyProvider> ProtectedStateStore for FileProtectedStateStore<K> {
    fn seal(&mut self, record: ProtectedRecord) -> Result<ProtectedHandle, ProtectedStateError> {
        let handle = self.inner.seal(record)?;
        self.persist()?;
        Ok(handle)
    }

    fn open(&self, handle: &ProtectedHandle) -> Result<Vec<u8>, ProtectedStateError> {
        self.inner.open(handle)
    }

    fn verify(&self, handle: &ProtectedHandle) -> Result<IntegrityReport, ProtectedStateError> {
        self.inner.verify(handle)
    }

    fn rotate_key(&mut self, key_id: KeyId, key: [u8; 32]) -> Result<(), ProtectedStateError> {
        self.inner.rotate_key(key_id, key)?;
        self.persist()
    }
}

const STORE_MAGIC_V1: &[u8; 8] = b"PTRPST01";
const STORE_MAGIC_V2: &[u8; 8] = b"PTRPST02";

fn encode_store<K: KeyProvider>(store: &InMemoryProtectedStateStore<K>) -> Vec<u8> {
    let mut out = STORE_MAGIC_V2.to_vec();
    put_u32(&mut out, store.records.len() as u32);
    for sealed in store.records.values() {
        put_domain(&mut out, sealed.handle.domain);
        put_text_bytes(&mut out, &sealed.handle.logical_id);
        put_u64(&mut out, sealed.handle.generation.0);
        put_u64(&mut out, sealed.handle.revision.0);
        out.extend_from_slice(&sealed.handle.ciphertext_digest);
        out.extend_from_slice(&sealed.handle.anchor_digest);
        put_text_bytes(&mut out, &sealed.handle.key_id.0);
        put_text_bytes(&mut out, &sealed.anchor.logical_id);
        put_u64(&mut out, sealed.anchor.generation.0);
        match sealed.anchor.previous_digest {
            Some(digest) => {
                out.push(1);
                out.extend_from_slice(&digest);
            }
            None => out.push(0),
        }
        out.extend_from_slice(&sealed.anchor.anchor_digest);
        put_bytes(&mut out, &sealed.aad);
        out.extend_from_slice(&sealed.nonce);
        put_bytes(&mut out, &sealed.ciphertext);
        out.extend_from_slice(&sealed.plaintext_digest);
    }
    out
}

fn decode_store<K: KeyProvider>(
    bytes: &[u8],
    keys: K,
) -> Result<InMemoryProtectedStateStore<K>, ProtectedStateError> {
    let mut cursor = Cursor { bytes, at: 0 };
    let magic = cursor.take(8)?;
    if magic != STORE_MAGIC_V1 && magic != STORE_MAGIC_V2 {
        return Err(ProtectedStateError::Io);
    }
    let count = cursor.u32()? as usize;
    let mut store = InMemoryProtectedStateStore::new(keys);
    for _ in 0..count {
        let domain = cursor.domain()?;
        let logical_id = cursor.text()?;
        let generation = Generation(cursor.u64()?);
        let revision = Revision(cursor.u64()?);
        let ciphertext_digest = cursor.digest()?;
        let handle_anchor_digest = cursor.digest()?;
        let key_id = KeyId(cursor.text()?);
        let anchor_logical_id = cursor.text()?;
        let anchor_generation = Generation(cursor.u64()?);
        let previous_digest = match cursor.byte()? {
            0 => None,
            1 => Some(cursor.digest()?),
            _ => return Err(ProtectedStateError::Io),
        };
        let anchor = GenerationAnchor {
            logical_id: anchor_logical_id,
            generation: anchor_generation,
            previous_digest,
            anchor_digest: cursor.digest()?,
        };
        let sealed = SealedRecord {
            handle: ProtectedHandle {
                domain,
                logical_id: logical_id.clone(),
                generation,
                revision,
                ciphertext_digest,
                anchor_digest: handle_anchor_digest,
                key_id,
            },
            anchor,
            aad: cursor.bytes_vec()?,
            nonce: cursor.fixed::<12>()?,
            ciphertext: cursor.bytes_vec()?,
            plaintext_digest: cursor.digest()?,
        };
        if sealed.anchor.anchor_digest != anchor_digest(&sealed.anchor)
            || sealed.handle.anchor_digest != sealed.anchor.anchor_digest
            || sealed.handle.logical_id != sealed.anchor.logical_id
            || sealed.handle.generation != sealed.anchor.generation
            || digest(&sealed.ciphertext) != sealed.handle.ciphertext_digest
            || !valid_stored_aad(&sealed)
        {
            return Err(ProtectedStateError::AnchorMismatch);
        }
        store
            .anchors
            .insert((domain, logical_id), sealed.anchor.clone());
        store.records.insert(ciphertext_digest, sealed);
    }
    if cursor.at != bytes.len() {
        return Err(ProtectedStateError::Io);
    }
    Ok(store)
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_text_bytes(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}
fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value);
}
fn put_domain(out: &mut Vec<u8>, domain: StateDomain) {
    out.push(match domain {
        StateDomain::Ledger => 0,
        StateDomain::Knowledge => 1,
        StateDomain::Artifact => 2,
        StateDomain::KvSnapshot => 3,
        StateDomain::RuntimeJournal => 4,
    });
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], ProtectedStateError> {
        let end = self.at.checked_add(len).ok_or(ProtectedStateError::Io)?;
        let value = self
            .bytes
            .get(self.at..end)
            .ok_or(ProtectedStateError::Io)?;
        self.at = end;
        Ok(value)
    }
    fn byte(&mut self) -> Result<u8, ProtectedStateError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, ProtectedStateError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| ProtectedStateError::Io)?,
        ))
    }
    fn u64(&mut self) -> Result<u64, ProtectedStateError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| ProtectedStateError::Io)?,
        ))
    }
    fn digest(&mut self) -> Result<Digest, ProtectedStateError> {
        self.take(32)?
            .try_into()
            .map_err(|_| ProtectedStateError::Io)
    }
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], ProtectedStateError> {
        self.take(N)?
            .try_into()
            .map_err(|_| ProtectedStateError::Io)
    }
    fn bytes_vec(&mut self) -> Result<Vec<u8>, ProtectedStateError> {
        let len = self.u32()? as usize;
        Ok(self.take(len)?.to_vec())
    }
    fn text(&mut self) -> Result<String, ProtectedStateError> {
        String::from_utf8(self.bytes_vec()?).map_err(|_| ProtectedStateError::Io)
    }
    fn domain(&mut self) -> Result<StateDomain, ProtectedStateError> {
        match self.byte()? {
            0 => Ok(StateDomain::Ledger),
            1 => Ok(StateDomain::Knowledge),
            2 => Ok(StateDomain::Artifact),
            3 => Ok(StateDomain::KvSnapshot),
            4 => Ok(StateDomain::RuntimeJournal),
            _ => Err(ProtectedStateError::Io),
        }
    }
}

fn digest(bytes: &[u8]) -> Digest {
    Sha256::digest(bytes).into()
}

fn aad(record: &ProtectedRecord) -> Vec<u8> {
    aad_fields(
        record.domain,
        &record.logical_id,
        record.generation,
        record.revision,
        &record.key_id,
        record.plaintext_digest,
        &record.anchor,
    )
}

fn aad_fields(
    domain: StateDomain,
    logical_id: &str,
    generation: Generation,
    revision: Revision,
    key_id: &KeyId,
    plaintext_digest: Digest,
    anchor: &GenerationAnchor,
) -> Vec<u8> {
    let mut value = Vec::new();
    value.extend_from_slice(b"ptr-protected-state-v2");
    value.extend_from_slice(domain.tag());
    value.extend_from_slice(&(logical_id.len() as u64).to_le_bytes());
    value.extend_from_slice(logical_id.as_bytes());
    value.extend_from_slice(&generation.0.to_le_bytes());
    value.extend_from_slice(&revision.0.to_le_bytes());
    value.extend_from_slice(&(key_id.0.len() as u64).to_le_bytes());
    value.extend_from_slice(key_id.0.as_bytes());
    value.extend_from_slice(&plaintext_digest);
    value.extend_from_slice(&(anchor.logical_id.len() as u64).to_le_bytes());
    value.extend_from_slice(anchor.logical_id.as_bytes());
    value.extend_from_slice(&anchor.generation.0.to_le_bytes());
    match anchor.previous_digest {
        Some(previous) => {
            value.push(1);
            value.extend_from_slice(&previous);
        }
        None => value.push(0),
    }
    value.extend_from_slice(&anchor.anchor_digest);
    value
}

fn legacy_aad_fields(
    domain: StateDomain,
    logical_id: &str,
    generation: Generation,
    revision: Revision,
    key_id: &KeyId,
    plaintext_digest: Digest,
) -> Vec<u8> {
    let mut value = Vec::new();
    value.extend_from_slice(b"ptr-protected-state-v1");
    value.extend_from_slice(domain.tag());
    value.extend_from_slice(&(logical_id.len() as u64).to_le_bytes());
    value.extend_from_slice(logical_id.as_bytes());
    value.extend_from_slice(&generation.0.to_le_bytes());
    value.extend_from_slice(&revision.0.to_le_bytes());
    value.extend_from_slice(key_id.0.as_bytes());
    value.extend_from_slice(&plaintext_digest);
    value
}

fn valid_stored_aad(sealed: &SealedRecord) -> bool {
    sealed.aad
        == aad_fields(
            sealed.handle.domain,
            &sealed.handle.logical_id,
            sealed.handle.generation,
            sealed.handle.revision,
            &sealed.handle.key_id,
            sealed.plaintext_digest,
            &sealed.anchor,
        )
        || sealed.aad
            == legacy_aad_fields(
                sealed.handle.domain,
                &sealed.handle.logical_id,
                sealed.handle.generation,
                sealed.handle.revision,
                &sealed.handle.key_id,
                sealed.plaintext_digest,
            )
}

fn nonce(aad: &[u8]) -> [u8; 12] {
    let digest = Sha256::digest(aad);
    digest[..12].try_into().expect("fixed nonce length")
}

fn anchor_digest(anchor: &GenerationAnchor) -> Digest {
    let mut value = Vec::new();
    value.extend_from_slice(anchor.logical_id.as_bytes());
    value.extend_from_slice(&anchor.generation.0.to_le_bytes());
    if let Some(previous) = anchor.previous_digest {
        value.extend_from_slice(&previous);
    }
    digest(&value)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRef {
    pub id: ArtifactId,
    pub locations: Vec<String>,
}

pub trait ObjectStore {
    fn put(&mut self, id: ArtifactId, bytes: Vec<u8>);
    fn get(&self, id: &ArtifactId) -> Option<&[u8]>;
}

#[derive(Default)]
pub struct InMemoryStore {
    values: BTreeMap<ArtifactId, Vec<u8>>,
}
impl ObjectStore for InMemoryStore {
    fn put(&mut self, id: ArtifactId, bytes: Vec<u8>) {
        self.values.insert(id, bytes);
    }
    fn get(&self, id: &ArtifactId) -> Option<&[u8]> {
        self.values.get(id).map(Vec::as_slice)
    }
}
