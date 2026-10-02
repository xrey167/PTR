use aes_gcm_siv::{
    aead::{Aead, KeyInit, Payload},
    Aes256GcmSiv, Nonce,
};
use ptr_types::{ArtifactId, Digest, Generation, KeyId, Revision};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

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
        let bytes = encode_store(&self.inner);
        let temp = self.path.with_extension("ptr.tmp");
        fs::write(&temp, bytes).map_err(|_| ProtectedStateError::Io)?;
        match fs::rename(&temp, &self.path) {
            Ok(()) => Ok(()),
            Err(_) if self.path.exists() => {
                fs::remove_file(&self.path).map_err(|_| ProtectedStateError::Io)?;
                fs::rename(&temp, &self.path).map_err(|_| ProtectedStateError::Io)
            }
            Err(_) => Err(ProtectedStateError::Io),
        }
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

const STORE_MAGIC: &[u8; 8] = b"PTRPST01";

fn encode_store<K: KeyProvider>(store: &InMemoryProtectedStateStore<K>) -> Vec<u8> {
    let mut out = STORE_MAGIC.to_vec();
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
    if cursor.take(8)? != STORE_MAGIC {
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
        let anchor_digest = cursor.digest()?;
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
                anchor_digest,
                key_id,
            },
            anchor,
            aad: cursor.bytes_vec()?,
            nonce: cursor.fixed::<12>()?,
            ciphertext: cursor.bytes_vec()?,
            plaintext_digest: cursor.digest()?,
        };
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
    let mut value = Vec::new();
    value.extend_from_slice(b"ptr-protected-state-v1");
    value.extend_from_slice(record.domain.tag());
    value.extend_from_slice(&(record.logical_id.len() as u64).to_le_bytes());
    value.extend_from_slice(record.logical_id.as_bytes());
    value.extend_from_slice(&record.generation.0.to_le_bytes());
    value.extend_from_slice(&record.revision.0.to_le_bytes());
    value.extend_from_slice(record.key_id.0.as_bytes());
    value.extend_from_slice(&record.plaintext_digest);
    value
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
