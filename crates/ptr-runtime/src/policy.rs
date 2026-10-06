use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use ptr_types::{CapabilityId, Digest, PodId, ProjectId, Revision, Timestamp, TypeId};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "oidc-http")]
use std::io::Read;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ProjectPolicy {
    pub project: ProjectId,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CapabilityPolicy {
    pub project: ProjectId,
    pub capability: CapabilityId,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PodPolicyBinding {
    pub project: ProjectId,
    pub pod: PodId,
    pub capability: CapabilityId,
    pub input_type: TypeId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicyBundlePayload {
    pub projects: Vec<ProjectPolicy>,
    pub capabilities: Vec<CapabilityPolicy>,
    pub input_types: Vec<TypeId>,
    pub pod_bindings: Vec<PodPolicyBinding>,
    pub valid_from: Timestamp,
    pub valid_until: Option<Timestamp>,
}

impl PolicyBundlePayload {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut projects = self.projects.clone();
        projects.sort();
        projects.dedup();
        let mut capabilities = self.capabilities.clone();
        capabilities.sort();
        capabilities.dedup();
        let mut input_types = self.input_types.clone();
        input_types.sort();
        input_types.dedup();
        let mut bindings = self.pod_bindings.clone();
        bindings.sort();
        bindings.dedup();

        let mut out = b"PTR-POLICY-PAYLOAD-V1\0".to_vec();
        put_u64(&mut out, self.valid_from.0);
        match self.valid_until {
            Some(value) => {
                out.push(1);
                put_u64(&mut out, value.0);
            }
            None => out.push(0),
        }
        put_u32(&mut out, projects.len() as u32);
        for item in projects {
            put_string(&mut out, &item.project.0);
        }
        put_u32(&mut out, capabilities.len() as u32);
        for item in capabilities {
            put_string(&mut out, &item.project.0);
            put_string(&mut out, &item.capability.0);
        }
        put_u32(&mut out, input_types.len() as u32);
        for item in input_types {
            put_string(&mut out, &item.0);
        }
        put_u32(&mut out, bindings.len() as u32);
        for item in bindings {
            put_string(&mut out, &item.project.0);
            put_string(&mut out, &item.pod.0);
            put_string(&mut out, &item.capability.0);
            put_string(&mut out, &item.input_type.0);
        }
        out
    }

    pub fn digest(&self) -> Digest {
        Sha256::digest(self.canonical_bytes()).into()
    }

    /// Decode the canonical payload encoding.
    fn decode(bytes: &[u8]) -> Result<Self, PolicyError> {
        let mut payload_reader = Reader { bytes, offset: 0 };
        if payload_reader.take(b"PTR-POLICY-PAYLOAD-V1\0".len())? != b"PTR-POLICY-PAYLOAD-V1\0" {
            return Err(PolicyError::InvalidEncoding);
        }
        let valid_from = Timestamp(payload_reader.u64()?);
        let valid_until = match payload_reader.byte()? {
            0 => None,
            1 => Some(Timestamp(payload_reader.u64()?)),
            _ => return Err(PolicyError::InvalidEncoding),
        };
        let projects = (0..payload_reader.count()?)
            .map(|_| {
                Ok(ProjectPolicy {
                    project: ProjectId(payload_reader.string()?),
                })
            })
            .collect::<Result<Vec<_>, PolicyError>>()?;
        let capabilities = (0..payload_reader.count()?)
            .map(|_| {
                Ok(CapabilityPolicy {
                    project: ProjectId(payload_reader.string()?),
                    capability: CapabilityId(payload_reader.string()?),
                })
            })
            .collect::<Result<Vec<_>, PolicyError>>()?;
        let input_types = (0..payload_reader.count()?)
            .map(|_| Ok(TypeId(payload_reader.string()?)))
            .collect::<Result<Vec<_>, PolicyError>>()?;
        let pod_bindings = (0..payload_reader.count()?)
            .map(|_| {
                Ok(PodPolicyBinding {
                    project: ProjectId(payload_reader.string()?),
                    pod: PodId(payload_reader.string()?),
                    capability: CapabilityId(payload_reader.string()?),
                    input_type: TypeId(payload_reader.string()?),
                })
            })
            .collect::<Result<Vec<_>, PolicyError>>()?;
        if !payload_reader.finished() {
            return Err(PolicyError::InvalidEncoding);
        }
        Ok(PolicyBundlePayload {
            projects,
            capabilities,
            input_types,
            pod_bindings,
            valid_from,
            valid_until,
        })
    }

    fn validate(&self) -> Result<(), PolicyError> {
        if let Some(until) = self.valid_until {
            if until < self.valid_from {
                return Err(PolicyError::InvalidValidityWindow);
            }
        }
        if self.projects.iter().any(|p| p.project.0.is_empty())
            || self
                .capabilities
                .iter()
                .any(|p| p.project.0.is_empty() || p.capability.0.is_empty())
            || self.input_types.iter().any(|p| p.0.is_empty())
            || self.pod_bindings.iter().any(|p| {
                p.project.0.is_empty()
                    || p.pod.0.is_empty()
                    || p.capability.0.is_empty()
                    || p.input_type.0.is_empty()
            })
        {
            return Err(PolicyError::InvalidPayload);
        }
        // Everything that verifies is committed to the ledger and decoded again on
        // replay, so a payload the decoder would refuse must be refused here, before
        // it is signed into a record that can no longer be read.
        let encoded = self.canonical_bytes();
        if encoded.len() > MAX_PAYLOAD_BYTES {
            return Err(PolicyError::PayloadTooLarge);
        }
        Self::decode(&encoded).map_err(|_| PolicyError::InvalidPayload)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedPolicyBundle {
    pub revision: Revision,
    pub key_id: String,
    pub payload: PolicyBundlePayload,
    pub payload_digest: Digest,
    pub signature: Vec<u8>,
}

pub trait PolicyBundleLoader: Send + Sync {
    fn load(&self) -> Result<SignedPolicyBundle, PolicyError>;
}

pub type PolicyFetchFn = std::sync::Arc<dyn Fn(&str) -> Result<Vec<u8>, PolicyError> + Send + Sync>;

pub struct HttpPolicyBundleLoader {
    url: String,
    fetch: PolicyFetchFn,
}

impl HttpPolicyBundleLoader {
    pub fn new(url: impl Into<String>, fetch: PolicyFetchFn) -> Result<Self, PolicyError> {
        let url = url.into();
        if !(url.starts_with("https://")
            || url.starts_with("http://127.0.0.1/")
            || url.starts_with("http://localhost/"))
        {
            return Err(PolicyError::InvalidUrl);
        }
        Ok(Self { url, fetch })
    }
}

impl PolicyBundleLoader for HttpPolicyBundleLoader {
    fn load(&self) -> Result<SignedPolicyBundle, PolicyError> {
        const MAX_POLICY_BYTES: usize = 4 * 1024 * 1024;
        let bytes = (self.fetch)(&self.url)?;
        if bytes.len() > MAX_POLICY_BYTES {
            return Err(PolicyError::PayloadTooLarge);
        }
        SignedPolicyBundle::decode_for_ledger(&bytes)
    }
}

#[cfg(feature = "oidc-http")]
pub struct ReqwestPolicyBundleLoader {
    url: String,
    client: reqwest::blocking::Client,
}

#[cfg(feature = "oidc-http")]
impl ReqwestPolicyBundleLoader {
    pub fn new(url: impl Into<String>) -> Result<Self, PolicyError> {
        let url = url.into();
        if !url.starts_with("https://") {
            return Err(PolicyError::InvalidUrl);
        }
        let client = crate::identity::hardened_client_builder()
            .build()
            .map_err(|error| PolicyError::FetchFailed(error.to_string()))?;
        Ok(Self { url, client })
    }
}

#[cfg(feature = "oidc-http")]
impl PolicyBundleLoader for ReqwestPolicyBundleLoader {
    fn load(&self) -> Result<SignedPolicyBundle, PolicyError> {
        const MAX_POLICY_BYTES: usize = 4 * 1024 * 1024;
        let response = self
            .client
            .get(&self.url)
            .send()
            .map_err(|error| PolicyError::FetchFailed(error.to_string()))?;
        if !response.status().is_success() {
            return Err(PolicyError::FetchFailed(format!(
                "policy endpoint returned {}",
                response.status()
            )));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_POLICY_BYTES as u64)
        {
            return Err(PolicyError::PayloadTooLarge);
        }
        let mut bytes = Vec::new();
        response
            .take((MAX_POLICY_BYTES as u64).saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|error| PolicyError::FetchFailed(error.to_string()))?;
        if bytes.len() > MAX_POLICY_BYTES {
            return Err(PolicyError::PayloadTooLarge);
        }
        SignedPolicyBundle::decode_for_ledger(&bytes)
    }
}

impl SignedPolicyBundle {
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut out = b"PTR-SIGNED-POLICY-V1\0".to_vec();
        put_u64(&mut out, self.revision.0);
        put_string(&mut out, &self.key_id);
        out.extend_from_slice(&self.payload_digest);
        out
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = self.signing_bytes();
        put_bytes(&mut out, &self.payload.canonical_bytes());
        put_bytes(&mut out, &self.signature);
        out
    }

    pub fn encode_for_ledger(&self) -> Vec<u8> {
        self.canonical_bytes()
    }

    pub fn decode_for_ledger(bytes: &[u8]) -> Result<Self, PolicyError> {
        let mut reader = Reader { bytes, offset: 0 };
        if reader.take(b"PTR-SIGNED-POLICY-V1\0".len())? != b"PTR-SIGNED-POLICY-V1\0" {
            return Err(PolicyError::InvalidEncoding);
        }
        let revision = Revision(reader.u64()?);
        let key_id = reader.string()?;
        let payload_digest = reader.digest()?;
        let payload_bytes = reader.bytes(MAX_PAYLOAD_BYTES)?;
        let signature = reader.bytes(MAX_SIGNATURE_BYTES)?;
        if !reader.finished() {
            return Err(PolicyError::InvalidEncoding);
        }
        let payload = PolicyBundlePayload::decode(&payload_bytes)?;
        Ok(Self {
            revision,
            key_id,
            payload,
            payload_digest,
            signature,
        })
    }

    pub fn verify(&self, trust: &PolicyTrustStore, now: Timestamp) -> Result<(), PolicyError> {
        // The key id is read back as an identifier, so it is bounded like one; the
        // payload is bounded by `validate` below.
        if self.key_id.is_empty()
            || self.key_id.len() > MAX_STRING_BYTES
            || self.signature.len() != 64
            || self.revision.0 == 0
        {
            return Err(PolicyError::InvalidBundle);
        }
        self.payload.validate()?;
        let actual = self.payload.digest();
        if actual != self.payload_digest {
            return Err(PolicyError::PayloadDigestMismatch);
        }
        if now < self.payload.valid_from
            || self.payload.valid_until.is_some_and(|until| now > until)
        {
            return Err(PolicyError::OutsideValidityWindow);
        }
        let key = trust
            .keys
            .get(&self.key_id)
            .ok_or_else(|| PolicyError::UnknownKey(self.key_id.clone()))?;
        if trust.revoked.contains(&self.key_id) {
            return Err(PolicyError::RevokedKey(self.key_id.clone()));
        }
        let signature = Signature::from_bytes(
            self.signature
                .as_slice()
                .try_into()
                .map_err(|_| PolicyError::InvalidSignature)?,
        );
        key.verify(&self.signing_bytes(), &signature)
            .map_err(|_| PolicyError::InvalidSignature)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PolicyError {
    InvalidBundle,
    InvalidPayload,
    InvalidValidityWindow,
    OutsideValidityWindow,
    PayloadDigestMismatch,
    InvalidSignature,
    UnknownKey(String),
    RevokedKey(String),
    ConflictingRevision(Revision),
    NonMonotoneRevision {
        current: Revision,
        requested: Revision,
    },
    RevokedRevision(Revision),
    InvalidEncoding,
    InvalidUrl,
    PayloadTooLarge,
    FetchFailed(String),
}

#[derive(Clone, Default)]
pub struct PolicyTrustStore {
    keys: BTreeMap<String, VerifyingKey>,
    revoked: BTreeSet<String>,
}

impl PolicyTrustStore {
    pub fn insert(&mut self, key_id: impl Into<String>, key: VerifyingKey) {
        self.keys.insert(key_id.into(), key);
    }

    pub fn revoke(&mut self, key_id: impl Into<String>) {
        self.revoked.insert(key_id.into());
    }

    pub fn contains(&self, key_id: &str) -> bool {
        self.keys.contains_key(key_id) && !self.revoked.contains(key_id)
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value);
}

/// Most items one list of a policy payload may carry.
const MAX_ITEMS: usize = 4096;
/// Longest identifier string in a policy payload, in bytes.
const MAX_STRING_BYTES: usize = 4096;
/// Longest signature field in the ledger encoding, in bytes.
const MAX_SIGNATURE_BYTES: usize = 4096;
/// Longest encoded payload, in bytes. Equal to what the loaders fetch, so a
/// bundle that was fetchable is also decodable.
const MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], PolicyError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(PolicyError::InvalidEncoding)?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(PolicyError::InvalidEncoding)?;
        self.offset = end;
        Ok(result)
    }

    fn byte(&mut self) -> Result<u8, PolicyError> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64, PolicyError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| PolicyError::InvalidEncoding)?,
        ))
    }

    fn count(&mut self) -> Result<usize, PolicyError> {
        let count = u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| PolicyError::InvalidEncoding)?,
        ) as usize;
        if count > MAX_ITEMS {
            return Err(PolicyError::InvalidEncoding);
        }
        Ok(count)
    }

    /// A length prefix, bounded by `limit`. Item counts and byte lengths are
    /// different quantities with different bounds, so they do not share `count`.
    fn length(&mut self, limit: usize) -> Result<usize, PolicyError> {
        let length = u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| PolicyError::InvalidEncoding)?,
        ) as usize;
        if length > limit {
            return Err(PolicyError::InvalidEncoding);
        }
        Ok(length)
    }

    fn string(&mut self) -> Result<String, PolicyError> {
        let len = self.length(MAX_STRING_BYTES)?;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| PolicyError::InvalidEncoding)
    }

    fn bytes(&mut self, limit: usize) -> Result<Vec<u8>, PolicyError> {
        let len = self.length(limit)?;
        Ok(self.take(len)?.to_vec())
    }

    fn digest(&mut self) -> Result<Digest, PolicyError> {
        self.take(32)?
            .try_into()
            .map_err(|_| PolicyError::InvalidEncoding)
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn bundle(key: &SigningKey) -> SignedPolicyBundle {
        let payload = PolicyBundlePayload {
            projects: vec![ProjectPolicy {
                project: ProjectId::from("p"),
            }],
            capabilities: vec![CapabilityPolicy {
                project: ProjectId::from("p"),
                capability: CapabilityId::from("read"),
            }],
            input_types: vec![TypeId::from("input")],
            pod_bindings: vec![PodPolicyBinding {
                project: ProjectId::from("p"),
                pod: PodId::from("pod"),
                capability: CapabilityId::from("read"),
                input_type: TypeId::from("input"),
            }],
            valid_from: Timestamp(1),
            valid_until: Some(Timestamp(10)),
        };
        let mut result = SignedPolicyBundle {
            revision: Revision(1),
            key_id: "root".into(),
            payload,
            payload_digest: [0; 32],
            signature: Vec::new(),
        };
        result.payload_digest = result.payload.digest();
        result.signature = key.sign(&result.signing_bytes()).to_bytes().to_vec();
        result
    }

    #[test]
    fn signed_bundle_verifies_and_binds_payload() {
        let signing = SigningKey::from_bytes(&[7; 32]);
        let mut trust = PolicyTrustStore::default();
        trust.insert("root", signing.verifying_key());
        assert!(bundle(&signing).verify(&trust, Timestamp(5)).is_ok());
        let mut changed = bundle(&signing);
        changed.payload.projects.push(ProjectPolicy {
            project: ProjectId::from("q"),
        });
        assert_eq!(
            changed.verify(&trust, Timestamp(5)),
            Err(PolicyError::PayloadDigestMismatch)
        );
    }

    fn resigned(mut signed: SignedPolicyBundle, key: &SigningKey) -> SignedPolicyBundle {
        signed.payload_digest = signed.payload.digest();
        signed.signature = key.sign(&signed.signing_bytes()).to_bytes().to_vec();
        signed
    }

    #[test]
    fn a_payload_longer_than_the_item_count_cap_round_trips_through_the_ledger_encoding() {
        let signing = SigningKey::from_bytes(&[10; 32]);
        let mut signed = bundle(&signing);
        // 300 projects of 12 bytes each: well under the item cap of 4096, but the
        // encoded payload is several times longer than 4096 bytes.
        signed.payload.projects = (0..300)
            .map(|index| ProjectPolicy {
                project: ProjectId(format!("project-{index:04}")),
            })
            .collect();
        let signed = resigned(signed, &signing);
        let mut trust = PolicyTrustStore::default();
        trust.insert("root", signing.verifying_key());
        signed.verify(&trust, Timestamp(5)).unwrap();
        assert!(signed.payload.canonical_bytes().len() > 4096);

        let decoded = SignedPolicyBundle::decode_for_ledger(&signed.encode_for_ledger())
            .expect("a signed bundle the runtime committed must decode again");
        assert_eq!(decoded, signed);
        decoded.verify(&trust, Timestamp(5)).unwrap();
    }

    #[test]
    fn a_payload_the_decoder_would_refuse_does_not_verify() {
        let signing = SigningKey::from_bytes(&[11; 32]);
        let mut trust = PolicyTrustStore::default();
        trust.insert("root", signing.verifying_key());

        // One item more than a list may carry.
        let mut too_many = bundle(&signing);
        too_many.payload.projects = (0..4097)
            .map(|index| ProjectPolicy {
                project: ProjectId(format!("p{index}")),
            })
            .collect();
        let too_many = resigned(too_many, &signing);
        assert_eq!(
            SignedPolicyBundle::decode_for_ledger(&too_many.encode_for_ledger()),
            Err(PolicyError::InvalidEncoding)
        );
        assert_eq!(
            too_many.verify(&trust, Timestamp(5)),
            Err(PolicyError::InvalidPayload)
        );

        // An identifier one byte longer than the bound.
        let mut long_name = bundle(&signing);
        long_name.payload.projects = vec![ProjectPolicy {
            project: ProjectId("x".repeat(MAX_STRING_BYTES + 1)),
        }];
        let long_name = resigned(long_name, &signing);
        assert_eq!(
            long_name.verify(&trust, Timestamp(5)),
            Err(PolicyError::InvalidPayload)
        );

        // Within every per-item bound, yet longer in total than the payload bound.
        let mut too_long = bundle(&signing);
        too_long.payload.projects = (0..1100)
            .map(|index| ProjectPolicy {
                project: ProjectId(format!("{index:04}{}", "y".repeat(MAX_STRING_BYTES - 4))),
            })
            .collect();
        let too_long = resigned(too_long, &signing);
        assert_eq!(
            too_long.verify(&trust, Timestamp(5)),
            Err(PolicyError::PayloadTooLarge)
        );

        // Control: the same shapes at the bound still verify and round-trip.
        let mut at_bound = bundle(&signing);
        // Zero-padded so the list is already in canonical (sorted) order, which is
        // the order decoding returns.
        at_bound.payload.projects = (0..4096)
            .map(|index| ProjectPolicy {
                project: ProjectId(format!("p{index:04}")),
            })
            .collect();
        let at_bound = resigned(at_bound, &signing);
        at_bound.verify(&trust, Timestamp(5)).unwrap();
        assert_eq!(
            SignedPolicyBundle::decode_for_ledger(&at_bound.encode_for_ledger()).unwrap(),
            at_bound
        );
    }

    #[test]
    fn a_key_id_longer_than_the_identifier_bound_does_not_verify() {
        let signing = SigningKey::from_bytes(&[12; 32]);
        // A host can install a trust key under any id, so the id has to be bounded
        // by verify as well: the ledger encoding reads it back as an identifier.
        let long_id = "k".repeat(MAX_STRING_BYTES + 1);
        let mut trust = PolicyTrustStore::default();
        trust.insert(long_id.clone(), signing.verifying_key());
        let mut signed = bundle(&signing);
        signed.key_id = long_id;
        let signed = resigned(signed, &signing);
        assert_eq!(
            SignedPolicyBundle::decode_for_ledger(&signed.encode_for_ledger()),
            Err(PolicyError::InvalidEncoding)
        );
        assert_eq!(
            signed.verify(&trust, Timestamp(5)),
            Err(PolicyError::InvalidBundle)
        );

        // Control: an id at the bound verifies and round-trips.
        let at_bound = "k".repeat(MAX_STRING_BYTES);
        let mut trust = PolicyTrustStore::default();
        trust.insert(at_bound.clone(), signing.verifying_key());
        let mut signed = bundle(&signing);
        signed.key_id = at_bound;
        let signed = resigned(signed, &signing);
        signed.verify(&trust, Timestamp(5)).unwrap();
        assert_eq!(
            SignedPolicyBundle::decode_for_ledger(&signed.encode_for_ledger()).unwrap(),
            signed
        );
    }

    #[test]
    fn validity_is_fail_closed() {
        let signing = SigningKey::from_bytes(&[8; 32]);
        let mut trust = PolicyTrustStore::default();
        trust.insert("root", signing.verifying_key());
        assert_eq!(
            bundle(&signing).verify(&trust, Timestamp(11)),
            Err(PolicyError::OutsideValidityWindow)
        );
    }

    #[test]
    fn policy_loader_decodes_but_does_not_bypass_signature_admission() {
        let signing = SigningKey::from_bytes(&[9; 32]);
        let bundle = bundle(&signing);
        let bytes = bundle.encode_for_ledger();
        let loader = HttpPolicyBundleLoader::new(
            "https://policy.example/bundle",
            std::sync::Arc::new(move |_| Ok(bytes.clone())),
        )
        .unwrap();
        let loaded = loader.load().unwrap();
        let mut trust = PolicyTrustStore::default();
        trust.insert("root", signing.verifying_key());
        assert!(loaded.verify(&trust, Timestamp(5)).is_ok());
    }

    #[test]
    fn policy_loader_rejects_non_https_or_local_test_urls() {
        assert!(HttpPolicyBundleLoader::new(
            "http://policy.example/bundle",
            std::sync::Arc::new(|_| Ok(Vec::new()))
        )
        .is_err());
    }
}

#[cfg(all(test, feature = "oidc-http"))]
mod oidc_http_tests {
    use super::*;

    #[test]
    fn reqwest_policy_loader_builds_without_a_network_or_a_panic() {
        ReqwestPolicyBundleLoader::new("https://example.invalid/policy")
            .expect("an HTTPS policy client must be constructible");
        assert!(matches!(
            ReqwestPolicyBundleLoader::new("http://example.invalid/policy"),
            Err(PolicyError::InvalidUrl)
        ));
    }
}
