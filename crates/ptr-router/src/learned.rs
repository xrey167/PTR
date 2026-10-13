//! Learned pod addresses (Dragonfly): an advisory router over per-pod representations.
//!
//! Each pod owns a vector `z` and a bias, trained offline (`training/ptr_training/dragonfly`)
//! against a frozen text encoder. There is no trainable state shared between pods, so revoking
//! one pod invalidates exactly that pod's representation and leaves the others usable.
//!
//! # Advisory only
//!
//! A learned score never grants anything. [`LearnedPodRouter`] only ranks the candidates the
//! registry already offers (`PodRegistry::candidates`), so capability, input type and project
//! scope stay hard constraints; the trusted boundary ([`crate::route_checked`]) still refuses a
//! pod outside that set. A representation is loaded only when it is *activated* (every trusted
//! alias scored at least 0.8 and at least 90 % of the negative examples were rejected during
//! training); anything else is refused with [`RouteError::InactiveRepresentation`]. When the best
//! candidate scores below the abstention threshold, or the runner-up is too close, the router
//! abstains with [`RouteError::Abstained`] instead of guessing.
//!
//! # Scoring
//!
//! `logit = cosine_weight * cos(q, z) + metadata_weight * m - bias` and `score = sigmoid(logit)`.
//! `q` is [`RouteFeatures::vector`]. `m` is `1.0`: every candidate already matched the hard
//! metadata (project, capability, input type) in the registry, which is exactly the "metadata
//! always matches" counterfactual the representation was trained under, so metadata flags cannot
//! carry the score. The weights are part of the representation and of its digest.
//!
//! # Artifact
//!
//! [`PodRepresentation::from_artifact`] reads the pair `dragonfly-<identity>.json` and
//! `dragonfly-<identity>.bin` the Python exporter writes. The binary file holds `dims` little
//! endian `f32` values of `z` followed by one `f32` bias. The manifest carries the SHA-256 of
//! that file and a digest over every field (see [`PodRepresentation::digest`]); a mismatch of
//! either refuses the load.

use crate::{RouteDistribution, RouteError, RouteRequest, Router};
use ptr_types::{ClusterKey, Generation, PodId, PodIdentity};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

/// The artifact schema this build reads and writes.
pub const REPRESENTATION_SCHEMA: u32 = 1;
/// The most dimensions a representation may have.
pub const MAX_REPRESENTATION_DIMS: usize = 4096;
/// Minimum score of the weakest trusted alias for activation, in permille.
pub const ACTIVATION_ALIAS_PERMILLE: u16 = 800;
/// Minimum share of rejected negative examples for activation, in permille.
pub const ACTIVATION_NEGATIVE_REJECT_PERMILLE: u16 = 900;

const ARTIFACT_KIND: &str = "dragonfly-pod-address";
const DIGEST_DOMAIN: &str = "ptr.dragonfly.artifact.v1";
const STATUS_ACTIVATED: &str = "activated";
const STATUS_NOT_ACTIVATED: &str = "not-activated";

/// The frozen encoder a representation was trained against. A query vector is only comparable
/// with representations of the same encoder identity and revision.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EncoderRef {
    pub id: String,
    pub revision: String,
}

/// Fixed coefficients of the score; trained under, and bound to, one representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreWeights {
    pub cosine: f32,
    pub metadata: f32,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            cosine: 12.0,
            metadata: 3.0,
        }
    }
}

/// What the training-time activation gate measured, in permille.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GateReport {
    pub alias_threshold_permille: u16,
    /// The score of the weakest trusted alias.
    pub alias_min_permille: u16,
    pub negative_threshold_permille: u16,
    /// The share of negative training examples the representation rejected.
    pub negative_reject_permille: u16,
    pub alias_count: u32,
    pub negative_count: u32,
}

impl GateReport {
    /// Whether the report meets its own thresholds. A gate with no aliases or no negatives
    /// measured nothing and does not pass.
    pub fn passes(&self) -> bool {
        self.alias_count > 0
            && self.negative_count > 0
            && self.alias_min_permille >= self.alias_threshold_permille
            && self.negative_reject_permille >= self.negative_threshold_permille
    }
}

/// The activation minima a [`RepresentationStore`] re-checks against the measured gate values,
/// whatever thresholds the artifact claims for itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActivationPolicy {
    pub min_alias_permille: u16,
    pub min_negative_reject_permille: u16,
}

impl Default for ActivationPolicy {
    fn default() -> Self {
        Self {
            min_alias_permille: ACTIVATION_ALIAS_PERMILLE,
            min_negative_reject_permille: ACTIVATION_NEGATIVE_REJECT_PERMILLE,
        }
    }
}

/// Why a representation was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RepresentationError {
    UnsupportedSchema {
        expected: u32,
        actual: u32,
    },
    EmptyField {
        field: &'static str,
    },
    InvalidDimensions {
        actual: usize,
        max: usize,
    },
    /// The vector length does not match the declared (or the store's) dimension.
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    NonFinite {
        field: &'static str,
    },
    /// `z` is the zero vector, so cosine similarity is undefined.
    ZeroVector,
    InvalidGate,
    /// `activated` disagrees with the measured gate.
    GateInconsistent,
    InvalidClusterKey,
    EncoderMismatch {
        expected: EncoderRef,
        actual: EncoderRef,
    },
    /// The weights file does not have `(dims + 1) * 4` bytes.
    WeightsLength {
        expected: usize,
        actual: usize,
    },
    /// The SHA-256 of the weights file differs from the manifest's.
    WeightsDigestMismatch,
    /// The digest over all fields differs from the manifest's.
    DigestMismatch,
    /// A newer or equal generation is already loaded for this pod.
    StaleGeneration {
        current: Generation,
        requested: Generation,
    },
    /// The manifest is not valid JSON of the expected shape; `field` names the first fault.
    Malformed {
        field: &'static str,
    },
}

impl fmt::Display for RepresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RepresentationError {}

/// The fields of a [`PodRepresentation`], before validation and hashing.
#[derive(Clone, Debug, PartialEq)]
pub struct PodRepresentationParts {
    pub pod: PodId,
    pub identity: PodIdentity,
    pub cluster_key: ClusterKey,
    pub encoder: EncoderRef,
    /// The pod generation this representation was trained for.
    pub generation: Generation,
    pub z: Vec<f32>,
    pub bias: f32,
    pub weights: ScoreWeights,
    pub gate: GateReport,
    pub activated: bool,
}

/// A validated, hashed learned address of one pod generation.
#[derive(Clone, Debug, PartialEq)]
pub struct PodRepresentation {
    parts: PodRepresentationParts,
    z_norm: f64,
    digest: [u8; 32],
}

impl PodRepresentation {
    /// Validates `parts` and computes the digest.
    pub fn new(parts: PodRepresentationParts) -> Result<Self, RepresentationError> {
        validate_parts(&parts)?;
        let z_norm = norm(&parts.z);
        let digest = digest_of(&parts);
        Ok(Self {
            parts,
            z_norm,
            digest,
        })
    }

    /// Reads the manifest and weights file written by the exporter.
    ///
    /// # Errors
    /// [`RepresentationError::Malformed`] for a manifest of the wrong shape,
    /// [`RepresentationError::WeightsLength`] and [`RepresentationError::WeightsDigestMismatch`]
    /// for a weights file that is not the one the manifest describes,
    /// [`RepresentationError::DigestMismatch`] when any field was changed after export, and the
    /// validation errors of [`PodRepresentation::new`].
    pub fn from_artifact(manifest_json: &str, weights: &[u8]) -> Result<Self, RepresentationError> {
        let manifest: serde_json::Value =
            serde_json::from_str(manifest_json).map_err(|_| malformed("manifest"))?;
        let root = manifest.as_object().ok_or_else(|| malformed("manifest"))?;
        let schema = uint(root.get("schema"), "schema")?;
        let schema = u32::try_from(schema).map_err(|_| malformed("schema"))?;
        if schema != REPRESENTATION_SCHEMA {
            return Err(RepresentationError::UnsupportedSchema {
                expected: REPRESENTATION_SCHEMA,
                actual: schema,
            });
        }
        if text(root.get("kind"), "kind")? != ARTIFACT_KIND {
            return Err(malformed("kind"));
        }
        let encoder = object(root.get("encoder"), "encoder")?;
        let encoder = EncoderRef {
            id: text(encoder.get("id"), "encoder.id")?.to_owned(),
            revision: text(encoder.get("revision"), "encoder.revision")?.to_owned(),
        };
        let dims =
            usize::try_from(uint(root.get("dims"), "dims")?).map_err(|_| malformed("dims"))?;
        if dims == 0 || dims > MAX_REPRESENTATION_DIMS {
            return Err(RepresentationError::InvalidDimensions {
                actual: dims,
                max: MAX_REPRESENTATION_DIMS,
            });
        }
        let expected_len = (dims + 1) * 4;
        if weights.len() != expected_len {
            return Err(RepresentationError::WeightsLength {
                expected: expected_len,
                actual: weights.len(),
            });
        }
        let claimed_weights_digest =
            hex_digest(text(root.get("weights_sha256"), "weights_sha256")?)
                .ok_or_else(|| malformed("weights_sha256"))?;
        if <[u8; 32]>::from(Sha256::digest(weights)) != claimed_weights_digest {
            return Err(RepresentationError::WeightsDigestMismatch);
        }
        let mut values = weights
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        let z: Vec<f32> = values.by_ref().take(dims).collect();
        let bias = values.next().ok_or_else(|| malformed("weights"))?;

        let scoring = object(root.get("scoring"), "scoring")?;
        let weights_cfg = ScoreWeights {
            cosine: exact_f32(scoring.get("cosine_weight"), "scoring.cosine_weight")?,
            metadata: exact_f32(scoring.get("metadata_weight"), "scoring.metadata_weight")?,
        };
        let gate_json = object(root.get("gate"), "gate")?;
        let permille = |key: &'static str| -> Result<u16, RepresentationError> {
            u16::try_from(uint(gate_json.get(key), key)?).map_err(|_| malformed(key))
        };
        let count = |key: &'static str| -> Result<u32, RepresentationError> {
            u32::try_from(uint(gate_json.get(key), key)?).map_err(|_| malformed(key))
        };
        let gate = GateReport {
            alias_threshold_permille: permille("alias_threshold_permille")?,
            alias_min_permille: permille("alias_min_permille")?,
            negative_threshold_permille: permille("negative_threshold_permille")?,
            negative_reject_permille: permille("negative_reject_permille")?,
            alias_count: count("alias_count")?,
            negative_count: count("negative_count")?,
        };
        let activated = match text(gate_json.get("status"), "gate.status")? {
            STATUS_ACTIVATED => true,
            STATUS_NOT_ACTIVATED => false,
            _ => return Err(malformed("gate.status")),
        };
        let cluster_key = ClusterKey::from_hex(text(root.get("cluster_key"), "cluster_key")?)
            .map_err(|_| RepresentationError::InvalidClusterKey)?;
        let claimed_digest =
            hex_digest(text(root.get("digest"), "digest")?).ok_or_else(|| malformed("digest"))?;

        let representation = Self::new(PodRepresentationParts {
            pod: PodId(text(root.get("pod_id"), "pod_id")?.to_owned()),
            identity: PodIdentity(text(root.get("pod_identity"), "pod_identity")?.to_owned()),
            cluster_key,
            encoder,
            generation: Generation(uint(root.get("generation"), "generation")?),
            z,
            bias,
            weights: weights_cfg,
            gate,
            activated,
        })?;
        if representation.digest != claimed_digest {
            return Err(RepresentationError::DigestMismatch);
        }
        Ok(representation)
    }

    /// The weights file: `z` followed by the bias, little endian `f32`.
    pub fn weights_bytes(&self) -> Vec<u8> {
        weights_bytes(&self.parts.z, self.parts.bias)
    }

    /// SHA-256 over a domain tag and every field (length-prefixed text, little endian numbers,
    /// `f32` as raw bits) followed by the weights file. Any changed field changes the digest.
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn pod(&self) -> &PodId {
        &self.parts.pod
    }

    pub fn identity(&self) -> &PodIdentity {
        &self.parts.identity
    }

    pub fn cluster_key(&self) -> ClusterKey {
        self.parts.cluster_key
    }

    pub fn encoder(&self) -> &EncoderRef {
        &self.parts.encoder
    }

    pub fn generation(&self) -> Generation {
        self.parts.generation
    }

    pub fn dims(&self) -> usize {
        self.parts.z.len()
    }

    pub fn z(&self) -> &[f32] {
        &self.parts.z
    }

    pub fn bias(&self) -> f32 {
        self.parts.bias
    }

    pub fn weights(&self) -> ScoreWeights {
        self.parts.weights
    }

    pub fn gate(&self) -> GateReport {
        self.parts.gate
    }

    pub fn activated(&self) -> bool {
        self.parts.activated
    }

    /// Re-checks every invariant [`PodRepresentation::new`] enforces and the digest.
    pub fn validate(&self) -> Result<(), RepresentationError> {
        validate_parts(&self.parts)?;
        if digest_of(&self.parts) != self.digest {
            return Err(RepresentationError::DigestMismatch);
        }
        Ok(())
    }

    /// The sigmoid score of `query` for this pod (the router's formula, `m = 1`). `None` when the
    /// query is the zero vector.
    fn score(&self, query: &[f32]) -> Option<f64> {
        let query_norm = norm(query);
        if query_norm == 0.0 || !query_norm.is_finite() {
            return None;
        }
        let dot: f64 = query
            .iter()
            .zip(&self.parts.z)
            .map(|(q, z)| f64::from(*q) * f64::from(*z))
            .sum();
        let cosine = (dot / (query_norm * self.z_norm)).clamp(-1.0, 1.0);
        let logit = f64::from(self.parts.weights.cosine) * cosine
            + f64::from(self.parts.weights.metadata)
            - f64::from(self.parts.bias);
        Some(sigmoid(logit))
    }
}

fn malformed(field: &'static str) -> RepresentationError {
    RepresentationError::Malformed { field }
}

fn uint(
    value: Option<&serde_json::Value>,
    field: &'static str,
) -> Result<u64, RepresentationError> {
    value
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| malformed(field))
}

fn text<'a>(
    value: Option<&'a serde_json::Value>,
    field: &'static str,
) -> Result<&'a str, RepresentationError> {
    value
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| malformed(field))
}

fn object<'a>(
    value: Option<&'a serde_json::Value>,
    field: &'static str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, RepresentationError> {
    value
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| malformed(field))
}

/// A JSON number that is exactly an `f32`; anything that would round is refused, so the manifest
/// cannot disagree with the bits the digest covers.
fn exact_f32(
    value: Option<&serde_json::Value>,
    field: &'static str,
) -> Result<f32, RepresentationError> {
    let wide = value
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| malformed(field))?;
    let narrow = wide as f32;
    if f64::from(narrow) != wide {
        return Err(malformed(field));
    }
    Ok(narrow)
}

fn hex_digest(value: &str) -> Option<[u8; 32]> {
    let bytes = value.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let nibble = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    };
    let mut digest = [0u8; 32];
    for (slot, pair) in digest.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(digest)
}

fn validate_parts(parts: &PodRepresentationParts) -> Result<(), RepresentationError> {
    for (field, empty) in [
        ("pod", parts.pod.0.is_empty()),
        ("identity", parts.identity.0.is_empty()),
        ("encoder.id", parts.encoder.id.is_empty()),
        ("encoder.revision", parts.encoder.revision.is_empty()),
    ] {
        if empty {
            return Err(RepresentationError::EmptyField { field });
        }
    }
    if parts.z.is_empty() || parts.z.len() > MAX_REPRESENTATION_DIMS {
        return Err(RepresentationError::InvalidDimensions {
            actual: parts.z.len(),
            max: MAX_REPRESENTATION_DIMS,
        });
    }
    if parts.z.iter().any(|value| !value.is_finite()) {
        return Err(RepresentationError::NonFinite { field: "z" });
    }
    if !parts.bias.is_finite() {
        return Err(RepresentationError::NonFinite { field: "bias" });
    }
    for (field, value) in [
        ("cosine_weight", parts.weights.cosine),
        ("metadata_weight", parts.weights.metadata),
    ] {
        if !value.is_finite() {
            return Err(RepresentationError::NonFinite { field });
        }
    }
    if parts.weights.cosine <= 0.0 {
        return Err(RepresentationError::NonFinite {
            field: "cosine_weight",
        });
    }
    let z_norm = norm(&parts.z);
    if !z_norm.is_finite() {
        return Err(RepresentationError::NonFinite { field: "z" });
    }
    if z_norm == 0.0 {
        return Err(RepresentationError::ZeroVector);
    }
    let gate = parts.gate;
    if [
        gate.alias_threshold_permille,
        gate.alias_min_permille,
        gate.negative_threshold_permille,
        gate.negative_reject_permille,
    ]
    .iter()
    .any(|permille| *permille > 1000)
    {
        return Err(RepresentationError::InvalidGate);
    }
    if parts.activated != gate.passes() {
        return Err(RepresentationError::GateInconsistent);
    }
    Ok(())
}

fn norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt()
}

fn sigmoid(logit: f64) -> f64 {
    1.0 / (1.0 + (-logit).exp())
}

fn weights_bytes(z: &[f32], bias: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((z.len() + 1) * 4);
    for value in z.iter().chain(std::iter::once(&bias)) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn put_text(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn digest_of(parts: &PodRepresentationParts) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_text(&mut bytes, DIGEST_DOMAIN);
    bytes.extend_from_slice(&REPRESENTATION_SCHEMA.to_le_bytes());
    put_text(&mut bytes, &parts.encoder.id);
    put_text(&mut bytes, &parts.encoder.revision);
    bytes.extend_from_slice(&(parts.z.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&parts.generation.0.to_le_bytes());
    put_text(&mut bytes, &parts.pod.0);
    put_text(&mut bytes, &parts.identity.0);
    bytes.extend_from_slice(parts.cluster_key.digest());
    bytes.extend_from_slice(&parts.weights.cosine.to_bits().to_le_bytes());
    bytes.extend_from_slice(&parts.weights.metadata.to_bits().to_le_bytes());
    bytes.push(u8::from(parts.activated));
    for permille in [
        parts.gate.alias_threshold_permille,
        parts.gate.alias_min_permille,
        parts.gate.negative_threshold_permille,
        parts.gate.negative_reject_permille,
    ] {
        bytes.extend_from_slice(&permille.to_le_bytes());
    }
    bytes.extend_from_slice(&parts.gate.alias_count.to_le_bytes());
    bytes.extend_from_slice(&parts.gate.negative_count.to_le_bytes());
    bytes.extend_from_slice(&weights_bytes(&parts.z, parts.bias));
    Sha256::digest(bytes).into()
}

/// The activated representations a router may score, one per pod, all of one encoder and one
/// dimension.
#[derive(Clone, Debug)]
pub struct RepresentationStore {
    encoder: EncoderRef,
    policy: ActivationPolicy,
    entries: BTreeMap<PodId, PodRepresentation>,
}

impl RepresentationStore {
    pub fn new(encoder: EncoderRef) -> Self {
        Self::with_policy(encoder, ActivationPolicy::default())
    }

    pub fn with_policy(encoder: EncoderRef, policy: ActivationPolicy) -> Self {
        Self {
            encoder,
            policy,
            entries: BTreeMap::new(),
        }
    }

    /// Loads an activated representation, replacing an older generation of the same pod.
    ///
    /// # Errors
    /// [`RouteError::InactiveRepresentation`] for a representation that is not activated, or
    /// whose measured gate values are below the store's [`ActivationPolicy`] whatever thresholds
    /// the artifact claims; [`RouteError::InvalidRepresentation`] for one that fails validation,
    /// belongs to another encoder, has another dimension than the store, or is not newer than
    /// the one loaded.
    pub fn insert(
        &mut self,
        representation: PodRepresentation,
    ) -> Result<Option<PodRepresentation>, RouteError> {
        let pod = representation.pod().clone();
        let invalid = |error| RouteError::InvalidRepresentation {
            pod: pod.clone(),
            error,
        };
        representation.validate().map_err(invalid)?;
        if !representation.activated() {
            return Err(RouteError::InactiveRepresentation { pod });
        }
        let gate = representation.gate();
        if gate.alias_min_permille < self.policy.min_alias_permille
            || gate.negative_reject_permille < self.policy.min_negative_reject_permille
        {
            return Err(RouteError::InactiveRepresentation { pod });
        }
        if representation.encoder() != &self.encoder {
            return Err(invalid(RepresentationError::EncoderMismatch {
                expected: self.encoder.clone(),
                actual: representation.encoder().clone(),
            }));
        }
        if let Some(dims) = self.dims() {
            if representation.dims() != dims {
                return Err(invalid(RepresentationError::DimensionMismatch {
                    expected: dims,
                    actual: representation.dims(),
                }));
            }
        }
        if let Some(current) = self.entries.get(&pod) {
            if representation.generation() <= current.generation() {
                return Err(invalid(RepresentationError::StaleGeneration {
                    current: current.generation(),
                    requested: representation.generation(),
                }));
            }
        }
        Ok(self.entries.insert(pod, representation))
    }

    /// Drops the representation of `pod`; the others are untouched.
    pub fn remove(&mut self, pod: &PodId) -> Option<PodRepresentation> {
        self.entries.remove(pod)
    }

    pub fn get(&self, pod: &PodId) -> Option<&PodRepresentation> {
        self.entries.get(pod)
    }

    pub fn encoder(&self) -> &EncoderRef {
        &self.encoder
    }

    /// The common dimension, once a representation is loaded.
    pub fn dims(&self) -> Option<usize> {
        self.entries.values().next().map(PodRepresentation::dims)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Thresholds of the learned router's abstention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearnedRouterConfig {
    /// Abstain when the best score is below this.
    pub abstain_below: f64,
    /// Abstain when the best score exceeds the runner-up's by less than this.
    pub min_margin: f64,
}

impl Default for LearnedRouterConfig {
    fn default() -> Self {
        Self {
            abstain_below: 0.5,
            min_margin: 0.15,
        }
    }
}

/// An advisory router that ranks registry candidates by their learned representation and abstains
/// when it is not confident. See the module documentation.
#[derive(Clone, Debug)]
pub struct LearnedPodRouter {
    store: RepresentationStore,
    config: LearnedRouterConfig,
}

impl LearnedPodRouter {
    /// # Errors
    /// [`RouteError::InvalidFeatures`] when a threshold is not a finite number in `[0, 1]`.
    pub fn new(
        store: RepresentationStore,
        config: LearnedRouterConfig,
    ) -> Result<Self, RouteError> {
        for (name, value) in [
            ("abstain_below", config.abstain_below),
            ("min_margin", config.min_margin),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(RouteError::InvalidFeatures {
                    message: format!("router threshold {name} must be within [0, 1]"),
                });
            }
        }
        Ok(Self { store, config })
    }

    pub fn store(&self) -> &RepresentationStore {
        &self.store
    }

    pub fn config(&self) -> LearnedRouterConfig {
        self.config
    }

    /// The learned score of `pod` for `query`; `None` when the pod has no representation.
    ///
    /// # Errors
    /// [`RouteError::InvalidFeatures`] for an empty, zero or wrongly sized query.
    pub fn score(&self, pod: &PodId, query: &[f32]) -> Result<Option<f64>, RouteError> {
        let Some(representation) = self.store.get(pod) else {
            return Ok(None);
        };
        self.check_query(query)?;
        Ok(representation.score(query))
    }

    fn check_query(&self, query: &[f32]) -> Result<(), RouteError> {
        let invalid = |message: String| RouteError::InvalidFeatures { message };
        if query.is_empty() {
            return Err(invalid("learned routing needs a query vector".into()));
        }
        if let Some(dims) = self.store.dims() {
            if query.len() != dims {
                return Err(invalid(format!(
                    "query vector has {} elements, representations have {dims}",
                    query.len()
                )));
            }
        }
        if norm(query) == 0.0 {
            return Err(invalid("query vector is the zero vector".into()));
        }
        Ok(())
    }
}

impl Router for LearnedPodRouter {
    /// A point mass on the best-scoring candidate that has a representation, or
    /// [`RouteError::Abstained`] when that candidate is not clearly the answer. Candidates
    /// without a representation are not scored; with none scored the result is
    /// [`RouteError::NoRepresentation`]. No candidate at all is the empty distribution.
    fn policy(&self, request: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        let candidates =
            request
                .registry
                .candidates(request.project, request.capability, request.input_type);
        if candidates.is_empty() {
            return Ok(RouteDistribution::none());
        }
        let query = request.features.vector();
        let mut scored: Vec<(PodId, f64)> = Vec::new();
        for pod in &candidates {
            if let Some(score) = self.score(pod, query)? {
                scored.push((pod.clone(), score));
            }
        }
        if scored.is_empty() {
            return Err(RouteError::NoRepresentation {
                candidates: candidates.len(),
            });
        }
        // Best first; equal scores resolve to the lesser PodId, as `candidates` is sorted.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let (best, best_score) = scored[0].clone();
        let runner_up = scored.get(1).cloned();
        let margin = best_score - runner_up.as_ref().map_or(0.0, |(_, score)| *score);
        let confident = best_score >= self.config.abstain_below
            && (runner_up.is_none() || margin >= self.config.min_margin);
        if !confident {
            return Err(RouteError::Abstained {
                best,
                runner_up: runner_up.map(|(pod, _)| pod),
                best_score,
                margin,
            });
        }
        Ok(RouteDistribution::certain(best))
    }
}
