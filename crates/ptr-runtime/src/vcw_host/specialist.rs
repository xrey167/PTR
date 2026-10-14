//! Specialists: the experts a structural Birth, Reactivate or Merge asks the host to produce.
//!
//! The structure policy only *requests* an expert; what the new expert can do is whatever the
//! host's training produced. [`SimulatedTrainer`] is the host's stand-in for training in the
//! CPU experiments: it produces a specialist whose skill per finding code is a **configuration**
//! (a seeded table), not the result of any learning. The manifest digest the specialist carries
//! commits to that configuration (the seed and the skill table are capabilities of the manifest),
//! so the ledger record of the Birth commits to what was configured, and a replay can check the
//! host's manifest against it. Nothing here claims a trained model.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use ptr_pods::{DynPod, ExpertInputs, ManifestDigest, OpKind, PodManifest, TypedPayload};
use ptr_router::FindingCode;
use ptr_types::{CapabilityId, PodId, ProjectId, TypeId};

use crate::vcw::experts::{expert_manifest, ExpertRole};

/// Prefix of the manifest capability that names a finding code the specialist repairs.
pub const REPAIRS_PREFIX: &str = "repairs:";

/// What the host asks a trainer to produce.
#[derive(Clone, Debug)]
pub struct TrainingRequest {
    pub kind: OpKind,
    /// The Pod id the structure policy chose for the result.
    pub pod: PodId,
    /// The finding codes the result is to repair.
    pub codes: BTreeSet<FindingCode>,
    /// The experts the result replaces (a merge's sources, a reactivation's target).
    pub sources: Vec<PodId>,
}

/// A change to another Pod's skill that "training" this specialist caused: the simulated
/// counterpart of catastrophic forgetting. The trainer's statement of it is bound into the
/// manifest (`sim:collateral:` capabilities) so the committed record commits to what was
/// claimed, but it is a claim: a held-out suite measures the effect itself (see
/// `SimulatedHeldOut::with_world_of`) and never reads this field.
#[derive(Clone, Debug, PartialEq)]
pub struct Collateral {
    pub pod: PodId,
    pub code: String,
    pub success_after: f64,
}

/// The product of training.
#[derive(Clone, Debug)]
pub struct TrainedSpecialist {
    pub pod: PodId,
    pub codes: BTreeSet<FindingCode>,
    /// Probability of clearing each code (simulation configuration). The trainer's own claim:
    /// untrusted, not read by the held-out suite.
    pub skill: BTreeMap<String, f64>,
    /// The collateral the trainer states. Untrusted, bound into the manifest digest, not read
    /// by the held-out suite.
    pub collateral: Vec<Collateral>,
    pub manifest: PodManifest,
    pub manifest_digest: ManifestDigest,
    /// The datasets the specialist rests on (its training and evaluation inputs), if the host
    /// declares them. Admission then binds the record to their generations and digests, and a
    /// later revocation of one of them fences the specialist.
    pub inputs: Option<ExpertInputs>,
    /// What the skill is: a statement that no learning happened.
    pub provenance: &'static str,
}

impl TrainedSpecialist {
    /// The registry entry for the specialist. It is driven through the host's `RolePods`;
    /// invoking it directly is refused.
    pub fn executable(&self) -> Arc<dyn DynPod> {
        Arc::new(SpecialistPod {
            manifest: self.manifest.clone(),
        })
    }
}

/// Why a specialist could not be produced.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TrainingError {
    /// The request names no code.
    NoCodes,
    /// A configured probability is outside `0..=1`.
    BadProbability,
    /// The manifest digest is the all-zero digest (cannot be bound).
    ZeroDigest,
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCodes => f.write_str("the training request names no finding code"),
            Self::BadProbability => f.write_str("a configured skill is not a probability"),
            Self::ZeroDigest => f.write_str("the manifest digest is all zero"),
        }
    }
}

impl std::error::Error for TrainingError {}

/// Produces the expert a Birth, Reactivate or Merge asks for.
pub trait SpecialistTrainer {
    /// # Errors
    /// [`TrainingError`] when no specialist can be produced; the request is then withdrawn.
    fn train(&mut self, request: &TrainingRequest) -> Result<TrainedSpecialist, TrainingError>;
}

/// A trainer that configures, rather than trains, a specialist: skill per code from a table
/// (default `default_skill`), manifest from the configuration.
#[derive(Clone, Debug)]
pub struct SimulatedTrainer {
    project: ProjectId,
    payload: TypeId,
    seed: u64,
    default_skill: f64,
    skills: BTreeMap<String, f64>,
    collateral: Vec<Collateral>,
    inputs: Option<ExpertInputs>,
    trained: u64,
}

impl SimulatedTrainer {
    pub const PROVENANCE: &'static str = "simulated: a seeded skill table; no learning took place";

    #[must_use]
    pub fn new(project: ProjectId, payload: TypeId, seed: u64, default_skill: f64) -> Self {
        Self {
            project,
            payload,
            seed,
            default_skill,
            skills: BTreeMap::new(),
            collateral: Vec::new(),
            inputs: None,
            trained: 0,
        }
    }

    /// Every specialist trained from now on declares these inputs.
    #[must_use]
    pub fn with_inputs(mut self, inputs: ExpertInputs) -> Self {
        self.inputs = Some(inputs);
        self
    }

    /// The skill of every specialist for `code`.
    #[must_use]
    pub fn with_skill(mut self, code: &str, probability: f64) -> Self {
        self.skills.insert(code.to_owned(), probability);
        self
    }

    /// Every specialist trained from now on also lowers (or changes) `pod`'s skill on `code`.
    #[must_use]
    pub fn with_collateral(mut self, pod: &str, code: &str, success_after: f64) -> Self {
        self.collateral.push(Collateral {
            pod: PodId::from(pod),
            code: code.to_owned(),
            success_after,
        });
        self
    }

    /// Specialists produced so far.
    pub fn trained(&self) -> u64 {
        self.trained
    }

    /// The configured skill for codes with no entry in [`Self::configured_skills`].
    pub fn default_skill(&self) -> f64 {
        self.default_skill
    }

    /// The configured per-code skills (simulation configuration, for building the world a
    /// simulated suite measures).
    pub fn configured_skills(&self) -> &BTreeMap<String, f64> {
        &self.skills
    }

    /// The configured collateral (simulation configuration).
    pub fn configured_collateral(&self) -> &[Collateral] {
        &self.collateral
    }
}

impl SpecialistTrainer for SimulatedTrainer {
    fn train(&mut self, request: &TrainingRequest) -> Result<TrainedSpecialist, TrainingError> {
        if request.codes.is_empty() {
            return Err(TrainingError::NoCodes);
        }
        let ok = |p: f64| p.is_finite() && (0.0..=1.0).contains(&p);
        let mut skill = BTreeMap::new();
        for code in &request.codes {
            let p = self
                .skills
                .get(code.as_str())
                .copied()
                .unwrap_or(self.default_skill);
            if !ok(p) {
                return Err(TrainingError::BadProbability);
            }
            skill.insert(code.as_str().to_owned(), p);
        }
        if self.collateral.iter().any(|c| !ok(c.success_after)) {
            return Err(TrainingError::BadProbability);
        }
        let mut manifest = expert_manifest(
            &self.project,
            &request.pod,
            ExpertRole::Repairer,
            &self.payload,
        );
        manifest.capabilities.push(CapabilityId::from(
            format!("sim:seed:{}", self.seed).as_str(),
        ));
        for (code, p) in &skill {
            manifest.capabilities.push(CapabilityId::from(
                format!("{REPAIRS_PREFIX}{code}").as_str(),
            ));
            manifest.capabilities.push(CapabilityId::from(
                format!("sim:skill:{code}:{}", (p * 1_000_000.0).round() as u64).as_str(),
            ));
        }
        // The collateral the trainer states is part of what the ledger commits to.
        for collateral in &self.collateral {
            manifest
                .capabilities
                .push(collateral_capability(collateral));
        }
        let manifest_digest =
            ManifestDigest::new(manifest.digest()).map_err(|_| TrainingError::ZeroDigest)?;
        self.trained += 1;
        Ok(TrainedSpecialist {
            pod: request.pod.clone(),
            codes: request.codes.clone(),
            skill,
            collateral: self.collateral.clone(),
            manifest,
            manifest_digest,
            inputs: self.inputs.clone(),
            provenance: Self::PROVENANCE,
        })
    }
}

/// The `sim:collateral:` capability of one stated collateral, version 2 (issue #141): the Pod id
/// and the code are each preceded by their byte length, so a `:` inside either cannot move a
/// field boundary. Version 1 joined them with a bare `:`, and Pod `a:b` with code `c` read the
/// same as Pod `a` with code `b:c`.
fn collateral_capability(collateral: &Collateral) -> CapabilityId {
    CapabilityId::from(
        format!(
            "sim:collateral:v2:{}:{}:{}:{}:{}",
            collateral.pod.0.len(),
            collateral.pod.0,
            collateral.code.len(),
            collateral.code,
            (collateral.success_after * 1_000_000.0).round() as u64
        )
        .as_str(),
    )
}

/// The codes a manifest declares it repairs (`repairs:<code>` capabilities), in code order.
#[must_use]
pub fn repaired_codes(manifest: &PodManifest) -> BTreeSet<FindingCode> {
    manifest
        .capabilities
        .iter()
        .filter_map(|capability| capability.0.strip_prefix(REPAIRS_PREFIX))
        .map(FindingCode::new)
        .collect()
}

/// The registry entry of a specialist.
struct SpecialistPod {
    manifest: PodManifest,
}

impl DynPod for SpecialistPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, _input: TypedPayload) -> Result<TypedPayload, String> {
        Err("a specialist is driven through RolePods".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(codes: &[&str]) -> TrainingRequest {
        TrainingRequest {
            kind: OpKind::Birth,
            pod: PodId::from("born"),
            codes: codes.iter().map(|c| FindingCode::new(*c)).collect(),
            sources: Vec::new(),
        }
    }

    fn trainer() -> SimulatedTrainer {
        SimulatedTrainer::new(ProjectId::from("p"), TypeId::from("t"), 9, 0.8)
    }

    #[test]
    fn the_manifest_digest_commits_to_the_seed_and_the_skill_table() {
        let a = trainer().train(&request(&["x"])).unwrap();
        let b = trainer().train(&request(&["x"])).unwrap();
        assert_eq!(a.manifest_digest, b.manifest_digest);
        let other_skill = trainer()
            .with_skill("x", 0.5)
            .train(&request(&["x"]))
            .unwrap();
        assert_ne!(a.manifest_digest, other_skill.manifest_digest);
        let other_seed = SimulatedTrainer::new(ProjectId::from("p"), TypeId::from("t"), 10, 0.8)
            .train(&request(&["x"]))
            .unwrap();
        assert_ne!(a.manifest_digest, other_seed.manifest_digest);
        let other_codes = trainer().train(&request(&["x", "y"])).unwrap();
        assert_ne!(a.manifest_digest, other_codes.manifest_digest);
        // The collateral a trainer states is committed to as well.
        let harmful = trainer()
            .with_collateral("o", "x", 0.0)
            .train(&request(&["x"]))
            .unwrap();
        assert_ne!(a.manifest_digest, harmful.manifest_digest);
        let milder = trainer()
            .with_collateral("o", "x", 0.5)
            .train(&request(&["x"]))
            .unwrap();
        assert_ne!(harmful.manifest_digest, milder.manifest_digest);
        assert_eq!(repaired_codes(&a.manifest), a.codes);
        assert_eq!(a.provenance, SimulatedTrainer::PROVENANCE);
    }

    /// Issue #141: a `:` inside the Pod id or the code must not move a field boundary.
    #[test]
    fn collateral_with_a_colon_in_the_pod_or_code_does_not_alias() {
        let left = trainer()
            .with_collateral("a:b", "c", 0.5)
            .train(&request(&["x"]))
            .unwrap();
        let right = trainer()
            .with_collateral("a", "b:c", 0.5)
            .train(&request(&["x"]))
            .unwrap();
        assert_ne!(left.manifest.capabilities, right.manifest.capabilities);
        assert_ne!(left.manifest_digest, right.manifest_digest);
    }

    #[test]
    fn bad_requests_and_bad_tables_are_refused() {
        assert_eq!(
            trainer().train(&request(&[])).unwrap_err(),
            TrainingError::NoCodes
        );
        assert_eq!(
            trainer()
                .with_skill("x", 1.5)
                .train(&request(&["x"]))
                .unwrap_err(),
            TrainingError::BadProbability
        );
        assert_eq!(
            trainer()
                .with_collateral("o", "x", f64::NAN)
                .train(&request(&["x"]))
                .unwrap_err(),
            TrainingError::BadProbability
        );
    }
}
