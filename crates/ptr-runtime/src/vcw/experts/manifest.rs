//! Manifests of expert Pods.

use ptr_pods::{ModelIdentity, PodManifest};
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, TypeId};

/// The role an expert Pod plays in the loop.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ExpertRole {
    /// Task to candidate.
    Proposer,
    /// Candidate and one finding code to repaired candidate.
    Repairer,
}

impl ExpertRole {
    /// The capability the manifest declares.
    #[must_use]
    pub fn capability(self) -> &'static str {
        match self {
            Self::Proposer => "propose",
            Self::Repairer => "repair",
        }
    }
}

/// The manifest of an expert Pod: it accepts and produces `payload_type`, has no effect
/// (`Effect::Pure`: its output is a candidate, not an action) and speaks protocol version 1.
/// Its generation is not part of the manifest; it is held by the lifecycle authority.
#[must_use]
pub fn expert_manifest(
    project: &ProjectId,
    pod: &PodId,
    role: ExpertRole,
    payload_type: &TypeId,
) -> PodManifest {
    PodManifest {
        project: project.clone(),
        id: pod.clone(),
        capabilities: vec![CapabilityId::from(role.capability())],
        accepts: vec![payload_type.clone()],
        produces: vec![payload_type.clone()],
        effects: vec![Effect::Pure],
        protocol_version: 1,
    }
}

/// [`expert_manifest`] plus one capability per model file, `model:<role>:sha256:<hex>`, so the
/// manifest (and its digest) attests which weights, config and tokenizer the Pod runs.
#[must_use]
pub fn expert_manifest_for_model(
    project: &ProjectId,
    pod: &PodId,
    role: ExpertRole,
    payload_type: &TypeId,
    model: &ModelIdentity,
) -> PodManifest {
    let mut manifest = expert_manifest(project, pod, role, payload_type);
    manifest
        .capabilities
        .extend(model.files.iter().map(|(name, sha256)| {
            CapabilityId::from(format!("model:{name}:sha256:{sha256}").as_str())
        }));
    manifest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manifest_declares_one_pure_capability_for_its_role() {
        let manifest = expert_manifest(
            &ProjectId::from("p"),
            &PodId::from("prop"),
            ExpertRole::Proposer,
            &TypeId::from("t"),
        );
        assert_eq!(manifest.capabilities, vec![CapabilityId::from("propose")]);
        assert_eq!(manifest.effects, vec![Effect::Pure]);
        assert_eq!(manifest.accepts, manifest.produces);
    }

    #[test]
    fn the_model_digests_are_in_the_manifest_and_change_its_digest() {
        let (project, pod, ty) = (ProjectId::from("p"), PodId::from("x"), TypeId::from("t"));
        let identity = ModelIdentity::new(vec![
            ("weights".into(), "aa".into()),
            ("config".into(), "bb".into()),
        ]);
        let plain = expert_manifest(&project, &pod, ExpertRole::Repairer, &ty);
        let pinned =
            expert_manifest_for_model(&project, &pod, ExpertRole::Repairer, &ty, &identity);
        let names: Vec<_> = pinned.capabilities.iter().map(|c| c.0.as_str()).collect();
        assert_eq!(
            names,
            [
                "repair",
                "model:config:sha256:bb",
                "model:weights:sha256:aa"
            ]
        );
        assert_ne!(plain.digest(), pinned.digest());
    }
}
