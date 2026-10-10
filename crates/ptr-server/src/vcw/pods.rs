//! The Pods of the path as the Pod registry and the lifecycle authority see them.

use std::sync::Arc;

use ptr_pods::{DynPod, ModelIdentity, PodManifest, TypedPayload};
use ptr_runtime::vcw::experts::{expert_manifest, expert_manifest_for_model, ExpertRole};
use ptr_types::{CapabilityId, PodId, ProjectId, TypeId};

/// A registry entry. The loop drives the experts through `RolePods`, so invoking the entry
/// directly is refused; the entry exists so the Pod has a manifest, a project and a digest.
pub(super) struct ServedPod {
    manifest: PodManifest,
}

impl ServedPod {
    /// The entry of `pod` in `role`, accepting and producing every type in `types`. A model's
    /// file digests, when known, are part of the manifest and so of its digest.
    pub(super) fn new(
        project: &ProjectId,
        pod: &PodId,
        role: ExpertRole,
        types: &[TypeId],
        model: Option<&ModelIdentity>,
        backend: &str,
    ) -> Self {
        let first = types
            .first()
            .cloned()
            .unwrap_or_else(|| TypeId::from("none"));
        let mut manifest = match model {
            Some(identity) => expert_manifest_for_model(project, pod, role, &first, identity),
            None => expert_manifest(project, pod, role, &first),
        };
        // The backend is part of the manifest, so a Pod served by simulated experts has another
        // manifest digest than the same Pod id served by a model.
        manifest
            .capabilities
            .push(CapabilityId::from(format!("backend:{backend}").as_str()));
        manifest.accepts = types.to_vec();
        manifest.produces = types.to_vec();
        Self { manifest }
    }

    pub(super) fn into_dyn(self) -> Arc<dyn DynPod> {
        Arc::new(self)
    }
}

impl DynPod for ServedPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, _input: TypedPayload) -> Result<TypedPayload, String> {
        Err("a vcw expert is driven through RolePods".to_owned())
    }
}

/// The one-line description a repair model is shown for a finding code.
#[cfg(feature = "qwen-cpu")]
pub(super) fn describe(code: &str) -> &'static str {
    match code {
        "arith.value" => "the integer is not the correct result",
        "arith.format" => "the answer is not a single integer",
        "arith.bounds" => "the number is too large",
        "arith.oversize" => "the answer is too long",
        "json.parse" => "the text is not one valid JSON object",
        "json.oversize" => "the document is too large",
        "json.depth" => "the document is nested too deeply",
        "json.type" => "a value has the wrong JSON type",
        "json.required" => "a required property is missing",
        "json.enum" => "a value is not one of the allowed values",
        "json.range" => "a number is outside the allowed range",
        "json.length" => "a string has the wrong length",
        "json.extra" => "a property is present that is not allowed",
        _ => "the checker rejected the answer",
    }
}
