//! The `qwen-cpu` backend: a real language model on the CPU behind `ptr_pods::TextGenerator`.
//!
//! The model loads only if the directory holds the four pinned files and every digest from the
//! configuration matches; a mismatch is a start-up error, not a warning.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ptr_config::{VcwServiceConfig, VcwTaskKind};
use ptr_pods::{ModelIdentity, QwenCpu, QwenCpuOptions};
use ptr_runtime::vcw::experts::{
    ExpertRole, ExpertTask, LlmExpertPod, LlmExpertPods, SharedGenerator, TaskKind,
};
use ptr_types::{PodId, ProjectId};

use super::pods::describe;
use super::service::{Experts, VcwBuildError};

pub(super) fn llm_experts(
    config: &VcwServiceConfig,
    kinds: &[VcwTaskKind],
    project: &ProjectId,
    proposer: &PodId,
    repairers: &[(PodId, Vec<String>)],
) -> Result<(BTreeMap<VcwTaskKind, Experts>, Option<ModelIdentity>), VcwBuildError> {
    let qwen = &config.experts.qwen_cpu;
    let options = QwenCpuOptions::default()
        .with_expected_weights_sha256(qwen.weights_sha256.clone())
        .with_expected_config_sha256(qwen.config_sha256.clone())
        .with_expected_tokenizer_sha256(qwen.tokenizer_sha256.clone())
        .with_expected_tokenizer_config_sha256(qwen.tokenizer_config_sha256.clone());
    let model = QwenCpu::load(Path::new(&qwen.model_dir), &options)
        .map_err(|error| VcwBuildError::Qwen(format!("{error:?}")))?;
    let generator: SharedGenerator = Arc::new(Mutex::new(model));
    let identity = generator
        .lock()
        .map_err(|_| VcwBuildError::Qwen("generator lock poisoned".into()))?
        .identity();
    let max_new_tokens = qwen.effective_max_new_tokens();

    let mut out = BTreeMap::new();
    for kind in kinds {
        let task_kind = match kind {
            VcwTaskKind::Arithmetic => TaskKind::Arithmetic,
            VcwTaskKind::JsonObject => TaskKind::JsonObject,
            VcwTaskKind::Code | VcwTaskKind::Fact => {
                return Err(VcwBuildError::Config(format!(
                    "the qwen-cpu backend has no prompts for {}",
                    kind.as_str()
                )))
            }
        };
        let pod = |id: &PodId, role: ExpertRole, system: &str| {
            LlmExpertPod::new(
                project,
                id,
                role,
                task_kind,
                system,
                max_new_tokens,
                generator.clone(),
            )
            .map(Arc::new)
            .map_err(|error| VcwBuildError::Pods(error.to_string()))
        };
        let mut experts = LlmExpertPods::new(ExpertTask::new(task_kind, "unset"));
        experts
            .add_proposer(pod(
                proposer,
                ExpertRole::Proposer,
                task_kind.proposer_system_prompt(),
            )?)
            .map_err(|error| VcwBuildError::Pods(error.to_string()))?;
        for (id, codes) in repairers {
            let described: Vec<(&str, &str)> = codes
                .iter()
                .map(|code| (code.as_str(), describe(code)))
                .collect();
            experts
                .add_repairer(
                    pod(id, ExpertRole::Repairer, task_kind.repairer_system_prompt())?,
                    &described,
                )
                .map_err(|error| VcwBuildError::Pods(error.to_string()))?;
        }
        out.insert(*kind, Experts::Llm(experts));
    }
    Ok((out, identity))
}
