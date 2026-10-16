//! Weight objects in the tier store: whole-tensor restore and one object per expert.

use ptr_pods::{
    ExpertWeightTierAdapter, ExpertWeightTierBinding, ModelWeightTierAdapter,
    ModelWeightTierBinding, PodTierError, MAX_EXPERTS_PER_MODEL,
};
use ptr_storage::{ChunkDescriptor, PreparedTierObject, TierObjectDomain, TierObjectManifest};
use ptr_types::{AdapterVersion, Generation, ModelVersion, Revision};

type BindingChange = Box<dyn Fn(&mut ModelWeightTierBinding)>;

fn weight_binding() -> ModelWeightTierBinding {
    ModelWeightTierBinding {
        logical_id: "olmoe/layer0".into(),
        generation: Generation(3),
        revision: Revision(5),
        execution_manifest: [1; 32],
        artifact_digest: [2; 32],
        model: ModelVersion::from("olmoe-v1"),
        adapter: AdapterVersion::from("base"),
        tensor_name: "layers.0.mlp.experts".into(),
        shape: vec![2, 4],
        dtype: "bf16".into(),
    }
}

fn expert(id: u32) -> ExpertWeightTierBinding {
    ExpertWeightTierBinding {
        base: weight_binding(),
        expert_id: id,
    }
}

const WEIGHTS: &[u8] = b"sixteen-byte-wts";

#[test]
fn a_weight_object_restores_to_the_exact_bytes_it_was_prepared_from() {
    let binding = weight_binding();
    let object = ModelWeightTierAdapter::prepare(WEIGHTS, &binding, 5).unwrap();
    assert!(object.chunks.len() > 1);
    assert_eq!(
        ModelWeightTierAdapter::restore(&object, &binding).unwrap(),
        WEIGHTS
    );
}

#[test]
fn restore_refuses_every_binding_field_that_prepare_bound() {
    let binding = weight_binding();
    let object = ModelWeightTierAdapter::prepare(WEIGHTS, &binding, 5).unwrap();
    let changes: Vec<BindingChange> = vec![
        Box::new(|b| b.logical_id = "olmoe/layer1".into()),
        Box::new(|b| b.generation = Generation(4)),
        Box::new(|b| b.revision = Revision(6)),
        Box::new(|b| b.execution_manifest = [9; 32]),
        Box::new(|b| b.artifact_digest = [9; 32]),
        Box::new(|b| b.model = ModelVersion::from("other")),
        Box::new(|b| b.adapter = AdapterVersion::from("lora")),
        Box::new(|b| b.tensor_name = "other".into()),
        Box::new(|b| b.shape = vec![4, 2]),
        Box::new(|b| b.dtype = "f32".into()),
    ];
    for change in changes {
        let mut other = binding.clone();
        change(&mut other);
        assert_eq!(
            ModelWeightTierAdapter::restore(&object, &other),
            Err(PodTierError::InvalidBinding)
        );
    }
}

#[test]
fn restore_refuses_a_wrong_domain_and_an_invalid_binding() {
    let binding = weight_binding();
    let object = ModelWeightTierAdapter::prepare(WEIGHTS, &binding, 5).unwrap();
    let mut wrong_domain = object.clone();
    wrong_domain.manifest = TierObjectManifest::new(
        TierObjectDomain::Artifact,
        object.manifest.logical_id.clone(),
        object.manifest.generation,
        object.manifest.revision,
        object.manifest.schema_digest,
        object.manifest.chunks.clone(),
    )
    .unwrap();
    assert_eq!(
        ModelWeightTierAdapter::restore(&wrong_domain, &binding),
        Err(PodTierError::InvalidBinding)
    );
    let mut empty = binding.clone();
    empty.shape.clear();
    assert_eq!(
        ModelWeightTierAdapter::restore(&object, &empty),
        Err(PodTierError::InvalidBinding)
    );
}

#[test]
fn restore_refuses_corrupt_or_missing_chunks_with_a_storage_error() {
    let binding = weight_binding();
    let object = ModelWeightTierAdapter::prepare(WEIGHTS, &binding, 5).unwrap();
    let mut corrupt = object.clone();
    corrupt.chunks[1][0] ^= 0xff;
    assert!(matches!(
        ModelWeightTierAdapter::restore(&corrupt, &binding),
        Err(PodTierError::Storage(_))
    ));
    let mut missing = object.clone();
    missing.chunks.pop();
    assert!(matches!(
        ModelWeightTierAdapter::restore(&missing, &binding),
        Err(PodTierError::Storage(_))
    ));
}

#[test]
fn restore_refuses_an_object_prepare_would_not_have_produced() {
    // Same bytes, same binding fields, but chunked unevenly: a valid manifest with the right
    // schema digest that `prepare` could not have built.
    let binding = weight_binding();
    let canonical = ModelWeightTierAdapter::prepare(WEIGHTS, &binding, 8).unwrap();
    let parts: [&[u8]; 2] = [&WEIGHTS[..3], &WEIGHTS[3..]];
    let mut offset = 0;
    let mut descriptors = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let descriptor = ChunkDescriptor::for_bytes(index as u32, offset, part);
        offset += descriptor.length;
        descriptors.push(descriptor);
    }
    let uneven = PreparedTierObject {
        manifest: TierObjectManifest::new(
            TierObjectDomain::ModelWeight,
            canonical.manifest.logical_id.clone(),
            canonical.manifest.generation,
            canonical.manifest.revision,
            canonical.manifest.schema_digest,
            descriptors,
        )
        .unwrap(),
        chunks: parts.iter().map(|part| part.to_vec()).collect(),
    };
    uneven.verify().unwrap();
    assert_eq!(
        ModelWeightTierAdapter::restore(&uneven, &binding),
        Err(PodTierError::InvalidBinding)
    );
}

#[test]
fn every_expert_is_its_own_tier_object_with_its_own_identity() {
    let first = ExpertWeightTierAdapter::prepare(b"expert-zero-wts!", &expert(0), 6).unwrap();
    let second = ExpertWeightTierAdapter::prepare(b"expert-one--wts!", &expert(1), 6).unwrap();
    assert_ne!(first.manifest.root_digest, second.manifest.root_digest);
    assert_ne!(first.manifest.logical_id, second.manifest.logical_id);
    assert_eq!(first.manifest.domain, TierObjectDomain::ModelWeight);
    assert_eq!(first.manifest.generation, Generation(3));
    // The same bytes under another expert id are another object.
    let same_bytes_other_id =
        ExpertWeightTierAdapter::prepare(b"expert-zero-wts!", &expert(1), 6).unwrap();
    assert_ne!(
        first.manifest.root_digest,
        same_bytes_other_id.manifest.root_digest
    );
    assert_eq!(
        ExpertWeightTierAdapter::restore(&first, &expert(0)).unwrap(),
        b"expert-zero-wts!"
    );
    assert_eq!(
        ExpertWeightTierAdapter::restore(&second, &expert(1)).unwrap(),
        b"expert-one--wts!"
    );
}

#[test]
fn an_expert_object_does_not_restore_under_another_expert_artifact_or_generation() {
    let object = ExpertWeightTierAdapter::prepare(WEIGHTS, &expert(0), 6).unwrap();
    assert_eq!(
        ExpertWeightTierAdapter::restore(&object, &expert(1)),
        Err(PodTierError::InvalidBinding)
    );
    let mut other_artifact = expert(0);
    other_artifact.base.artifact_digest = [7; 32];
    assert_eq!(
        ExpertWeightTierAdapter::restore(&object, &other_artifact),
        Err(PodTierError::InvalidBinding)
    );
    let mut other_generation = expert(0);
    other_generation.base.generation = Generation(4);
    assert_eq!(
        ExpertWeightTierAdapter::restore(&object, &other_generation),
        Err(PodTierError::InvalidBinding)
    );
    // A whole-tensor object is not an expert object, and the other way round.
    let whole = ModelWeightTierAdapter::prepare(WEIGHTS, &weight_binding(), 6).unwrap();
    assert_eq!(
        ExpertWeightTierAdapter::restore(&whole, &expert(0)),
        Err(PodTierError::InvalidBinding)
    );
    assert_eq!(
        ModelWeightTierAdapter::restore(&object, &weight_binding()),
        Err(PodTierError::InvalidBinding)
    );
}

#[test]
fn prepare_all_makes_one_object_per_expert_and_bounds_and_dedupes_them() {
    let bytes: Vec<Vec<u8>> = (0..3).map(|id| vec![id as u8 + 1; 12]).collect();
    let experts: Vec<(ExpertWeightTierBinding, &[u8])> = bytes
        .iter()
        .enumerate()
        .map(|(id, weights)| (expert(id as u32), weights.as_slice()))
        .collect();
    let objects = ExpertWeightTierAdapter::prepare_all(&experts, 4).unwrap();
    assert_eq!(objects.len(), 3);
    for (id, object) in objects.iter().enumerate() {
        assert_eq!(
            ExpertWeightTierAdapter::restore(object, &expert(id as u32)).unwrap(),
            bytes[id]
        );
    }
    let repeated = vec![
        (expert(0), bytes[0].as_slice()),
        (expert(0), bytes[1].as_slice()),
    ];
    assert_eq!(
        ExpertWeightTierAdapter::prepare_all(&repeated, 4),
        Err(PodTierError::InvalidBinding)
    );
    let too_many: Vec<(ExpertWeightTierBinding, &[u8])> = (0..=MAX_EXPERTS_PER_MODEL as u32)
        .map(|id| (expert(id), bytes[0].as_slice()))
        .collect();
    assert_eq!(
        ExpertWeightTierAdapter::prepare_all(&too_many, 4),
        Err(PodTierError::InvalidBinding)
    );
}

#[test]
fn an_expert_binding_with_an_invalid_base_is_refused() {
    let mut binding = expert(0);
    binding.base.generation = Generation(0);
    assert_eq!(
        ExpertWeightTierAdapter::prepare(WEIGHTS, &binding, 6),
        Err(PodTierError::InvalidBinding)
    );
}

#[test]
fn an_expert_id_at_or_above_the_model_bound_is_refused() {
    let at_bound = expert(MAX_EXPERTS_PER_MODEL as u32);
    assert_eq!(
        ExpertWeightTierAdapter::prepare(WEIGHTS, &at_bound, 6),
        Err(PodTierError::InvalidBinding)
    );
    let last = expert(MAX_EXPERTS_PER_MODEL as u32 - 1);
    let object = ExpertWeightTierAdapter::prepare(WEIGHTS, &last, 6).unwrap();
    assert_eq!(
        ExpertWeightTierAdapter::restore(&object, &last).unwrap(),
        WEIGHTS
    );
    assert_eq!(
        ExpertWeightTierAdapter::restore(&object, &at_bound),
        Err(PodTierError::InvalidBinding)
    );
}
