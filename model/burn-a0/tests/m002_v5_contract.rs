use burn::prelude::*;
use ptr_burn_a0::{
    load_bound, save_bound, CheckpointIoError, M002V5Contract, M002V5ContractError, PtrA0Config,
    RouterMode, TypedAttentionMode,
};
use sha2::{Digest, Sha256};

fn bytes(dropout: &str) -> Vec<u8> {
    format!(
        "M002-V5-CONTRACT/1\nattention_mode=factorized-v2\nd_model=48\nrank=16\nbias_limit=2\nmetadata_dropout={dropout}\nrouter_mode=calibrated-cosine-v2\nrouter_logit_scale=5\nlabel_smoothing=0.05\nconsistency_weight=0.1\nlatent_steps=2\ntyped_query=true\nlatent_nonlinearity=true\n"
    )
    .into_bytes()
}

fn contract(dropout: &str) -> M002V5Contract {
    let bytes = bytes(dropout);
    let digest = format!("{:x}", Sha256::digest(&bytes));
    M002V5Contract::from_bytes(&bytes, &digest).expect("valid frozen contract")
}

fn config() -> PtrA0Config {
    PtrA0Config::new(16, 48)
        .with_latent_steps(2)
        .with_typed_attention_mode(TypedAttentionMode::FactorizedV2)
        .with_factorized_attention(16, 2.0)
        .with_typed_query(true)
        .with_latent_nonlinearity(true)
        .with_router_mode(RouterMode::CalibratedCosineV2)
        .with_router_logit_scale(5.0)
}

#[test]
fn contract_constructs_and_roundtrips_only_the_bound_architecture() {
    let device = Device::default();
    let contract = contract("0.1");
    let selected_config = config();
    contract
        .verify_training(0.1, 0.05, 0.10)
        .expect("training constants are frozen");
    let model = contract
        .init(&selected_config, &device)
        .expect("contract constructs model");
    let saved = save_bound(&model, &selected_config, &contract).expect("bound checkpoint writes");
    load_bound(&saved, &selected_config, &contract, &device).expect("same contract loads");
}

#[test]
fn self_asserted_digest_cannot_hide_a_different_runtime_or_checkpoint_contract() {
    let device = Device::default();
    let selected = contract("0.1");
    let selected_config = config();
    let model = selected
        .init(&selected_config, &device)
        .expect("selected model");
    let saved = save_bound(&model, &selected_config, &selected).expect("bound checkpoint");

    let wrong_config = config().with_factorized_attention(8, 2.0);
    assert_eq!(
        selected
            .init(&wrong_config, &device)
            .expect_err("rank differs"),
        M002V5ContractError::ConfigMismatch,
    );
    let different_training_contract = contract("0.0");
    assert_eq!(
        load_bound(
            &saved,
            &selected_config,
            &different_training_contract,
            &device
        )
        .expect_err("different digest cannot load"),
        CheckpointIoError::ArchitectureMismatch,
    );
}
