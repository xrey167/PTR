//! Commit-bound construction and checkpoint binding for an M002-v5 backend.
//!
//! A TOML/JSON configuration that merely repeats an expected digest proves
//! nothing: the model has to consume the same bytes while it is built and while
//! it writes or reads weights.  This small, dependency-free wire format is the
//! artifact an eventual M009 command must pass to [`M002V5Contract::read`].

use crate::{PtrA0, PtrA0Config, RouterMode, TypedAttentionMode};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

const PREFIX: &str = "M002-V5-CONTRACT/1";

/// A failure to bind a learned backend to the frozen FactorizedV2 contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum M002V5ContractError {
    Read,
    Digest,
    Utf8,
    Syntax,
    MissingField,
    DuplicateField,
    Value,
    ConfigMismatch,
    TrainingMismatch,
}

impl core::fmt::Display for M002V5ContractError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let code = match self {
            Self::Read => "PTR_A0_M002V5_CONTRACT_READ",
            Self::Digest => "PTR_A0_M002V5_CONTRACT_DIGEST",
            Self::Utf8 => "PTR_A0_M002V5_CONTRACT_UTF8",
            Self::Syntax => "PTR_A0_M002V5_CONTRACT_SYNTAX",
            Self::MissingField => "PTR_A0_M002V5_CONTRACT_MISSING",
            Self::DuplicateField => "PTR_A0_M002V5_CONTRACT_DUPLICATE",
            Self::Value => "PTR_A0_M002V5_CONTRACT_VALUE",
            Self::ConfigMismatch => "PTR_A0_M002V5_CONTRACT_CONFIG",
            Self::TrainingMismatch => "PTR_A0_M002V5_CONTRACT_TRAINING",
        };
        formatter.write_str(code)
    }
}

impl std::error::Error for M002V5ContractError {}

/// The exact selected architecture and repaired-router training constants.
#[derive(Clone, Debug, PartialEq)]
pub struct M002V5Contract {
    digest: [u8; 32],
    d_model: usize,
    rank: usize,
    bias_limit: f32,
    metadata_dropout: f32,
}

impl M002V5Contract {
    /// Read bytes and reject them unless their supplied SHA-256 is exact.
    pub fn read(
        path: impl AsRef<Path>,
        expected_sha256: &str,
    ) -> Result<Self, M002V5ContractError> {
        let bytes = fs::read(path).map_err(|_| M002V5ContractError::Read)?;
        Self::from_bytes(&bytes, expected_sha256)
    }

    /// Parse a strictly finite key/value contract after its digest is checked.
    pub fn from_bytes(bytes: &[u8], expected_sha256: &str) -> Result<Self, M002V5ContractError> {
        let digest = parse_digest(expected_sha256)?;
        if Sha256::digest(bytes).as_slice() != digest {
            return Err(M002V5ContractError::Digest);
        }
        let text = core::str::from_utf8(bytes).map_err(|_| M002V5ContractError::Utf8)?;
        let mut lines = text.lines();
        if lines.next() != Some(PREFIX) {
            return Err(M002V5ContractError::Syntax);
        }
        let mut values = BTreeMap::new();
        for line in lines {
            let (key, value) = line.split_once('=').ok_or(M002V5ContractError::Syntax)?;
            if key.is_empty() || value.is_empty() || values.insert(key, value).is_some() {
                return Err(M002V5ContractError::DuplicateField);
            }
        }
        let exact = [
            "attention_mode",
            "d_model",
            "rank",
            "bias_limit",
            "metadata_dropout",
            "router_mode",
            "router_logit_scale",
            "label_smoothing",
            "consistency_weight",
            "latent_steps",
            "typed_query",
            "latent_nonlinearity",
        ];
        if values.len() != exact.len() || exact.iter().any(|key| !values.contains_key(key)) {
            return Err(M002V5ContractError::MissingField);
        }
        let value = |key| {
            values
                .get(key)
                .copied()
                .ok_or(M002V5ContractError::MissingField)
        };
        if value("attention_mode")? != "factorized-v2"
            || value("router_mode")? != "calibrated-cosine-v2"
            || value("typed_query")? != "true"
            || value("latent_nonlinearity")? != "true"
            || value("latent_steps")? != "2"
        {
            return Err(M002V5ContractError::Value);
        }
        let d_model = value("d_model")?
            .parse()
            .map_err(|_| M002V5ContractError::Value)?;
        let rank = value("rank")?
            .parse()
            .map_err(|_| M002V5ContractError::Value)?;
        let bias_limit = finite(value("bias_limit")?)?;
        let metadata_dropout = finite(value("metadata_dropout")?)?;
        if d_model != 48
            || rank == 0
            || rank > d_model
            || bias_limit <= 0.0
            || !(0.0..=1.0).contains(&metadata_dropout)
        {
            return Err(M002V5ContractError::Value);
        }
        for (key, expected) in [
            ("router_logit_scale", 5.0),
            ("label_smoothing", 0.05),
            ("consistency_weight", 0.10),
        ] {
            if finite(value(key)?)? != expected {
                return Err(M002V5ContractError::Value);
            }
        }
        Ok(Self {
            digest,
            d_model,
            rank,
            bias_limit,
            metadata_dropout,
        })
    }

    /// SHA-256 of the exact contract artifact, carried into the checkpoint.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// The only A0 configuration a learned M009 backend may construct from
    /// this contract.  Callers do not receive an unbound rank/mode builder.
    pub fn config(&self, vocabulary: usize) -> PtrA0Config {
        PtrA0Config::new(vocabulary, self.d_model)
            .with_latent_steps(2)
            .with_typed_attention_mode(TypedAttentionMode::FactorizedV2)
            .with_factorized_attention(self.rank, self.bias_limit)
            .with_typed_query(true)
            .with_latent_nonlinearity(true)
            .with_router_mode(RouterMode::CalibratedCosineV2)
            .with_router_logit_scale(5.0)
    }

    /// Construct only the FactorizedV2 graph the contract describes.
    pub fn init(
        &self,
        config: &PtrA0Config,
        device: &crate::Device,
    ) -> Result<PtrA0, M002V5ContractError> {
        self.verify_config(config)?;
        Ok(config.init_lazy(device))
    }

    /// Refuse a config that names the digest but builds another graph.
    pub fn verify_config(&self, config: &PtrA0Config) -> Result<(), M002V5ContractError> {
        if config.d_model != self.d_model
            || config.typed_attention != TypedAttentionMode::FactorizedV2
            || config.typed_attention_rank != self.rank
            || config.typed_attention_limit.to_bits() != self.bias_limit.to_bits()
            || !config.typed_query
            || config.latent_steps != 2
            || !config.latent_nonlinearity
            || config.router_mode != RouterMode::CalibratedCosineV2
            || config.router_logit_scale.to_bits() != 5.0_f32.to_bits()
        {
            return Err(M002V5ContractError::ConfigMismatch);
        }
        Ok(())
    }

    /// Bind training-only repaired-router constants that do not live in `PtrA0Config`.
    pub fn verify_training(
        &self,
        metadata_dropout: f32,
        label_smoothing: f32,
        consistency_weight: f32,
    ) -> Result<(), M002V5ContractError> {
        if metadata_dropout.to_bits() != self.metadata_dropout.to_bits()
            || label_smoothing.to_bits() != 0.05_f32.to_bits()
            || consistency_weight.to_bits() != 0.10_f32.to_bits()
        {
            return Err(M002V5ContractError::TrainingMismatch);
        }
        Ok(())
    }
}

fn finite(value: &str) -> Result<f32, M002V5ContractError> {
    let parsed: f32 = value.parse().map_err(|_| M002V5ContractError::Value)?;
    parsed
        .is_finite()
        .then_some(parsed)
        .ok_or(M002V5ContractError::Value)
}

fn parse_digest(value: &str) -> Result<[u8; 32], M002V5ContractError> {
    if value.len() != 64 {
        return Err(M002V5ContractError::Digest);
    }
    let mut result = [0; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| M002V5ContractError::Digest)?;
    }
    Ok(result)
}
