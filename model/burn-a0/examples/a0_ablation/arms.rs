//! The study's arms: which switches each turns and which batch it is fed.
//! model/configs/a0_ablation_study.toml and model/configs/ablations.toml must
//! agree with this table; scripts/tests/test_a0_ablation_config.py checks that
//! against `--phase list-arms`.

use ptr_burn_a0::PtrA0Config;

/// What the slots of a batch carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Batch {
    /// Slot i is fact i: role, epistemic state, confidence, payload, admission.
    Typed,
    /// Every slot is emptied of typed content and carries only its index; the
    /// facts reach the model only as raw tokens. `masked` keeps the admission.
    ContentFree { masked: bool },
    /// Typed slots, and every raw token replaced by PAD.
    RawBlind,
}

impl Batch {
    pub fn name(self) -> &'static str {
        match self {
            Self::Typed => "typed",
            Self::ContentFree { masked: true } => "content-free-masked",
            Self::ContentFree { masked: false } => "content-free",
            Self::RawBlind => "raw-blind",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub name: &'static str,
    pub experiment: &'static str,
    /// 1: always run. 2: dropped first by the budget rule.
    pub tier: u8,
    pub typed_attention: bool,
    pub typed_query: bool,
    pub latent_steps: usize,
    pub latent_nonlinearity: bool,
    pub frozen_router: bool,
    pub batch: Batch,
}

const FULL: Arm = Arm {
    name: "full",
    experiment: "M001",
    tier: 1,
    typed_attention: true,
    typed_query: true,
    latent_steps: 2,
    latent_nonlinearity: true,
    frozen_router: false,
    batch: Batch::Typed,
};

pub const ARMS: [Arm; 12] = [
    FULL,
    Arm {
        name: "no-semantic-slots",
        batch: Batch::ContentFree { masked: false },
        ..FULL
    },
    Arm {
        name: "no-semantic-slots-masked",
        batch: Batch::ContentFree { masked: true },
        ..FULL
    },
    Arm {
        name: "no-typed-attention",
        experiment: "M002",
        typed_attention: false,
        ..FULL
    },
    Arm {
        name: "raw-blind",
        experiment: "M002",
        batch: Batch::RawBlind,
        ..FULL
    },
    Arm {
        name: "blind-query-k0",
        experiment: "M002",
        tier: 2,
        typed_query: false,
        latent_steps: 0,
        ..FULL
    },
    Arm {
        name: "blind-query-k0-no-typed-attention",
        experiment: "M002",
        tier: 2,
        typed_query: false,
        latent_steps: 0,
        typed_attention: false,
        ..FULL
    },
    Arm {
        name: "latent-0",
        experiment: "M003",
        latent_steps: 0,
        ..FULL
    },
    Arm {
        name: "latent-1",
        experiment: "M003",
        latent_steps: 1,
        ..FULL
    },
    Arm {
        name: "latent-linear",
        experiment: "M003",
        latent_nonlinearity: false,
        ..FULL
    },
    Arm {
        name: "latent-4",
        experiment: "M003",
        tier: 2,
        latent_steps: 4,
        ..FULL
    },
    Arm {
        name: "frozen-router",
        experiment: "M004",
        frozen_router: true,
        ..FULL
    },
];

pub fn find(name: &str) -> Result<Arm, String> {
    ARMS.iter()
        .copied()
        .find(|arm| arm.name == name)
        .ok_or_else(|| format!("unknown arm {name:?}"))
}

pub const VOCABULARY: usize = 216;
pub const PROVENANCE_BUCKETS: usize = 8;

impl Arm {
    pub fn config(&self, d_model: usize) -> PtrA0Config {
        PtrA0Config::new(VOCABULARY, d_model)
            .with_provenance_buckets(PROVENANCE_BUCKETS)
            .with_latent_steps(self.latent_steps)
            .with_typed_attention(self.typed_attention)
            .with_typed_query(self.typed_query)
            .with_latent_nonlinearity(self.latent_nonlinearity)
            .with_frozen_router(self.frozen_router)
    }

    fn on(flag: bool) -> &'static str {
        if flag {
            "on"
        } else {
            "off"
        }
    }

    /// The arm as one JSON object; its string fields name it in a row.
    pub fn json(&self) -> String {
        format!(
            r#"{{"arm":"{}","experiment":"{}","tier":"{}","typed_attention":"{}","typed_query":"{}","latent_steps":"{}","latent_nonlinearity":"{}","frozen_router":"{}","batch":"{}"}}"#,
            self.name,
            self.experiment,
            self.tier,
            Self::on(self.typed_attention),
            Self::on(self.typed_query),
            self.latent_steps,
            Self::on(self.latent_nonlinearity),
            Self::on(self.frozen_router),
            self.batch.name(),
        )
    }
}
