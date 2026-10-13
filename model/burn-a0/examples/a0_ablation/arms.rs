//! The study's arms: which switches each turns and which batch it is fed.
//! model/configs/a0_ablation_study.toml and model/configs/ablations.toml must
//! agree with this table; scripts/tests/test_a0_ablation_config.py checks that
//! against `--phase list-arms`.

use ptr_burn_a0::{PtrA0Config, RouterMode, TypedAttentionMode};

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
    /// Return the batch variant's name for study output rows.
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
    pub typed_attention_mode: TypedAttentionMode,
    pub typed_attention_rank: usize,
    pub typed_attention_limit: f32,
    pub typed_query: bool,
    pub latent_steps: usize,
    pub latent_nonlinearity: bool,
    pub frozen_router: bool,
    pub router_mode: RouterMode,
    pub router_logit_scale: f32,
    pub label_smoothing: f32,
    pub metadata_dropout: f32,
    pub consistency_weight: f32,
    pub batch: Batch,
}

const FULL: Arm = Arm {
    name: "full",
    experiment: "M001",
    tier: 1,
    typed_attention: true,
    typed_attention_mode: TypedAttentionMode::LegacyScalarV1,
    typed_attention_rank: 16,
    typed_attention_limit: 2.0,
    typed_query: true,
    latent_steps: 2,
    latent_nonlinearity: true,
    frozen_router: false,
    router_mode: RouterMode::LegacyMeanLinear,
    router_logit_scale: 5.0,
    label_smoothing: 0.0,
    metadata_dropout: 0.0,
    consistency_weight: 0.0,
    batch: Batch::Typed,
};

/// Historical A0-v1 arms. Keep this table and its list-arms wire format stable;
/// M002-v5 is a separate study below.
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
        typed_attention_mode: TypedAttentionMode::Off,
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
        typed_attention_mode: TypedAttentionMode::Off,
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

pub const V5_ARMS: [Arm; 2] = [
    Arm {
        name: "factorized-v2",
        experiment: "M002-v5",
        typed_attention: true,
        typed_attention_mode: TypedAttentionMode::FactorizedV2,
        router_mode: RouterMode::CalibratedCosineV2,
        label_smoothing: 0.05,
        metadata_dropout: 0.10,
        consistency_weight: 0.10,
        ..FULL
    },
    Arm {
        name: "factorized-v2-off",
        experiment: "M002-v5",
        typed_attention: false,
        typed_attention_mode: TypedAttentionMode::Off,
        router_mode: RouterMode::CalibratedCosineV2,
        label_smoothing: 0.05,
        metadata_dropout: 0.10,
        consistency_weight: 0.10,
        ..FULL
    },
];

/// The v7 successor has a distinct runner identity.  The model switches are
/// deliberately identical to v5's selected pair, but a new study must not
/// borrow v5's arm names: `arms_of` rejects cross-study selection before any
/// dataset is read or training begins.
pub const V7_ARMS: [Arm; 2] = [
    Arm {
        name: "factorized-v2-v7",
        experiment: "M002-v7",
        typed_attention: true,
        typed_attention_mode: TypedAttentionMode::FactorizedV2,
        router_mode: RouterMode::CalibratedCosineV2,
        label_smoothing: 0.05,
        metadata_dropout: 0.10,
        consistency_weight: 0.10,
        ..FULL
    },
    Arm {
        name: "factorized-v2-off-v7",
        experiment: "M002-v7",
        typed_attention: false,
        typed_attention_mode: TypedAttentionMode::Off,
        router_mode: RouterMode::CalibratedCosineV2,
        label_smoothing: 0.05,
        metadata_dropout: 0.10,
        consistency_weight: 0.10,
        ..FULL
    },
];

/// M002-v8 is the first successor whose complete runner and source-tree
/// binding is part of its preregistration before any execution attempt.
pub const V8_ARMS: [Arm; 2] = [
    Arm {
        name: "factorized-v2-v8",
        experiment: "M002-v8",
        typed_attention: true,
        typed_attention_mode: TypedAttentionMode::FactorizedV2,
        router_mode: RouterMode::CalibratedCosineV2,
        label_smoothing: 0.05,
        metadata_dropout: 0.10,
        consistency_weight: 0.10,
        ..FULL
    },
    Arm {
        name: "factorized-v2-off-v8",
        experiment: "M002-v8",
        typed_attention: false,
        typed_attention_mode: TypedAttentionMode::Off,
        router_mode: RouterMode::CalibratedCosineV2,
        label_smoothing: 0.05,
        metadata_dropout: 0.10,
        consistency_weight: 0.10,
        ..FULL
    },
];

/// M002-v9 uses a separately registered pair after the complete v8 gate
/// implementation was independently exercised before this study is enrolled.
pub const V9_ARMS: [Arm; 2] = [
    Arm {
        name: "factorized-v2-v9",
        experiment: "M002-v9",
        typed_attention: true,
        typed_attention_mode: TypedAttentionMode::FactorizedV2,
        router_mode: RouterMode::CalibratedCosineV2,
        label_smoothing: 0.05,
        metadata_dropout: 0.10,
        consistency_weight: 0.10,
        ..FULL
    },
    Arm {
        name: "factorized-v2-off-v9",
        experiment: "M002-v9",
        typed_attention: false,
        typed_attention_mode: TypedAttentionMode::Off,
        router_mode: RouterMode::CalibratedCosineV2,
        label_smoothing: 0.05,
        metadata_dropout: 0.10,
        consistency_weight: 0.10,
        ..FULL
    },
];

/// The sole ordered matched pair admitted by each FactorizedV2 study.
pub fn paired_names(experiment: &str) -> Option<[&'static str; 2]> {
    match experiment {
        "M002-v5" => Some(["factorized-v2", "factorized-v2-off"]),
        "M002-v7" => Some(["factorized-v2-v7", "factorized-v2-off-v7"]),
        "M002-v8" => Some(["factorized-v2-v8", "factorized-v2-off-v8"]),
        "M002-v9" => Some(["factorized-v2-v9", "factorized-v2-off-v9"]),
        _ => None,
    }
}

/// Look up an arm by its exact study name, returning an error if it is unknown.
pub fn find(name: &str) -> Result<Arm, String> {
    ARMS.iter()
        .chain(V5_ARMS.iter())
        .chain(V7_ARMS.iter())
        .chain(V8_ARMS.iter())
        .chain(V9_ARMS.iter())
        .copied()
        .find(|arm| arm.name == name)
        .ok_or_else(|| format!("unknown arm {name:?}"))
}

pub const VOCABULARY: usize = 216;
pub const PROVENANCE_BUCKETS: usize = 8;

impl Arm {
    /// Configure A0 at width `d_model` with this arm's switches and input tables.
    pub fn config(&self, d_model: usize) -> PtrA0Config {
        PtrA0Config::new(VOCABULARY, d_model)
            .with_provenance_buckets(PROVENANCE_BUCKETS)
            .with_latent_steps(self.latent_steps)
            .with_typed_attention_mode(self.typed_attention_mode)
            .with_factorized_attention(
                self.typed_attention_rank.min(d_model),
                self.typed_attention_limit,
            )
            .with_typed_query(self.typed_query)
            .with_latent_nonlinearity(self.latent_nonlinearity)
            .with_router_mode(self.router_mode)
            .with_router_logit_scale(self.router_logit_scale)
            .with_frozen_router(self.frozen_router)
    }

    /// Encode a switch as the string label used in study rows.
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

/// Name of the plain-transformer arm in the baseline-v1 studies. It is not an
/// `Arm` (it has no A0 switches); every study table appends it implicitly.
pub const PLAIN_ARM: &str = "plain";

/// A baseline-v1 study. The arm tables below are scoped to the study value, not
/// to a per-experiment name suffix, so a successor study is a new variant and
/// cannot silently borrow (or lose) another study's arm names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Study {
    M001V5,
    M002V10,
    M003V2,
}

/// Typed failure of study and arm resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StudyError {
    UnknownStudy(String),
    UnknownArm { study: &'static str, arm: String },
    DuplicateArm { study: &'static str, arm: String },
    EmptySelection { study: &'static str },
}

impl std::fmt::Display for StudyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownStudy(name) => write!(
                f,
                "unknown study {name:?}; expected one of {}",
                Study::ALL.map(Study::id).join(", ")
            ),
            Self::UnknownArm { study, arm } => write!(
                f,
                "arm {arm:?} is not in the {study} table (arms: {})",
                Study::from_id(study)
                    .map(|s| s.arm_names().join(", "))
                    .unwrap_or_default()
            ),
            Self::DuplicateArm { study, arm } => {
                write!(f, "arm {arm:?} selected twice for {study}")
            }
            Self::EmptySelection { study } => write!(f, "no arms selected for {study}"),
        }
    }
}

/// Calibrated-router base of every baseline-v1 A0 arm.
const BASE: Arm = Arm {
    router_mode: RouterMode::CalibratedCosineV2,
    ..FULL
};

/// Regularised FactorizedV2 settings of the M002-v9 pair, kept so the isolated
/// on/off contrast trains exactly as the last registered pair did.
const FACTORIZED: Arm = Arm {
    typed_attention: true,
    typed_attention_mode: TypedAttentionMode::FactorizedV2,
    label_smoothing: 0.05,
    metadata_dropout: 0.10,
    consistency_weight: 0.10,
    ..BASE
};

impl Study {
    pub const ALL: [Study; 3] = [Study::M001V5, Study::M002V10, Study::M003V2];

    /// Registry identifier, also used as the `experiment` of every arm of the study.
    pub const fn id(self) -> &'static str {
        match self {
            Self::M001V5 => "m001-v5",
            Self::M002V10 => "m002-v10",
            Self::M003V2 => "m003-v2",
        }
    }

    pub fn from_id(id: &str) -> Result<Self, StudyError> {
        Self::ALL
            .into_iter()
            .find(|study| study.id() == id)
            .ok_or_else(|| StudyError::UnknownStudy(id.to_owned()))
    }

    /// A0 arm names of the study in report order, then [`PLAIN_ARM`] is implicit.
    /// `no-semantic-slots` (M001-v5) is the unmasked diagnostic arm that makes the
    /// A0-v2 mask share (masked - unmasked accuracy) computable; no gate uses it.
    pub const fn arm_names(self) -> &'static [&'static str] {
        match self {
            Self::M001V5 => &["full", "no-semantic-slots-masked", "no-semantic-slots"],
            Self::M002V10 => &["factorized-v2", "factorized-v2-off", "full"],
            Self::M003V2 => &["full", "latent-0", "latent-1", "latent-linear"],
        }
    }

    /// Resolve one A0 arm of this study; `plain` and foreign names are errors.
    pub fn arm(self, name: &str) -> Result<Arm, StudyError> {
        let base = match (self, name) {
            (_, "full") => BASE,
            (Self::M001V5, "no-semantic-slots-masked") => Arm {
                name: "no-semantic-slots-masked",
                batch: Batch::ContentFree { masked: true },
                ..BASE
            },
            (Self::M001V5, "no-semantic-slots") => Arm {
                name: "no-semantic-slots",
                batch: Batch::ContentFree { masked: false },
                ..BASE
            },
            (Self::M002V10, "factorized-v2") => Arm {
                name: "factorized-v2",
                ..FACTORIZED
            },
            (Self::M002V10, "factorized-v2-off") => Arm {
                name: "factorized-v2-off",
                typed_attention: false,
                typed_attention_mode: TypedAttentionMode::Off,
                ..FACTORIZED
            },
            (Self::M003V2, "latent-0") => Arm {
                name: "latent-0",
                latent_steps: 0,
                ..BASE
            },
            (Self::M003V2, "latent-1") => Arm {
                name: "latent-1",
                latent_steps: 1,
                ..BASE
            },
            // Affine-only recurrence: the same `latent_refine` layer, so the
            // parameter count is that of `full`, without the GELU.
            (Self::M003V2, "latent-linear") => Arm {
                name: "latent-linear",
                latent_nonlinearity: false,
                ..BASE
            },
            _ => {
                return Err(StudyError::UnknownArm {
                    study: self.id(),
                    arm: name.to_owned(),
                })
            }
        };
        Ok(Arm {
            experiment: self.id(),
            ..base
        })
    }

    /// All A0 arms of the study in report order.
    pub fn arms(self) -> Vec<Arm> {
        self.arm_names()
            .iter()
            .map(|name| self.arm(name).expect("the study's own names resolve"))
            .collect()
    }

    /// Resolve an explicit selection. The literal `plain` is split off (returned
    /// as a flag); duplicates, foreign names and an empty selection are errors.
    pub fn select(self, names: &[&str]) -> Result<(Vec<Arm>, bool), StudyError> {
        let mut seen: Vec<&str> = Vec::new();
        let mut arms = Vec::new();
        let mut plain = false;
        for name in names {
            if seen.contains(name) {
                return Err(StudyError::DuplicateArm {
                    study: self.id(),
                    arm: (*name).to_owned(),
                });
            }
            seen.push(name);
            if *name == PLAIN_ARM {
                plain = true;
            } else {
                arms.push(self.arm(name)?);
            }
        }
        if arms.is_empty() && !plain {
            return Err(StudyError::EmptySelection { study: self.id() });
        }
        Ok((arms, plain))
    }
}
