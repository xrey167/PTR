//! Validated configuration of the CPU reference forward pass.

use std::fmt;

/// Largest accepted model width.
pub const MAX_D_MODEL: usize = 512;
/// Largest accepted feed-forward width.
pub const MAX_D_FF: usize = 2048;
/// Largest accepted slot count (Invariant 8: bounded workspace).
pub const MAX_SLOTS: usize = 1024;
/// Largest accepted recurrence depth.
pub const MAX_LATENT_STEPS: u32 = 64;
/// Largest accepted PTR-Diff vocabulary (including the mask token).
pub const MAX_DIFF_VOCAB: usize = 4096;
/// Largest accepted head count.
pub const MAX_HEADS: usize = 64;

/// Why a [`CoreConfig`] was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigError {
    /// A field that must be positive is zero.
    Zero { field: &'static str },
    /// The head count does not divide the model width.
    HeadsDoNotDivide { d_model: usize, heads: usize },
    /// A field exceeds its documented bound.
    TooLarge {
        field: &'static str,
        value: usize,
        max: usize,
    },
    /// The diffusion vocabulary needs at least one real token plus the mask token.
    VocabTooSmall { vocab: usize },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero { field } => write!(f, "{field} must be positive"),
            Self::HeadsDoNotDivide { d_model, heads } => {
                write!(f, "heads {heads} do not divide d_model {d_model}")
            }
            Self::TooLarge { field, value, max } => write!(f, "{field} {value} exceeds {max}"),
            Self::VocabTooSmall { vocab } => write!(f, "diff vocab {vocab} < 2"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Dimensions and options of the reference model.
///
/// Distinct from the older [`crate::PtrCoreConfig`] research descriptor, which
/// sizes the 512-wide A0 target; this one sizes the CPU reference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoreConfig {
    /// Slot / hidden width.
    pub d_model: usize,
    /// Attention heads; must divide `d_model`.
    pub heads: usize,
    /// Feed-forward width.
    pub d_ff: usize,
    /// Maximum number of slots (or canvas positions) per forward pass.
    pub max_slots: usize,
    /// Hard cap on recurrence steps, applied on top of any `ReasoningBudget`.
    pub max_latent_steps: u32,
    /// PTR-Diff vocabulary size; the last id is the mask token.
    pub diff_vocab: usize,
    /// Whether the learned per-(role, role) attention bias is applied.
    pub role_bias: bool,
    /// Initialiser seed. Not part of the architecture binding.
    pub seed: u64,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            heads: 4,
            d_ff: 64,
            max_slots: 64,
            max_latent_steps: 8,
            diff_vocab: 32,
            role_bias: true,
            seed: 0x5054_525f_434f_5245,
        }
    }
}

impl CoreConfig {
    /// Check every bound; a config that passes can be instantiated.
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (field, value) in [
            ("d_model", self.d_model),
            ("heads", self.heads),
            ("d_ff", self.d_ff),
            ("max_slots", self.max_slots),
            ("diff_vocab", self.diff_vocab),
        ] {
            if value == 0 {
                return Err(ConfigError::Zero { field });
            }
        }
        if self.max_latent_steps == 0 {
            return Err(ConfigError::Zero {
                field: "max_latent_steps",
            });
        }
        for (field, value, max) in [
            ("d_model", self.d_model, MAX_D_MODEL),
            ("heads", self.heads, MAX_HEADS),
            ("d_ff", self.d_ff, MAX_D_FF),
            ("max_slots", self.max_slots, MAX_SLOTS),
            ("diff_vocab", self.diff_vocab, MAX_DIFF_VOCAB),
            (
                "max_latent_steps",
                self.max_latent_steps as usize,
                MAX_LATENT_STEPS as usize,
            ),
        ] {
            if value > max {
                return Err(ConfigError::TooLarge { field, value, max });
            }
        }
        if !self.d_model.is_multiple_of(self.heads) {
            return Err(ConfigError::HeadsDoNotDivide {
                d_model: self.d_model,
                heads: self.heads,
            });
        }
        if self.diff_vocab < 2 {
            return Err(ConfigError::VocabTooSmall {
                vocab: self.diff_vocab,
            });
        }
        Ok(())
    }

    /// Canonical little-endian bytes of the architecture-defining fields.
    ///
    /// Stored in the checkpoint header's architecture binding. The seed and the
    /// runtime recurrence cap are deliberately excluded: neither changes any
    /// parameter shape.
    pub fn architecture_bytes(&self) -> Vec<u8> {
        let mut out = b"ptr-core-ref/arch/v1".to_vec();
        for v in [
            self.d_model,
            self.heads,
            self.d_ff,
            self.max_slots,
            self.diff_vocab,
        ] {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        out.push(u8::from(self.role_bias));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_validates() {
        assert_eq!(CoreConfig::default().validate(), Ok(()));
    }

    #[test]
    fn every_refusal_is_typed() {
        let base = CoreConfig::default();
        let bad = |f: fn(&mut CoreConfig)| {
            let mut c = base.clone();
            f(&mut c);
            c.validate()
        };
        assert_eq!(
            bad(|c| c.d_model = 0),
            Err(ConfigError::Zero { field: "d_model" })
        );
        assert_eq!(
            bad(|c| c.max_latent_steps = 0),
            Err(ConfigError::Zero {
                field: "max_latent_steps"
            })
        );
        assert_eq!(
            bad(|c| c.heads = 5),
            Err(ConfigError::HeadsDoNotDivide {
                d_model: 32,
                heads: 5
            })
        );
        assert!(matches!(
            bad(|c| c.max_slots = MAX_SLOTS + 1),
            Err(ConfigError::TooLarge {
                field: "max_slots",
                ..
            })
        ));
        assert!(matches!(
            bad(|c| c.max_latent_steps = MAX_LATENT_STEPS + 1),
            Err(ConfigError::TooLarge { .. })
        ));
        assert_eq!(
            bad(|c| c.diff_vocab = 1),
            Err(ConfigError::VocabTooSmall { vocab: 1 })
        );
    }

    #[test]
    fn architecture_bytes_ignore_seed_but_not_shape() {
        let a = CoreConfig::default();
        let mut b = a.clone();
        b.seed ^= 1;
        assert_eq!(a.architecture_bytes(), b.architecture_bytes());
        b.d_ff += 1;
        assert_ne!(a.architecture_bytes(), b.architecture_bytes());
        b = a.clone();
        b.role_bias = false;
        assert_ne!(a.architecture_bytes(), b.architecture_bytes());
    }
}
