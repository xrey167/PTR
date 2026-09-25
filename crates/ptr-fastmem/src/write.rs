use ptr_types::Generation;

use crate::config::FastMemoryConfig;
use crate::error::FastMemoryError;

/// Position of a write in one memory's journal. The first write is `1`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WriteSeq(pub u64);

/// The semantic input a write was derived from.
///
/// This is what makes the memory revocable: a write is attributable to exactly
/// one lifecycle-managed input at exactly one generation, so when that
/// generation is revoked or superseded the writes it produced can be found and
/// removed.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceRef {
    /// A semantic key: a capsule id, `constraint:<key>` or `procedure:<id>`, the
    /// same target vocabulary a neural-state declaration uses.
    pub key: String,
    pub generation: Generation,
    /// Digest of the exact input value the write consumed (the canonical
    /// journal encoding of the key and value, as neural-state admission
    /// digests it). A write stays admissible only while the input still has
    /// this digest at this generation: an edit or a removal of the input
    /// excludes the write even when no generation changed.
    pub input_digest: [u8; 32],
}

/// How the existing state fades before a write lands.
///
/// `Scalar` is the Gated DeltaNet form (one factor for the whole head) and the
/// default for an untrained memory. `PerChannel` is the Kimi Delta Attention
/// form (one factor per key channel, `heads * key_dim` values); it is only
/// meaningful when key channels mean something, which random projections do
/// not, so it is an ablation rather than a default. `None` is plain DeltaNet:
/// with no factor below one the state is non-expansive but not bounded, so a
/// long-lived memory should decay. Factors are journaled and must be pure
/// functions of the write itself (its source, time and a pinned configuration),
/// never of the state: a state-dependent gate would make a refold without a
/// revoked write replay gates that no longer apply.
#[derive(Clone, Debug, PartialEq)]
pub enum Decay {
    None,
    Scalar(f32),
    PerChannel(Vec<f32>),
}

/// An unvalidated write as a caller composes it.
#[derive(Clone, Debug, PartialEq)]
pub struct WriteRequest {
    pub source: SourceRef,
    /// `heads * key_dim` values; each head's slice is normalised on admission.
    pub key: Vec<f32>,
    /// `heads * value_dim` values.
    pub value: Vec<f32>,
    /// Write strength in `(0, 1]`. At `1` the value previously associated with
    /// this exact key is replaced outright.
    pub beta: f32,
    pub decay: Decay,
}

/// A validated write: finite, correctly shaped, unit keys per head.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryWrite {
    seq: WriteSeq,
    source: SourceRef,
    key: Vec<f32>,
    value: Vec<f32>,
    beta: f32,
    decay: Decay,
}

impl MemoryWrite {
    pub fn seq(&self) -> WriteSeq {
        self.seq
    }
    pub fn source(&self) -> &SourceRef {
        &self.source
    }
    pub fn key(&self) -> &[f32] {
        &self.key
    }
    pub fn value(&self) -> &[f32] {
        &self.value
    }
    pub fn beta(&self) -> f32 {
        self.beta
    }
    pub fn decay(&self) -> &Decay {
        &self.decay
    }
}

/// A validated query: one unit vector per head.
#[derive(Clone, Debug, PartialEq)]
pub struct Query {
    key: Vec<f32>,
}

impl Query {
    /// Normalise a raw query against `config`.
    pub fn new(config: &FastMemoryConfig, raw: Vec<f32>) -> Result<Self, FastMemoryError> {
        let key = normalized_heads(config, "query", raw)?;
        Ok(Self { key })
    }

    pub fn key(&self) -> &[f32] {
        &self.key
    }
}

pub(crate) fn admit_write(
    config: &FastMemoryConfig,
    seq: WriteSeq,
    request: WriteRequest,
) -> Result<MemoryWrite, FastMemoryError> {
    let WriteRequest {
        source,
        key,
        value,
        beta,
        decay,
    } = request;
    if !(beta.is_finite() && beta > 0.0 && beta <= 1.0) {
        return Err(FastMemoryError::InvalidBeta { value: beta });
    }
    check_len("value", config.value_len(), value.len())?;
    check_finite("value", &value)?;
    match &decay {
        Decay::None => {}
        Decay::Scalar(factor) => check_decay(0, *factor)?,
        Decay::PerChannel(factors) => {
            check_len("decay", config.key_len(), factors.len())?;
            for (index, factor) in factors.iter().enumerate() {
                check_decay(index, *factor)?;
            }
        }
    }
    let key = normalized_heads(config, "key", key)?;
    Ok(MemoryWrite {
        seq,
        source,
        key,
        value,
        beta,
        decay,
    })
}

fn normalized_heads(
    config: &FastMemoryConfig,
    field: &'static str,
    mut raw: Vec<f32>,
) -> Result<Vec<f32>, FastMemoryError> {
    check_len(field, config.key_len(), raw.len())?;
    check_finite(field, &raw)?;
    for (head, chunk) in raw.chunks_mut(config.key_dim).enumerate() {
        let norm = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(FastMemoryError::ZeroKey { head });
        }
        for x in chunk.iter_mut() {
            *x /= norm;
        }
    }
    Ok(raw)
}

fn check_len(field: &'static str, expected: usize, actual: usize) -> Result<(), FastMemoryError> {
    if expected != actual {
        return Err(FastMemoryError::DimensionMismatch {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn check_finite(field: &'static str, values: &[f32]) -> Result<(), FastMemoryError> {
    match values.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(FastMemoryError::NonFinite { field, index }),
        None => Ok(()),
    }
}

fn check_decay(index: usize, value: f32) -> Result<(), FastMemoryError> {
    if !(value.is_finite() && value > 0.0 && value <= 1.0) {
        return Err(FastMemoryError::InvalidDecay { index, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FastMemoryConfig {
        FastMemoryConfig {
            heads: 2,
            key_dim: 3,
            value_dim: 2,
            checkpoint_interval: 4,
            max_writes: 16,
        }
    }

    fn request() -> WriteRequest {
        WriteRequest {
            source: SourceRef {
                key: "capsule-a".into(),
                generation: Generation(1),
                input_digest: [1; 32],
            },
            key: vec![3.0, 0.0, 4.0, 0.0, 2.0, 0.0],
            value: vec![1.0, 2.0, 3.0, 4.0],
            beta: 0.5,
            decay: Decay::None,
        }
    }

    #[test]
    fn every_head_key_is_normalised_independently() {
        let write = admit_write(&config(), WriteSeq(1), request()).unwrap();
        assert_eq!(write.key(), &[0.6, 0.0, 0.8, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_zero_head_key_is_refused_with_its_head() {
        let mut bad = request();
        bad.key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(
            admit_write(&config(), WriteSeq(1), bad).unwrap_err(),
            FastMemoryError::ZeroKey { head: 1 }
        );
    }

    #[test]
    fn beta_outside_the_unit_interval_is_refused() {
        for beta in [0.0, -0.1, 1.5, f32::NAN] {
            let mut bad = request();
            bad.beta = beta;
            assert!(matches!(
                admit_write(&config(), WriteSeq(1), bad),
                Err(FastMemoryError::InvalidBeta { .. })
            ));
        }
    }

    #[test]
    fn a_per_channel_decay_must_cover_every_key_channel() {
        let mut bad = request();
        bad.decay = Decay::PerChannel(vec![0.9; 5]);
        assert!(matches!(
            admit_write(&config(), WriteSeq(1), bad),
            Err(FastMemoryError::DimensionMismatch {
                field: "decay",
                expected: 6,
                actual: 5
            })
        ));
    }

    #[test]
    fn a_non_finite_value_is_refused_at_its_index() {
        let mut bad = request();
        bad.value[2] = f32::INFINITY;
        assert_eq!(
            admit_write(&config(), WriteSeq(1), bad).unwrap_err(),
            FastMemoryError::NonFinite {
                field: "value",
                index: 2
            }
        );
    }
}
