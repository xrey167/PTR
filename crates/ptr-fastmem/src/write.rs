use ptr_types::Generation;

use crate::config::{FastMemoryConfig, MAX_VALUE_MAGNITUDE};
use crate::error::FastMemoryError;

/// Position of a write in one memory's journal. The first write is `1`; the
/// last a journal may hold is `u64::MAX - 1`, so every journaled write has a
/// successor to number the next one.
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
    /// `heads * key_dim` values; each head's slice is normalised on admission
    /// to a unit vector, at any scale: only an all-zero head is refused.
    pub key: Vec<f32>,
    /// `heads * value_dim` values, each at most [`MAX_VALUE_MAGNITUDE`] in
    /// magnitude.
    pub value: Vec<f32>,
    /// Write strength in `(0, 1]`. At `1` the value previously associated with
    /// this exact key is replaced outright.
    pub beta: f32,
    pub decay: Decay,
}

/// A validated write: finite, correctly shaped, values within
/// [`MAX_VALUE_MAGNITUDE`], unit keys per head.
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

/// A validated query: one unit vector per head, and the head shape it was
/// normalised for.
///
/// The flattened key alone does not say where one head ends and the next
/// begins, so the query keeps `heads` and `key_dim`; a memory reads it only
/// when both equal its own configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Query {
    heads: usize,
    key_dim: usize,
    key: Vec<f32>,
}

impl Query {
    /// Normalise a raw query against `config`.
    ///
    /// # Errors
    /// Rejects a configuration outside the supported ranges, a raw query whose
    /// length is not `heads * key_dim`, nonfinite entries, and all-zero heads.
    pub fn new(config: &FastMemoryConfig, raw: Vec<f32>) -> Result<Self, FastMemoryError> {
        crate::config::check_config(config)?;
        let key = normalized_heads(config, "query", raw)?;
        Ok(Self {
            heads: config.heads,
            key_dim: config.key_dim,
            key,
        })
    }

    /// The number of heads the query was normalised for.
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// The per-head width the query was normalised for.
    pub fn key_dim(&self) -> usize {
        self.key_dim
    }

    pub fn key(&self) -> &[f32] {
        &self.key
    }

    /// Refuse a query normalised for another head shape than `config`'s. Only
    /// `heads` and `key_dim` shape a query, so memories that differ only in
    /// value width, checkpoint interval or journal bound share queries.
    pub(crate) fn check_shape(&self, config: &FastMemoryConfig) -> Result<(), FastMemoryError> {
        check_len("query_heads", config.heads, self.heads)?;
        check_len("query_key_dim", config.key_dim, self.key_dim)
    }
}

/// Validate a composed write against a memory's configuration without changing it.
///
/// Uses the same admission rules as writing and restoring a memory. Lifecycle,
/// journal capacity and sequence checks remain the caller's responsibility.
///
/// # Errors
/// Rejects invalid configurations, dimensions, nonfinite cells, value cells
/// beyond [`MAX_VALUE_MAGNITUDE`], all-zero key heads, and beta or
/// decay factors outside their supported ranges.
pub fn validate_write(
    config: &FastMemoryConfig,
    request: &WriteRequest,
) -> Result<(), FastMemoryError> {
    crate::config::check_config(config)?;
    admit_write(config, WriteSeq(0), request.clone()).map(|_| ())
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
    // A per-write bound, not a check on the folded state: admission must not
    // depend on the state, or a refold without a revoked write could admit a
    // different set of writes than a memory that never saw it.
    if let Some(index) = value
        .iter()
        .position(|cell| cell.abs() > MAX_VALUE_MAGNITUDE)
    {
        return Err(FastMemoryError::ValueOutOfRange {
            index,
            value: value[index],
        });
    }
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

/// Normalise every head of a finite key or query to a unit vector.
///
/// A head is first divided by its largest magnitude, so its entries lie in
/// `[-1, 1]` with one of them `±1`, and only then is its norm taken: no
/// square underflows or overflows `f32` whatever the head's scale, and the
/// norm lies in `[1, sqrt(key_dim)]`. Every head that is not all zero is
/// therefore stored with unit norm to within `f32` rounding, which keeps the
/// delta rule a contraction along the key (the premise of
/// [`MAX_VALUE_MAGNITUDE`]), and a head and every exact power-of-two
/// rescaling of it normalise to the same bits. An all-zero head has no
/// direction and is refused as [`FastMemoryError::ZeroKey`].
fn normalized_heads(
    config: &FastMemoryConfig,
    field: &'static str,
    mut raw: Vec<f32>,
) -> Result<Vec<f32>, FastMemoryError> {
    check_len(field, config.key_len(), raw.len())?;
    check_finite(field, &raw)?;
    for (head, chunk) in raw.chunks_mut(config.key_dim).enumerate() {
        let largest = chunk
            .iter()
            .fold(0.0_f32, |largest, x| largest.max(x.abs()));
        if largest == 0.0 {
            return Err(FastMemoryError::ZeroKey { head });
        }
        chunk.iter_mut().for_each(|x| *x /= largest);
        let norm = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        chunk.iter_mut().for_each(|x| *x /= norm);
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

    /// `2^exponent`, exactly.
    fn power_of_two(exponent: i32) -> f32 {
        f32::from_bits(((127 + exponent) as u32) << 23)
    }

    #[test]
    fn a_head_normalises_to_the_same_unit_key_at_every_power_of_two_scale() {
        let config = FastMemoryConfig {
            heads: 1,
            key_dim: 10,
            value_dim: 1,
            checkpoint_interval: 1000,
            max_writes: 65_536,
        };
        // Every square of this head underflows f32 (it used to be stored with
        // norm 2.4); at 2^170 times it every square overflows (it used to be
        // refused as a zero key). Rescaling by a power of two is exact, so every
        // scale must normalise to the same bits, and those bits to a unit key.
        let mut tiny = vec![2.6e-23_f32; 10];
        tiny[0] = 4.5e-23;
        let unit = normalized_heads(&config, "key", tiny.clone()).unwrap();
        let norm = unit.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "{norm}");
        for exponents in [&[-20][..], &[40], &[75], &[100, 70]] {
            let scaled: Vec<f32> = tiny
                .iter()
                .map(|x| exponents.iter().fold(*x, |x, &e| x * power_of_two(e)))
                .collect();
            assert!(scaled.iter().all(|x| x.is_finite()), "{exponents:?}");
            assert_eq!(
                normalized_heads(&config, "key", scaled.clone()).unwrap(),
                unit,
                "{exponents:?}"
            );
            assert_eq!(Query::new(&config, scaled).unwrap().key(), unit);
        }
        // Subnormal entries have a direction too; only an all-zero head has none.
        let mut subnormal = vec![0.0_f32; 10];
        subnormal[2] = f32::MIN_POSITIVE / 8.0;
        subnormal[7] = -f32::MIN_POSITIVE / 8.0;
        let unit = normalized_heads(&config, "key", subnormal).unwrap();
        assert_eq!(
            (unit[2], unit[7]),
            (1.0 / 2.0_f32.sqrt(), -1.0 / 2.0_f32.sqrt())
        );
        assert_eq!(
            normalized_heads(&config, "key", vec![-0.0; 10]).unwrap_err(),
            FastMemoryError::ZeroKey { head: 0 }
        );
    }

    #[test]
    fn a_query_carries_its_head_shape_and_refuses_an_unsupported_configuration() {
        let query = Query::new(&config(), vec![3.0, 0.0, 4.0, 0.0, 2.0, 0.0]).unwrap();
        assert_eq!((query.heads(), query.key_dim()), (2, 3));
        assert_eq!(query.key(), &[0.6, 0.0, 0.8, 0.0, 1.0, 0.0]);
        // A zero key width used to reach `chunks_mut(0)` and panic.
        assert!(matches!(
            Query::new(
                &FastMemoryConfig {
                    key_dim: 0,
                    ..config()
                },
                Vec::new()
            ),
            Err(FastMemoryError::InvalidConfig {
                field: "key_dim",
                ..
            })
        ));
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

    #[test]
    fn a_value_beyond_the_magnitude_bound_is_refused_at_its_index() {
        let beyond = f32::from_bits(MAX_VALUE_MAGNITUDE.to_bits() + 1);
        for value in [beyond, -beyond, f32::MAX, -f32::MAX] {
            let mut bad = request();
            bad.value[3] = value;
            assert_eq!(
                admit_write(&config(), WriteSeq(1), bad.clone()).unwrap_err(),
                FastMemoryError::ValueOutOfRange { index: 3, value }
            );
            assert_eq!(
                validate_write(&config(), &bad).unwrap_err(),
                FastMemoryError::ValueOutOfRange { index: 3, value }
            );
        }
        let mut at_bound = request();
        at_bound.value = vec![MAX_VALUE_MAGNITUDE, -MAX_VALUE_MAGNITUDE, 0.0, 1.0];
        assert!(admit_write(&config(), WriteSeq(1), at_bound).is_ok());
    }
}
