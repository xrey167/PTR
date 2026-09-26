use std::collections::BTreeMap;

use crate::error::LineageError;

/// Which split a sample belongs to. Only `Train` samples may enter a replay
/// pool; a held-out sample that was ever replayed stops measuring anything.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Split {
    Train,
    HeldOut,
}

/// A monotone training clock.
///
/// Forgetting in a network is driven by parameter updates, not by wall-clock
/// time: a week without training forgets nothing, one large update can forget a
/// lot. The clock is therefore advanced by the trainer, for example by
/// cumulative optimiser steps or cumulative update norm (the "model time" of
/// FOREVER-style replay scheduling), never by the calendar.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct ModelTime(pub f64);

/// Memory state of one sample in the style of FSRS: stability `S` (model time
/// until retrievability falls to 0.9) and difficulty `D` in `[1, 10]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryState {
    pub stability: f64,
    pub difficulty: f64,
    pub last_probe: ModelTime,
    /// How often probes found the sample forgotten.
    pub lapses: u32,
}

/// A sample in the replay pool.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplaySample {
    pub id: String,
    /// Stratum for diversity: task, domain or embedding cluster.
    pub stratum: String,
    pub memory: MemoryState,
}

/// A sample offered to the pool.
#[derive(Clone, Debug, PartialEq)]
pub struct NewSample {
    pub id: String,
    pub stratum: String,
    pub split: Split,
}

/// Parameters of the memory model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayParams {
    /// Stability of a sample that has never been probed.
    pub initial_stability: f64,
    /// Stability growth after a successful probe, scaled by how forgotten the
    /// sample was and how easy it is.
    pub growth: f64,
    /// Factor applied to stability after a lapse.
    pub lapse_factor: f64,
    pub min_stability: f64,
    /// Difficulty change per probe.
    pub difficulty_step: f64,
    /// Probe loss above which the sample counts as forgotten.
    pub lapse_loss: f64,
    /// Lapses after which a sample is withheld from replay until its label is
    /// audited: a sample the model keeps forgetting is as likely mislabelled
    /// as hard, and hard-example replay amplifies label noise.
    pub max_lapses: u32,
}

impl ReplayParams {
    fn check(&self) -> Result<(), LineageError> {
        let positive = [
            ("initial_stability", self.initial_stability),
            ("min_stability", self.min_stability),
            ("growth", self.growth),
            ("difficulty_step", self.difficulty_step),
        ];
        for (field, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(LineageError::InvalidParameter {
                    field,
                    message: "must be finite and positive",
                });
            }
        }
        if !(self.lapse_factor.is_finite() && self.lapse_factor > 0.0 && self.lapse_factor < 1.0) {
            return Err(LineageError::InvalidParameter {
                field: "lapse_factor",
                message: "must lie in (0, 1)",
            });
        }
        if !self.lapse_loss.is_finite() {
            return Err(LineageError::InvalidParameter {
                field: "lapse_loss",
                message: "must be finite",
            });
        }
        Ok(())
    }
}

/// FSRS-4.5 power forgetting curve `R(t, S) = (1 + F t / S)^C` with
/// `C = -0.5` and `F = 19/81`, so that `R(S, S) = 0.9`.
pub fn retrievability(elapsed: f64, stability: f64) -> f64 {
    const DECAY: f64 = -0.5;
    const FACTOR: f64 = 19.0 / 81.0;
    (1.0 + FACTOR * elapsed.max(0.0) / stability).powf(DECAY)
}

/// A pool of replayable training samples scheduled by measured forgetting.
#[derive(Clone, Debug)]
pub struct ReplayPool {
    params: ReplayParams,
    samples: BTreeMap<String, ReplaySample>,
}

impl ReplayPool {
    /// Create an empty replay pool. Returns `LineageError::InvalidParameter`
    /// for nonfinite or nonpositive stability, growth, or difficulty-step
    /// parameters, a lapse factor outside `(0, 1)`, or nonfinite lapse loss.
    pub fn new(params: ReplayParams) -> Result<Self, LineageError> {
        params.check()?;
        Ok(Self {
            params,
            samples: BTreeMap::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&ReplaySample> {
        self.samples.get(id)
    }

    /// Admit a training sample. Held-out samples are refused by type of split,
    /// not by convention.
    ///
    /// # Errors
    /// Returns `LineageError::HeldOutSample` for a held-out sample,
    /// `LineageError::NonFinite` for a nonfinite `now` (a NaN or infinite
    /// last probe would make the sample's priority zero forever), or
    /// `LineageError::DuplicateSample` for an id already pooled.
    pub fn insert(&mut self, sample: NewSample, now: ModelTime) -> Result<(), LineageError> {
        if sample.split == Split::HeldOut {
            return Err(LineageError::HeldOutSample { id: sample.id });
        }
        if !now.0.is_finite() {
            return Err(LineageError::NonFinite {
                field: "model time",
            });
        }
        if self.samples.contains_key(&sample.id) {
            return Err(LineageError::DuplicateSample { id: sample.id });
        }
        self.samples.insert(
            sample.id.clone(),
            ReplaySample {
                id: sample.id,
                stratum: sample.stratum,
                memory: MemoryState {
                    stability: self.params.initial_stability,
                    difficulty: 5.0,
                    last_probe: now,
                    lapses: 0,
                },
            },
        );
        Ok(())
    }

    /// Update a sample's memory state from a probe of the current model.
    ///
    /// A probe is a measurement, not a guess: the sample's loss under the
    /// current weights says whether it is still retained. A retained sample's
    /// stability grows, more so when it was probed late (low retrievability)
    /// and when it is easy; a lapsed sample's stability shrinks and its
    /// difficulty rises.
    ///
    /// # Errors
    /// Returns `LineageError::NonFinite` for a nonfinite loss or `now`,
    /// `LineageError::UnknownSample` for an id not in the pool, and
    /// `LineageError::InvalidParameter` when `now` precedes the sample's last
    /// probe: the clock is monotone, and moving the last probe backward would
    /// make the sample look more forgotten than it is. A refused probe leaves
    /// the sample's memory unchanged.
    pub fn record_probe(
        &mut self,
        id: &str,
        loss: f64,
        now: ModelTime,
    ) -> Result<(), LineageError> {
        if !loss.is_finite() {
            return Err(LineageError::NonFinite {
                field: "probe loss",
            });
        }
        if !now.0.is_finite() {
            return Err(LineageError::NonFinite {
                field: "model time",
            });
        }
        let params = self.params;
        let sample = self
            .samples
            .get_mut(id)
            .ok_or_else(|| LineageError::UnknownSample { id: id.to_owned() })?;
        let memory = &mut sample.memory;
        if now.0 < memory.last_probe.0 {
            return Err(LineageError::InvalidParameter {
                field: "model time",
                message: "must not precede the sample's last probe",
            });
        }
        let elapsed = now.0 - memory.last_probe.0;
        let recall = retrievability(elapsed, memory.stability);
        if loss > params.lapse_loss {
            memory.stability = (memory.stability * params.lapse_factor).max(params.min_stability);
            memory.difficulty = (memory.difficulty + params.difficulty_step).min(10.0);
            memory.lapses = memory.lapses.saturating_add(1);
        } else {
            let ease = (11.0 - memory.difficulty) / 10.0;
            memory.stability *= 1.0 + params.growth * ease * (1.0 - recall);
            memory.difficulty = (memory.difficulty - params.difficulty_step).max(1.0);
        }
        memory.last_probe = now;
        Ok(())
    }

    /// Replay priority: how forgotten the sample probably is, weighted up for
    /// difficult samples.
    pub fn priority(&self, sample: &ReplaySample, now: ModelTime) -> f64 {
        let recall = retrievability(now.0 - sample.memory.last_probe.0, sample.memory.stability);
        (1.0 - recall) * (1.0 + sample.memory.difficulty / 10.0)
    }

    /// Samples withheld from replay pending a label audit.
    pub fn needing_audit(&self) -> impl Iterator<Item = &ReplaySample> {
        let limit = self.params.max_lapses;
        self.samples
            .values()
            .filter(move |sample| sample.memory.lapses >= limit)
    }

    /// Draw up to `count` distinct samples.
    ///
    /// Samples with zero priority (probed at this model time, so nothing can
    /// have been forgotten) and samples awaiting a label audit are not
    /// eligible. The count is split across strata in proportion to their total
    /// priority (largest remainder), then each stratum is sampled without
    /// replacement by Gumbel-top-k: keep the smallest `-ln(u) / w`, which is
    /// the Efraimidis-Spirakis rule in a numerically stable form. Its
    /// inclusion probabilities are not proportional to `w`, so a trainer that
    /// reweights by inverse inclusion must compute them, not assume them.
    /// A stratum assigned a zero quota contributes no samples. Within each
    /// selected stratum, sampling uses the positive priorities as weights.
    /// The same seed returns the same draw on the same platform (`ln` comes
    /// from the platform's math library).
    pub fn sample(&self, count: usize, now: ModelTime, seed: u64) -> Vec<&str> {
        let mut strata: BTreeMap<&str, Vec<(&ReplaySample, f64)>> = BTreeMap::new();
        for sample in self.samples.values() {
            if sample.memory.lapses >= self.params.max_lapses {
                continue;
            }
            let weight = self.priority(sample, now);
            // A zero or undefined priority is never drawn.
            if weight.is_nan() || weight <= 0.0 {
                continue;
            }
            strata
                .entry(sample.stratum.as_str())
                .or_default()
                .push((sample, weight));
        }
        let masses: Vec<(&str, f64, usize)> = strata
            .iter()
            .map(|(name, members)| (*name, members.iter().map(|(_, w)| w).sum(), members.len()))
            .collect();
        let eligible: usize = masses.iter().map(|(_, _, size)| size).sum();
        let quotas = allocate(count.min(eligible), &masses);

        let mut drawn = Vec::with_capacity(count);
        for (name, quota) in quotas {
            let mut keyed: Vec<(f64, &str)> = strata[name]
                .iter()
                .map(|(sample, weight)| {
                    let u = unit_draw(seed, name, &sample.id);
                    (-u.ln() / weight, sample.id.as_str())
                })
                .collect();
            keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
            drawn.extend(keyed.into_iter().take(quota).map(|(_, id)| id));
        }
        drawn
    }
}

/// Largest-remainder allocation of `count` across strata by mass, capped by
/// each stratum's size, with any capped excess passed on to the rest.
fn allocate<'a>(count: usize, masses: &[(&'a str, f64, usize)]) -> Vec<(&'a str, usize)> {
    let mut quotas: Vec<(&str, usize)> = masses.iter().map(|(name, _, _)| (*name, 0)).collect();
    let mut remaining = count;
    while remaining > 0 {
        let open: Vec<usize> = (0..masses.len())
            .filter(|&i| quotas[i].1 < masses[i].2)
            .collect();
        if open.is_empty() {
            break;
        }
        let total: f64 = open.iter().map(|&i| masses[i].1).sum();
        let mut shares: Vec<(usize, usize, f64)> = open
            .iter()
            .map(|&i| {
                let exact = remaining as f64 * masses[i].1 / total;
                (i, exact.floor() as usize, exact - exact.floor())
            })
            .collect();
        let given: usize = shares.iter().map(|(_, whole, _)| whole).sum();
        shares.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        // The units the floors left over go to the largest remainders, one each.
        for share in shares.iter_mut().take(remaining.saturating_sub(given)) {
            share.1 += 1;
        }
        let mut placed = 0;
        for (i, whole, _) in shares {
            let room = masses[i].2 - quotas[i].1;
            let take = whole.min(room);
            quotas[i].1 += take;
            placed += take;
        }
        if placed == 0 {
            break;
        }
        remaining -= placed;
    }
    quotas
}

/// A uniform draw in `(0, 1)` determined by the seed, the stratum and the
/// sample id, so a draw does not depend on the order samples are stored in.
fn unit_draw(seed: u64, stratum: &str, id: &str) -> f64 {
    let mut state = stratum
        .bytes()
        .chain([0xffu8])
        .chain(id.bytes())
        .fold(0xcbf2_9ce4_8422_2325_u64 ^ seed, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrievability_is_point_nine_after_one_stability() {
        assert!((retrievability(7.0, 7.0) - 0.9).abs() < 1e-12);
        assert!((retrievability(0.0, 7.0) - 1.0).abs() < 1e-12);
        assert!(retrievability(70.0, 7.0) < retrievability(7.0, 7.0));
    }

    #[test]
    fn allocation_follows_mass_and_respects_capacity() {
        let quotas = allocate(10, &[("a", 3.0, 2), ("b", 1.0, 100)]);
        assert_eq!(quotas, vec![("a", 2), ("b", 8)]);
        let quotas = allocate(4, &[("a", 1.0, 10), ("b", 1.0, 10)]);
        assert_eq!(quotas, vec![("a", 2), ("b", 2)]);
    }

    #[test]
    fn unit_draws_lie_strictly_inside_the_unit_interval() {
        for i in 0..1000 {
            let u = unit_draw(i, "t", &format!("s{i}"));
            assert!(u > 0.0 && u < 1.0);
        }
    }
}
