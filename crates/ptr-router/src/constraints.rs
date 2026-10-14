//! Cost, latency, uncertainty and capability constraints over eligible routing options.
//!
//! Constraints only *remove* options from the set the caller marked eligible; nothing here can
//! make an ineligible option selectable (AGENTS.md section 3: the router never changes
//! eligibility). Among the feasible options the choice is deterministic: lexicographic over an
//! ordered list of criteria or by an integer weighted sum, ties broken by the lesser id.

use crate::recipe::{BackendId, CapabilitySet};
use ptr_types::Probability;
use std::fmt;

/// Upper bound on options one selection considers (Invariant 8).
pub const MAX_OPTIONS: usize = 256;

/// An option the caller may mark eligible, with the caller's estimates for it.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteOption {
    id: BackendId,
    cost_units: u32,
    latency_ms: u32,
    capabilities: CapabilitySet,
    uncertainty: Probability,
}

impl RouteOption {
    /// `uncertainty` is the caller's estimate of the residual uncertainty when this option is
    /// used.
    pub fn new(
        id: BackendId,
        cost_units: u32,
        latency_ms: u32,
        capabilities: CapabilitySet,
        uncertainty: Probability,
    ) -> Self {
        Self {
            id,
            cost_units,
            latency_ms,
            capabilities,
            uncertainty,
        }
    }

    pub fn id(&self) -> &BackendId {
        &self.id
    }
    pub fn cost_units(&self) -> u32 {
        self.cost_units
    }
    pub fn latency_ms(&self) -> u32 {
        self.latency_ms
    }
    pub fn capabilities(&self) -> CapabilitySet {
        self.capabilities
    }
    pub fn uncertainty(&self) -> Probability {
        self.uncertainty
    }
}

/// An option together with the caller's eligibility verdict. Policies and selectors choose only
/// among entries with `eligible == true`.
#[derive(Clone, Debug, PartialEq)]
pub struct Offer<T> {
    pub option: T,
    pub eligible: bool,
}

impl<T> Offer<T> {
    pub fn eligible(option: T) -> Self {
        Self {
            option,
            eligible: true,
        }
    }

    pub fn ineligible(option: T) -> Self {
        Self {
            option,
            eligible: false,
        }
    }
}

/// Limits an option must satisfy to stay feasible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteConstraints {
    pub max_cost_units: u32,
    pub max_latency_ms: u32,
    pub min_capabilities: CapabilitySet,
    pub max_uncertainty: Probability,
}

impl RouteConstraints {
    /// Constraints that every option satisfies.
    pub fn unbounded() -> Self {
        Self {
            max_cost_units: u32::MAX,
            max_latency_ms: u32::MAX,
            min_capabilities: CapabilitySet::empty(),
            max_uncertainty: Probability::new(1.0).unwrap_or_default(),
        }
    }

    /// The same constraints with `required` added to the minimum capability set. Only ever
    /// tightens.
    pub fn requiring(mut self, required: CapabilitySet) -> Self {
        self.min_capabilities = self.min_capabilities.union(required);
        self
    }

    /// Every reason `option` is infeasible, in a fixed order; empty when feasible.
    pub fn violations(&self, option: &RouteOption) -> Vec<Infeasibility> {
        let mut reasons = Vec::new();
        if option.cost_units > self.max_cost_units {
            reasons.push(Infeasibility::CostExceeded {
                required: option.cost_units,
                limit: self.max_cost_units,
            });
        }
        if option.latency_ms > self.max_latency_ms {
            reasons.push(Infeasibility::LatencyExceeded {
                estimate: option.latency_ms,
                limit: self.max_latency_ms,
            });
        }
        let missing = self.min_capabilities.missing_from(option.capabilities);
        if !missing.is_empty() {
            reasons.push(Infeasibility::MissingCapabilities { missing });
        }
        if option.uncertainty.get() > self.max_uncertainty.get() {
            reasons.push(Infeasibility::UncertaintyTooHigh {
                estimate: option.uncertainty,
                limit: self.max_uncertainty,
            });
        }
        reasons
    }
}

/// Why an option cannot be chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum Infeasibility {
    /// The caller did not mark the option eligible.
    NotEligible,
    CostExceeded {
        required: u32,
        limit: u32,
    },
    LatencyExceeded {
        estimate: u32,
        limit: u32,
    },
    MissingCapabilities {
        missing: CapabilitySet,
    },
    UncertaintyTooHigh {
        estimate: Probability,
        limit: Probability,
    },
}

/// One option removed from the choice, with every reason.
#[derive(Clone, Debug, PartialEq)]
pub struct Rejection {
    pub id: BackendId,
    pub reasons: Vec<Infeasibility>,
}

/// Ordering criterion; smaller is better for all of them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Criterion {
    Cost,
    Latency,
    Uncertainty,
}

/// Integer weights of a weighted sum over cost units, latency ms and uncertainty in parts per
/// million.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Weights {
    pub cost: u32,
    pub latency: u32,
    pub uncertainty: u32,
}

/// How the feasible options are ordered. Both are exact integer computations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Objective {
    /// Compare by the first criterion, then the second, then the third.
    Lexicographic([Criterion; 3]),
    Weighted(Weights),
}

impl Default for Objective {
    fn default() -> Self {
        Self::Lexicographic([Criterion::Uncertainty, Criterion::Cost, Criterion::Latency])
    }
}

fn uncertainty_ppm(option: &RouteOption) -> u64 {
    (f64::from(option.uncertainty.get()) * 1_000_000.0).round() as u64
}

fn criterion_value(option: &RouteOption, criterion: Criterion) -> u64 {
    match criterion {
        Criterion::Cost => u64::from(option.cost_units),
        Criterion::Latency => u64::from(option.latency_ms),
        Criterion::Uncertainty => uncertainty_ppm(option),
    }
}

impl Objective {
    fn key(&self, option: &RouteOption) -> [u128; 3] {
        match self {
            Self::Lexicographic(order) => {
                order.map(|criterion| u128::from(criterion_value(option, criterion)))
            }
            Self::Weighted(weights) => [
                u128::from(weights.cost) * u128::from(option.cost_units)
                    + u128::from(weights.latency) * u128::from(option.latency_ms)
                    + u128::from(weights.uncertainty) * u128::from(uncertainty_ppm(option)),
                0,
                0,
            ],
        }
    }
}

/// The selected option with the sizes of the sets it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub chosen: RouteOption,
    pub eligible: usize,
    pub feasible: usize,
}

/// Typed failure of a constrained selection.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintError {
    NoOptions,
    TooManyOptions {
        count: usize,
    },
    DuplicateOption {
        id: BackendId,
    },
    /// No option is both eligible and feasible; every option is listed with its reasons.
    Infeasible {
        rejections: Vec<Rejection>,
    },
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoOptions => write!(f, "no routing options were offered"),
            Self::TooManyOptions { count } => {
                write!(f, "{count} options exceed the limit of {MAX_OPTIONS}")
            }
            Self::DuplicateOption { id } => write!(f, "option {id} is offered twice"),
            Self::Infeasible { rejections } => {
                write!(f, "no option is feasible ({} rejected)", rejections.len())
            }
        }
    }
}

impl std::error::Error for ConstraintError {}

/// What the constraints did to a selection, for telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintsOutcome {
    NotApplied,
    Satisfied { eligible: usize, feasible: usize },
    Infeasible { options: usize },
}

/// Filters infeasible options with typed reasons, then picks deterministically.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstrainedSelector {
    pub constraints: RouteConstraints,
    pub objective: Objective,
}

impl ConstrainedSelector {
    pub fn new(constraints: RouteConstraints, objective: Objective) -> Self {
        Self {
            constraints,
            objective,
        }
    }

    /// The options that are eligible and feasible, in offer order. Always a subset of the
    /// eligible offers.
    pub fn feasible<'a>(&self, offers: &'a [Offer<RouteOption>]) -> Vec<&'a RouteOption> {
        offers
            .iter()
            .filter(|offer| offer.eligible && self.constraints.violations(&offer.option).is_empty())
            .map(|offer| &offer.option)
            .collect()
    }

    pub fn select(&self, offers: &[Offer<RouteOption>]) -> Result<Selection, ConstraintError> {
        if offers.is_empty() {
            return Err(ConstraintError::NoOptions);
        }
        if offers.len() > MAX_OPTIONS {
            return Err(ConstraintError::TooManyOptions {
                count: offers.len(),
            });
        }
        let mut ids: Vec<&BackendId> = offers.iter().map(|offer| offer.option.id()).collect();
        ids.sort();
        if let Some(pair) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(ConstraintError::DuplicateOption {
                id: pair[0].clone(),
            });
        }
        let eligible = offers.iter().filter(|offer| offer.eligible).count();
        let feasible = self.feasible(offers);
        let best = feasible.iter().copied().min_by(|a, b| {
            self.objective
                .key(a)
                .cmp(&self.objective.key(b))
                .then_with(|| a.id().cmp(b.id()))
        });
        match best {
            Some(chosen) => Ok(Selection {
                chosen: chosen.clone(),
                eligible,
                feasible: feasible.len(),
            }),
            None => Err(ConstraintError::Infeasible {
                rejections: self.rejections(offers),
            }),
        }
    }

    fn rejections(&self, offers: &[Offer<RouteOption>]) -> Vec<Rejection> {
        let mut rejections: Vec<Rejection> = offers
            .iter()
            .map(|offer| {
                let mut reasons = Vec::new();
                if !offer.eligible {
                    reasons.push(Infeasibility::NotEligible);
                }
                reasons.extend(self.constraints.violations(&offer.option));
                Rejection {
                    id: offer.option.id().clone(),
                    reasons,
                }
            })
            .collect();
        rejections.sort_by(|a, b| a.id.cmp(&b.id));
        rejections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(name: &str) -> BackendId {
        BackendId::new(name).unwrap()
    }

    fn opt(name: &str, cost: u32, latency: u32, unc: f32) -> RouteOption {
        RouteOption::new(
            id(name),
            cost,
            latency,
            CapabilitySet::empty(),
            Probability::new(unc).unwrap(),
        )
    }

    #[test]
    fn weighted_sum_is_exact_integer_arithmetic() {
        let selector = ConstrainedSelector::new(
            RouteConstraints::unbounded(),
            Objective::Weighted(Weights {
                cost: 1,
                latency: 1,
                uncertainty: 0,
            }),
        );
        let offers = vec![
            Offer::eligible(opt("a", 10, 5, 0.1)),
            Offer::eligible(opt("b", 7, 7, 0.9)),
        ];
        // 15 vs 14.
        assert_eq!(selector.select(&offers).unwrap().chosen.id(), &id("b"));
    }

    #[test]
    fn equal_keys_break_by_lesser_id_in_any_order() {
        let selector =
            ConstrainedSelector::new(RouteConstraints::unbounded(), Objective::default());
        let mut offers = vec![
            Offer::eligible(opt("zz", 1, 1, 0.5)),
            Offer::eligible(opt("aa", 1, 1, 0.5)),
        ];
        let first = selector.select(&offers).unwrap().chosen;
        offers.reverse();
        assert_eq!(selector.select(&offers).unwrap().chosen, first);
        assert_eq!(first.id(), &id("aa"));
    }

    #[test]
    fn violations_list_every_reason_in_fixed_order() {
        let constraints = RouteConstraints {
            max_cost_units: 1,
            max_latency_ms: 1,
            min_capabilities: CapabilitySet::empty(),
            max_uncertainty: Probability::new(0.1).unwrap(),
        };
        let reasons = constraints.violations(&opt("x", 5, 5, 0.5));
        assert_eq!(reasons.len(), 3);
        assert!(matches!(reasons[0], Infeasibility::CostExceeded { .. }));
        assert!(matches!(
            reasons[2],
            Infeasibility::UncertaintyTooHigh { .. }
        ));
    }
}
