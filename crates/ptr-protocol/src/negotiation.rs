//! Semantic protocol versions, per-feature flags and deterministic version negotiation.
//!
//! This is the generic control layer's handshake. It is std-only and independent of any
//! transport. A [`CompatibilityMatrix`] declares which major-version pairs may talk to each
//! other and from which version each [`Feature`] is usable; [`CompatibilityMatrix::negotiate`]
//! turns two [`VersionOffer`]s into a [`NegotiatedSession`] or a typed [`NegotiationError`].
//!
//! The result is symmetric: both ends compute the same version and feature set from the same
//! pair of offers, so neither end has to trust the other's claim about what was selected.

use std::fmt;

/// Most versions one offer may list; keeps a Hello frame and the negotiation loop bounded.
pub const MAX_OFFER_VERSIONS: usize = 16;

/// A semantic `major.minor.patch` protocol version, ordered lexicographically.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl ProtocolVersion {
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// An optional protocol capability that two ends enable by agreement.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Feature {
    /// `Call` frames carry an idempotency key and duplicates are answered from a cache.
    Idempotency,
    /// A caller may `Cancel` an in-flight call.
    Cancel,
    /// `Bye` drains: in-flight work finishes before the session closes.
    Drain,
}

impl Feature {
    pub const ALL: [Feature; 3] = [Feature::Idempotency, Feature::Cancel, Feature::Drain];

    pub const fn bit(self) -> u32 {
        match self {
            Self::Idempotency => 1,
            Self::Cancel => 1 << 1,
            Self::Drain => 1 << 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Idempotency => "idempotency",
            Self::Cancel => "cancel",
            Self::Drain => "drain",
        }
    }
}

/// A set of [`Feature`]s, stored as the bit mask that goes on the wire.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct FeatureSet(u32);

impl FeatureSet {
    pub const EMPTY: Self = Self(0);

    pub fn all() -> Self {
        Self::of(&Feature::ALL)
    }

    pub fn of(features: &[Feature]) -> Self {
        Self(
            features
                .iter()
                .fold(0, |bits, feature| bits | feature.bit()),
        )
    }

    /// `None` when the mask has a bit no known feature owns, so an unknown bit is rejected
    /// rather than silently dropped.
    pub fn from_bits(bits: u32) -> Option<Self> {
        let known = Feature::ALL
            .iter()
            .fold(0, |all, feature| all | feature.bit());
        (bits & !known == 0).then_some(Self(bits))
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, feature: Feature) -> bool {
        self.0 & feature.bit() != 0
    }

    pub const fn with(self, feature: Feature) -> Self {
        Self(self.0 | feature.bit())
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Members in declaration order.
    pub fn members(self) -> Vec<Feature> {
        Feature::ALL
            .into_iter()
            .filter(|feature| self.contains(*feature))
            .collect()
    }

    /// The first member of `self` that `other` lacks.
    pub fn first_missing_from(self, other: Self) -> Option<Feature> {
        Feature::ALL
            .into_iter()
            .find(|feature| self.contains(*feature) && !other.contains(*feature))
    }
}

/// What one end offers in its Hello.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VersionOffer {
    /// Versions this end can speak. Must be non-empty and at most [`MAX_OFFER_VERSIONS`].
    pub supported: Vec<ProtocolVersion>,
    /// The lowest version this end will accept.
    pub minimum: ProtocolVersion,
    /// Features this end can use.
    pub features: FeatureSet,
    /// Features without which this end refuses the session; must be a subset of `features`.
    pub required: FeatureSet,
}

/// Why an offer is not well formed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OfferFault {
    NoVersions,
    TooManyVersions {
        len: usize,
    },
    /// `minimum` is above every supported version, so the offer can never be satisfied.
    MinimumAboveSupported,
    /// A required feature the offer does not itself list as offered.
    RequiredNotOffered(Feature),
}

impl VersionOffer {
    pub fn validate(&self) -> Result<(), OfferFault> {
        let Some(highest) = self.supported.iter().max() else {
            return Err(OfferFault::NoVersions);
        };
        if self.supported.len() > MAX_OFFER_VERSIONS {
            return Err(OfferFault::TooManyVersions {
                len: self.supported.len(),
            });
        }
        if self.minimum > *highest {
            return Err(OfferFault::MinimumAboveSupported);
        }
        if let Some(feature) = self.required.first_missing_from(self.features) {
            return Err(OfferFault::RequiredNotOffered(feature));
        }
        Ok(())
    }

    fn highest(&self) -> ProtocolVersion {
        self.supported.iter().copied().max().unwrap_or_default()
    }
}

/// Why two offers cannot become a session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NegotiationError {
    /// The local offer is malformed.
    InvalidLocalOffer(OfferFault),
    /// The remote offer is malformed.
    InvalidRemoteOffer(OfferFault),
    /// No pair of supported versions has major versions the matrix lets talk.
    MajorMismatch {
        local_highest: ProtocolVersion,
        remote_highest: ProtocolVersion,
    },
    /// The best common version is below the higher of the two minimums.
    BelowMinimum {
        selected: ProtocolVersion,
        required_minimum: ProtocolVersion,
    },
    /// A feature one side requires is not usable at the selected version on both sides.
    RequiredFeatureMissing {
        feature: Feature,
        selected: ProtocolVersion,
    },
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLocalOffer(fault) => write!(f, "invalid local offer: {fault:?}"),
            Self::InvalidRemoteOffer(fault) => write!(f, "invalid remote offer: {fault:?}"),
            Self::MajorMismatch {
                local_highest,
                remote_highest,
            } => write!(
                f,
                "no compatible major version (local up to {local_highest}, remote up to \
                 {remote_highest})"
            ),
            Self::BelowMinimum {
                selected,
                required_minimum,
            } => write!(
                f,
                "selected version {selected} is below minimum {required_minimum}"
            ),
            Self::RequiredFeatureMissing { feature, selected } => write!(
                f,
                "required feature {} is not usable at version {selected}",
                feature.name()
            ),
        }
    }
}

impl std::error::Error for NegotiationError {}

/// The agreed version and the features usable at it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NegotiatedSession {
    pub version: ProtocolVersion,
    pub features: FeatureSet,
}

/// Which version pairs are compatible and from which version each feature is usable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityMatrix {
    major_pairs: Vec<(u16, u16)>,
    feature_since: Vec<(Feature, ProtocolVersion)>,
}

impl Default for CompatibilityMatrix {
    /// Major 1 talks to major 1. Idempotency and cancel are usable from 1.0.0, draining
    /// close from 1.1.0.
    fn default() -> Self {
        Self::empty()
            .with_major_pair(1, 1)
            .with_feature_since(Feature::Idempotency, ProtocolVersion::new(1, 0, 0))
            .with_feature_since(Feature::Cancel, ProtocolVersion::new(1, 0, 0))
            .with_feature_since(Feature::Drain, ProtocolVersion::new(1, 1, 0))
    }
}

impl CompatibilityMatrix {
    /// A matrix that admits nothing; build up with the `with_*` methods.
    pub fn empty() -> Self {
        Self {
            major_pairs: Vec::new(),
            feature_since: Vec::new(),
        }
    }

    /// Declares that majors `a` and `b` may talk, in either direction.
    pub fn with_major_pair(mut self, a: u16, b: u16) -> Self {
        self.major_pairs.push((a, b));
        self
    }

    /// Declares that `feature` is usable at `since` and above. A feature with no entry is
    /// never usable.
    pub fn with_feature_since(mut self, feature: Feature, since: ProtocolVersion) -> Self {
        self.feature_since.retain(|(known, _)| *known != feature);
        self.feature_since.push((feature, since));
        self
    }

    pub fn compatible(&self, local: ProtocolVersion, remote: ProtocolVersion) -> bool {
        self.major_pairs.iter().any(|(a, b)| {
            (*a == local.major && *b == remote.major) || (*a == remote.major && *b == local.major)
        })
    }

    pub fn feature_usable(&self, feature: Feature, version: ProtocolVersion) -> bool {
        self.feature_since
            .iter()
            .any(|(known, since)| *known == feature && version >= *since)
    }

    /// Picks the highest version both ends can speak and the features usable at it.
    ///
    /// For each compatible pair the effective version is the lower of the two (the higher
    /// end speaks down), and the highest effective version wins. The result does not
    /// depend on which end is "local".
    pub fn negotiate(
        &self,
        local: &VersionOffer,
        remote: &VersionOffer,
    ) -> Result<NegotiatedSession, NegotiationError> {
        local
            .validate()
            .map_err(NegotiationError::InvalidLocalOffer)?;
        remote
            .validate()
            .map_err(NegotiationError::InvalidRemoteOffer)?;

        let mut selected: Option<ProtocolVersion> = None;
        for l in &local.supported {
            for r in &remote.supported {
                if self.compatible(*l, *r) {
                    let effective = (*l).min(*r);
                    selected = Some(selected.map_or(effective, |best| best.max(effective)));
                }
            }
        }
        let Some(selected) = selected else {
            return Err(NegotiationError::MajorMismatch {
                local_highest: local.highest(),
                remote_highest: remote.highest(),
            });
        };

        let required_minimum = local.minimum.max(remote.minimum);
        if selected < required_minimum {
            return Err(NegotiationError::BelowMinimum {
                selected,
                required_minimum,
            });
        }

        let usable = Feature::ALL
            .into_iter()
            .filter(|feature| {
                local.features.contains(*feature)
                    && remote.features.contains(*feature)
                    && self.feature_usable(*feature, selected)
            })
            .fold(FeatureSet::EMPTY, FeatureSet::with);
        if let Some(feature) = local
            .required
            .union(remote.required)
            .first_missing_from(usable)
        {
            return Err(NegotiationError::RequiredFeatureMissing { feature, selected });
        }
        Ok(NegotiatedSession {
            version: selected,
            features: usable,
        })
    }
}

/// Negotiates with the default [`CompatibilityMatrix`].
pub fn negotiate(
    local: &VersionOffer,
    remote: &VersionOffer,
) -> Result<NegotiatedSession, NegotiationError> {
    CompatibilityMatrix::default().negotiate(local, remote)
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn v(major: u16, minor: u16, patch: u16) -> ProtocolVersion {
        ProtocolVersion::new(major, minor, patch)
    }

    fn offer(
        supported: &[ProtocolVersion],
        minimum: ProtocolVersion,
        features: &[Feature],
        required: &[Feature],
    ) -> VersionOffer {
        VersionOffer {
            supported: supported.to_vec(),
            minimum,
            features: FeatureSet::of(features),
            required: FeatureSet::of(required),
        }
    }

    #[test]
    fn version_pair_table() {
        struct Case {
            name: &'static str,
            local: VersionOffer,
            remote: VersionOffer,
            expect: Result<(ProtocolVersion, Vec<Feature>), NegotiationError>,
        }
        let all = Feature::ALL;
        let cases = vec![
            Case {
                name: "same version, all features",
                local: offer(&[v(1, 1, 0)], v(1, 0, 0), &all, &[]),
                remote: offer(&[v(1, 1, 0)], v(1, 0, 0), &all, &[]),
                expect: Ok((v(1, 1, 0), all.to_vec())),
            },
            Case {
                name: "newer side speaks down; drain not usable at 1.0",
                local: offer(&[v(1, 1, 0)], v(1, 0, 0), &all, &[]),
                remote: offer(&[v(1, 0, 0)], v(1, 0, 0), &all, &[]),
                expect: Ok((v(1, 0, 0), vec![Feature::Idempotency, Feature::Cancel])),
            },
            Case {
                name: "highest common wins among several",
                local: offer(&[v(1, 0, 0), v(1, 1, 0), v(1, 2, 3)], v(1, 0, 0), &all, &[]),
                remote: offer(&[v(1, 0, 5), v(1, 1, 0)], v(1, 0, 0), &all, &[]),
                expect: Ok((v(1, 1, 0), all.to_vec())),
            },
            Case {
                name: "patch difference resolves to the lower patch",
                local: offer(&[v(1, 0, 7)], v(1, 0, 0), &[], &[]),
                remote: offer(&[v(1, 0, 2)], v(1, 0, 0), &[], &[]),
                expect: Ok((v(1, 0, 2), vec![])),
            },
            Case {
                name: "feature only one side offers is not usable",
                local: offer(&[v(1, 1, 0)], v(1, 0, 0), &all, &[]),
                remote: offer(&[v(1, 1, 0)], v(1, 0, 0), &[Feature::Cancel], &[]),
                expect: Ok((v(1, 1, 0), vec![Feature::Cancel])),
            },
            Case {
                name: "major mismatch",
                local: offer(&[v(1, 0, 0)], v(1, 0, 0), &[], &[]),
                remote: offer(&[v(2, 0, 0)], v(2, 0, 0), &[], &[]),
                expect: Err(NegotiationError::MajorMismatch {
                    local_highest: v(1, 0, 0),
                    remote_highest: v(2, 0, 0),
                }),
            },
            Case {
                name: "remote minimum above the common version",
                local: offer(&[v(1, 0, 0)], v(1, 0, 0), &[], &[]),
                remote: offer(&[v(1, 0, 0), v(1, 3, 0)], v(1, 2, 0), &[], &[]),
                expect: Err(NegotiationError::BelowMinimum {
                    selected: v(1, 0, 0),
                    required_minimum: v(1, 2, 0),
                }),
            },
            Case {
                name: "local minimum above the common version",
                local: offer(&[v(1, 0, 0), v(1, 4, 0)], v(1, 3, 0), &[], &[]),
                remote: offer(&[v(1, 1, 0)], v(1, 0, 0), &[], &[]),
                expect: Err(NegotiationError::BelowMinimum {
                    selected: v(1, 1, 0),
                    required_minimum: v(1, 3, 0),
                }),
            },
            Case {
                name: "required feature the peer lacks",
                local: offer(&[v(1, 1, 0)], v(1, 0, 0), &all, &[Feature::Drain]),
                remote: offer(&[v(1, 1, 0)], v(1, 0, 0), &[Feature::Cancel], &[]),
                expect: Err(NegotiationError::RequiredFeatureMissing {
                    feature: Feature::Drain,
                    selected: v(1, 1, 0),
                }),
            },
            Case {
                name: "required feature not yet usable at the selected version",
                local: offer(&[v(1, 1, 0)], v(1, 0, 0), &all, &[]),
                remote: offer(&[v(1, 0, 0)], v(1, 0, 0), &all, &[Feature::Drain]),
                expect: Err(NegotiationError::RequiredFeatureMissing {
                    feature: Feature::Drain,
                    selected: v(1, 0, 0),
                }),
            },
        ];
        for case in cases {
            let forward = negotiate(&case.local, &case.remote);
            let backward = negotiate(&case.remote, &case.local);
            match &case.expect {
                Ok((version, features)) => {
                    let session = forward.unwrap_or_else(|e| panic!("{}: {e}", case.name));
                    assert_eq!(session.version, *version, "{}", case.name);
                    assert_eq!(session.features.members(), *features, "{}", case.name);
                    assert_eq!(forward, backward, "{}: symmetric", case.name);
                }
                Err(error) => {
                    assert_eq!(forward, Err(*error), "{}", case.name);
                    assert!(backward.is_err(), "{}: symmetric rejection", case.name);
                }
            }
        }
    }

    #[test]
    fn malformed_offers_are_rejected_with_the_side_that_sent_them() {
        let good = offer(&[v(1, 0, 0)], v(1, 0, 0), &[], &[]);
        let empty = offer(&[], v(1, 0, 0), &[], &[]);
        assert_eq!(
            negotiate(&empty, &good),
            Err(NegotiationError::InvalidLocalOffer(OfferFault::NoVersions))
        );
        let too_many = offer(&[v(1, 0, 0); MAX_OFFER_VERSIONS + 1], v(1, 0, 0), &[], &[]);
        assert_eq!(
            negotiate(&good, &too_many),
            Err(NegotiationError::InvalidRemoteOffer(
                OfferFault::TooManyVersions {
                    len: MAX_OFFER_VERSIONS + 1
                }
            ))
        );
        let high_min = offer(&[v(1, 0, 0)], v(1, 5, 0), &[], &[]);
        assert_eq!(high_min.validate(), Err(OfferFault::MinimumAboveSupported));
        let unoffered = offer(&[v(1, 0, 0)], v(1, 0, 0), &[], &[Feature::Cancel]);
        assert_eq!(
            unoffered.validate(),
            Err(OfferFault::RequiredNotOffered(Feature::Cancel))
        );
    }

    #[test]
    fn custom_matrix_allows_cross_major_pairs_and_gates_features() {
        let matrix = CompatibilityMatrix::empty()
            .with_major_pair(1, 2)
            .with_feature_since(Feature::Cancel, v(1, 0, 0));
        let a = offer(&[v(1, 4, 0)], v(1, 0, 0), &Feature::ALL, &[]);
        let b = offer(&[v(2, 0, 0)], v(1, 0, 0), &Feature::ALL, &[]);
        let session = matrix.negotiate(&a, &b).unwrap();
        assert_eq!(session.version, v(1, 4, 0));
        assert_eq!(session.features.members(), vec![Feature::Cancel]);
        assert!(!matrix.compatible(v(1, 0, 0), v(1, 0, 0)));
    }

    #[test]
    fn unknown_feature_bits_are_refused() {
        assert_eq!(FeatureSet::from_bits(0b111), Some(FeatureSet::all()));
        assert_eq!(FeatureSet::from_bits(0b1000), None);
    }
}
