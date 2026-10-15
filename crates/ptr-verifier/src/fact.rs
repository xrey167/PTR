//! Checking claims against one immutable SemDB snapshot.
//!
//! A claim is `(subject, attribute, value)`. The verifier reads a [`SemanticSnapshot`] and never
//! writes: the snapshot is an immutable revision-bound view, the report names its revision in
//! every finding and [`FactVerifier::verify_bound`] returns it next to the report. Facts live in
//! the snapshot under the key `fact/<subject>/<attribute>` as a text value (see [`fact_key`]);
//! text is compared byte for byte.
//!
//! * the snapshot holds another value: `fact.contradicted`, hard (`Fail`);
//! * the snapshot holds nothing (or a non-text payload) for it: `fact.unknown`, soft, and the
//!   status is `Unknown`, not `Fail`: absence of evidence is not a contradiction;
//! * the snapshot is not the host's live revision (required by [`FactVerifier::new`]):
//!   `fact.stale_snapshot`, soft, `Unknown`; a stale snapshot is not allowed to contradict
//!   anything, so no claim is judged;
//! * an ill-formed or oversize claim list: `fact.malformed` / `fact.oversize`, hard.
//!
//! `Pass` needs every claim to match the snapshot. An empty claim list is `Unknown`: asserting
//! nothing proves nothing. See [`crate::codes`].

use ptr_semdb::{SemanticSnapshot, SemanticValue};
use ptr_types::{Generation, Probability, Revision, VerificationLevel};

use crate::bound::{BoundVerificationReport, BoundVerifier, NamedBoundVerifier, SnapshotBinding};
use crate::codes::finding;
use crate::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// Most claims in one candidate.
pub const MAX_CLAIMS: usize = 64;
/// Longest subject, attribute or value, in bytes.
pub const MAX_FIELD_BYTES: usize = 256;

/// One claimed fact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactClaim {
    pub subject: String,
    pub attribute: String,
    pub value: String,
}

impl FactClaim {
    pub fn new(subject: &str, attribute: &str, value: &str) -> Self {
        Self {
            subject: subject.to_owned(),
            attribute: attribute.to_owned(),
            value: value.to_owned(),
        }
    }
}

/// The snapshot key of a fact. `subject` and `attribute` must not contain `/`
/// (a claim that does is `fact.malformed`).
pub fn fact_key(subject: &str, attribute: &str) -> String {
    format!("fact/{subject}/{attribute}")
}

/// A report together with the snapshot revision it was judged against.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundReport {
    pub snapshot_revision: Revision,
    pub report: VerificationReport,
}

/// Verifies claims against one snapshot. See the module documentation.
#[derive(Clone, Debug)]
pub struct FactVerifier {
    snapshot: SemanticSnapshot,
    live: Option<Revision>,
    generation: Generation,
}

impl FactVerifier {
    /// A verifier for `snapshot`, which must be at `live`, the host's current revision: if it is
    /// not, every report is `Unknown` (`fact.stale_snapshot`) and no claim is judged. The live
    /// revision is mandatory so a stale snapshot cannot be used by forgetting to say so.
    pub fn new(snapshot: SemanticSnapshot, live: Revision) -> Self {
        Self {
            snapshot,
            live: Some(live),
            generation: Generation::default(),
        }
    }

    /// A verifier that never checks staleness: the host guarantees that `snapshot` is current
    /// (for example it takes the snapshot and verifies within one critical section). Prefer
    /// [`new`](Self::new); with this constructor a stale snapshot silently judges claims against
    /// old facts.
    pub fn without_staleness_check(snapshot: SemanticSnapshot) -> Self {
        Self {
            snapshot,
            live: None,
            generation: Generation::default(),
        }
    }

    /// Declares the generation the snapshot belongs to (default `Generation(0)`), so the
    /// [`BoundVerifier`] binding names it. The snapshot itself carries only a revision; the
    /// generation is the host's statement and is not checked here.
    #[must_use]
    pub fn with_generation(mut self, generation: Generation) -> Self {
        self.generation = generation;
        self
    }

    pub fn snapshot_revision(&self) -> Revision {
        self.snapshot.revision
    }

    /// Verifies and names the revision the verdict is bound to. The [`Verifier`] trait returns a
    /// bare [`VerificationReport`] with no field for a revision, so the structured revision is
    /// only available here; through the trait it appears in the message of every finding
    /// (`revision N`).
    pub fn verify_bound(&self, claims: &Vec<FactClaim>) -> BoundReport {
        BoundReport {
            snapshot_revision: self.snapshot.revision,
            report: self.verify(claims),
        }
    }
}

fn report(status: VerificationStatus, findings: Vec<Finding>) -> VerificationReport {
    let score = if status == VerificationStatus::Pass {
        1.0
    } else {
        0.0
    };
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(score).expect("bounded score"),
        findings,
    }
}

fn well_formed(field: &str) -> bool {
    !field.is_empty() && !field.contains('/') && !field.chars().any(char::is_control)
}

impl Verifier<Vec<FactClaim>> for FactVerifier {
    fn verify(&self, claims: &Vec<FactClaim>) -> VerificationReport {
        let revision = self.snapshot.revision.0;
        if claims.len() > MAX_CLAIMS
            || claims.iter().any(|c| {
                c.subject.len() > MAX_FIELD_BYTES
                    || c.attribute.len() > MAX_FIELD_BYTES
                    || c.value.len() > MAX_FIELD_BYTES
            })
        {
            return report(
                VerificationStatus::Fail,
                vec![finding(
                    "fact.oversize",
                    format!(
                        "at most {MAX_CLAIMS} claims of at most {MAX_FIELD_BYTES} bytes per field"
                    ),
                )],
            );
        }
        let mut findings: Vec<Finding> = claims
            .iter()
            .enumerate()
            .filter(|(_, c)| !well_formed(&c.subject) || !well_formed(&c.attribute))
            .map(|(index, _)| {
                finding(
                    "fact.malformed",
                    format!("claim {index}: subject and attribute must be non-empty, without '/' or control characters"),
                )
            })
            .collect();
        if !findings.is_empty() {
            return report(VerificationStatus::Fail, findings);
        }
        if self.live.is_some_and(|live| live != self.snapshot.revision) {
            return report(
                VerificationStatus::Unknown,
                vec![finding(
                    "fact.stale_snapshot",
                    format!(
                        "snapshot revision {revision} is not the live revision {}",
                        self.live.map_or(0, |live| live.0)
                    ),
                )],
            );
        }
        if claims.is_empty() {
            return report(
                VerificationStatus::Unknown,
                vec![finding(
                    "fact.unknown",
                    format!("no claims to check against snapshot revision {revision}"),
                )],
            );
        }
        let mut contradicted = false;
        for (index, claim) in claims.iter().enumerate() {
            let key = fact_key(&claim.subject, &claim.attribute);
            match self.snapshot.value(&key) {
                Some(SemanticValue::Text(held)) if *held == claim.value => {}
                Some(SemanticValue::Text(_)) => {
                    contradicted = true;
                    findings.push(finding(
                        "fact.contradicted",
                        format!(
                            "claim {index} ({} {}): snapshot revision {revision} holds another value",
                            claim.subject, claim.attribute
                        ),
                    ));
                }
                Some(SemanticValue::Payload(_)) | None => findings.push(finding(
                    "fact.unknown",
                    format!(
                        "claim {index} ({} {}): snapshot revision {revision} holds no text value",
                        claim.subject, claim.attribute
                    ),
                )),
            }
        }
        let status = if contradicted {
            VerificationStatus::Fail
        } else if findings.is_empty() {
            VerificationStatus::Pass
        } else {
            VerificationStatus::Unknown
        };
        report(status, findings)
    }
}

impl BoundVerifier<Vec<FactClaim>> for FactVerifier {
    fn verify_at_snapshot(&self, claims: &Vec<FactClaim>) -> BoundVerificationReport {
        BoundVerificationReport {
            report: self.verify(claims),
            snapshot: SnapshotBinding::new(self.snapshot.revision, self.generation),
        }
    }
}

impl NamedBoundVerifier<Vec<FactClaim>> for FactVerifier {
    fn bound_name(&self) -> &'static str {
        "fact"
    }
}

impl NamedVerifier<Vec<FactClaim>> for FactVerifier {
    fn name(&self) -> &'static str {
        "fact"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_semdb::{SemanticDelta, SemanticHost};

    fn host() -> SemanticHost {
        let mut host = SemanticHost::default();
        let mut delta = SemanticDelta::default();
        for (s, a, v) in [
            ("paris", "capital-of", "france"),
            ("water", "boils-at", "100"),
        ] {
            delta.upserts.insert(fact_key(s, a), v.into());
        }
        host.apply_delta(delta).unwrap();
        host
    }

    fn claims(items: &[(&str, &str, &str)]) -> Vec<FactClaim> {
        items
            .iter()
            .map(|(s, a, v)| FactClaim::new(s, a, v))
            .collect()
    }

    fn codes(report: &VerificationReport) -> Vec<&str> {
        report.findings.iter().map(|f| f.code.as_str()).collect()
    }

    #[test]
    fn matching_claims_pass_at_the_snapshot_revision() {
        let host = host();
        let verifier = FactVerifier::new(host.snapshot(), host.revision());
        let bound = verifier.verify_bound(&claims(&[
            ("paris", "capital-of", "france"),
            ("water", "boils-at", "100"),
        ]));
        assert_eq!(bound.report.status, VerificationStatus::Pass);
        assert_eq!(bound.report.level, VerificationLevel::Deterministic);
        assert_eq!(bound.report.score.get(), 1.0);
        assert!(bound.report.findings.is_empty());
        assert_eq!(bound.snapshot_revision, host.revision());
        assert_eq!(verifier.snapshot_revision(), host.revision());
    }

    #[test]
    fn a_contradiction_is_hard_and_names_the_revision() {
        let host = host();
        let verifier = FactVerifier::new(host.snapshot(), host.revision());
        let report = verifier.verify(&claims(&[
            ("paris", "capital-of", "france"),
            ("water", "boils-at", "90"),
        ]));
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(codes(&report), ["fact.contradicted"]);
        assert!(report.findings[0].hard);
        assert!(report.findings[0].message.contains("claim 1"));
        assert!(report.findings[0]
            .message
            .contains(&format!("revision {}", host.revision().0)));
    }

    #[test]
    fn an_unknown_fact_is_unknown_not_fail() {
        let host = host();
        let report = FactVerifier::new(host.snapshot(), host.revision())
            .verify(&claims(&[("mars", "moons", "2")]));
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(codes(&report), ["fact.unknown"]);
        assert!(!report.findings[0].hard);
        assert_eq!(report.score.get(), 0.0);
        // Known + unknown stays Unknown; a contradiction elsewhere makes it Fail.
        let verifier = FactVerifier::new(host.snapshot(), host.revision());
        let mixed = verifier.verify(&claims(&[
            ("paris", "capital-of", "france"),
            ("mars", "moons", "2"),
        ]));
        assert_eq!(mixed.status, VerificationStatus::Unknown);
        let worst = verifier.verify(&claims(&[
            ("mars", "moons", "2"),
            ("water", "boils-at", "1"),
        ]));
        assert_eq!(worst.status, VerificationStatus::Fail);
        assert_eq!(codes(&worst), ["fact.unknown", "fact.contradicted"]);
        assert_eq!(
            verifier.verify(&Vec::new()).status,
            VerificationStatus::Unknown
        );
    }

    #[test]
    fn a_non_text_value_is_not_evidence() {
        let mut host = host();
        let mut delta = SemanticDelta::default();
        delta.upserts.insert(
            fact_key("blob", "x"),
            ptr_semdb::SemanticPayload {
                type_id: ptr_types::TypeId::from("t"),
                source: "s".into(),
                bytes: vec![1],
            }
            .into(),
        );
        host.apply_delta(delta).unwrap();
        let report = FactVerifier::new(host.snapshot(), host.revision())
            .verify(&claims(&[("blob", "x", "1")]));
        assert_eq!(codes(&report), ["fact.unknown"]);
    }

    #[test]
    fn a_stale_snapshot_is_unknown_and_judges_nothing() {
        let mut host = host();
        let old = host.snapshot();
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert(fact_key("paris", "capital-of"), "lyon".into());
        host.apply_delta(delta).unwrap();
        let verifier = FactVerifier::new(old.clone(), host.revision());
        // The old snapshot would contradict nothing and confirm nothing.
        let report = verifier.verify(&claims(&[("paris", "capital-of", "france")]));
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(codes(&report), ["fact.stale_snapshot"]);
        assert!(!report.findings[0].hard);
        let report = verifier.verify(&claims(&[("paris", "capital-of", "lyon")]));
        assert_eq!(codes(&report), ["fact.stale_snapshot"]);
        // A fresh snapshot sees the new value.
        // The unchecked constructor is the host-duty case: it would judge against old facts.
        let unchecked = FactVerifier::without_staleness_check(old);
        assert_eq!(
            codes(&unchecked.verify(&claims(&[("paris", "capital-of", "lyon")]))),
            ["fact.contradicted"]
        );
        let fresh = FactVerifier::new(host.snapshot(), host.revision());
        assert_eq!(
            fresh
                .verify(&claims(&[("paris", "capital-of", "lyon")]))
                .status,
            VerificationStatus::Pass
        );
        assert_eq!(
            codes(&fresh.verify(&claims(&[("paris", "capital-of", "france")]))),
            ["fact.contradicted"]
        );
    }

    #[test]
    fn malformed_and_oversize_claims_are_hard_findings() {
        let host = host();
        let verifier = FactVerifier::new(host.snapshot(), host.revision());
        for bad in [
            claims(&[("", "a", "v")]),
            claims(&[("s", "", "v")]),
            claims(&[("a/b", "a", "v")]),
            claims(&[("s", "a\nb", "v")]),
        ] {
            let report = verifier.verify(&bad);
            assert_eq!(report.status, VerificationStatus::Fail);
            assert_eq!(codes(&report), ["fact.malformed"]);
            assert!(report.findings[0].hard);
        }
        let long = "x".repeat(MAX_FIELD_BYTES + 1);
        for big in [
            claims(&[(&long, "a", "v")]),
            claims(&[("s", &long, "v")]),
            claims(&[("s", "a", &long)]),
            vec![FactClaim::new("s", "a", "v"); MAX_CLAIMS + 1],
        ] {
            let report = verifier.verify(&big);
            assert_eq!(codes(&report), ["fact.oversize"]);
            assert_eq!(report.status, VerificationStatus::Fail);
        }
        assert_eq!(
            verifier
                .verify(&vec![
                    FactClaim::new("paris", "capital-of", "france");
                    MAX_CLAIMS
                ])
                .status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn verification_is_deterministic_read_only_and_named() {
        let host = host();
        let before = host.snapshot().canonical_digest();
        let verifier = FactVerifier::new(host.snapshot(), host.revision());
        let input = claims(&[("paris", "capital-of", "rome"), ("mars", "moons", "2")]);
        assert_eq!(verifier.verify(&input), verifier.verify(&input));
        assert_eq!(host.snapshot().canonical_digest(), before);
        assert_eq!(NamedVerifier::<Vec<FactClaim>>::name(&verifier), "fact");
    }
}
