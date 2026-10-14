//! Typed semantic objects and lossless conversion from/to the `String` buckets of
//! [`SemanticCapsule`](crate::SemanticCapsule).
//!
//! The capsule keeps its `String` buckets unchanged. A relation bucket entry is the text
//! `"{subject} -[{predicate}]-> {object}"`; conversion in both directions is exact for every
//! relation the formatter accepts, and entries that do not follow the format are reported as
//! [`SemanticConversionError::RelationFormat`] instead of being guessed at.

use crate::SemanticCapsule;

const REL_OPEN: &str = " -[";
const REL_CLOSE: &str = "]-> ";

/// A typed semantic object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticObject {
    Goal(String),
    Constraint(String),
    KnownFact(String),
    Hypothesis(String),
    Unknown(String),
    Relation {
        subject: String,
        predicate: String,
        object: String,
    },
}

/// The six `String` buckets of a capsule, detached from it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SemanticBuckets {
    pub goals: Vec<String>,
    pub hard_constraints: Vec<String>,
    pub known: Vec<String>,
    pub hypotheses: Vec<String>,
    pub unknowns: Vec<String>,
    pub relations: Vec<String>,
}

/// Why a conversion failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticConversionError {
    /// A relation bucket entry is not of the form `subject -[predicate]-> object`.
    RelationFormat { raw: String },
    /// A relation field is empty or contains the delimiter and could not be formatted exactly.
    UnrepresentableRelation { field: &'static str },
}

impl SemanticObject {
    /// Formats a relation as its bucket text.
    pub fn format_relation(
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<String, SemanticConversionError> {
        if subject.is_empty() || subject.contains(REL_OPEN) {
            return Err(SemanticConversionError::UnrepresentableRelation { field: "subject" });
        }
        if predicate.is_empty() || predicate.contains(REL_CLOSE) {
            return Err(SemanticConversionError::UnrepresentableRelation { field: "predicate" });
        }
        if object.is_empty() {
            return Err(SemanticConversionError::UnrepresentableRelation { field: "object" });
        }
        Ok(format!("{subject}{REL_OPEN}{predicate}{REL_CLOSE}{object}"))
    }

    /// Parses a relation bucket entry.
    pub fn parse_relation(raw: &str) -> Result<Self, SemanticConversionError> {
        let error = || SemanticConversionError::RelationFormat {
            raw: raw.to_owned(),
        };
        let (subject, rest) = raw.split_once(REL_OPEN).ok_or_else(error)?;
        let (predicate, object) = rest.split_once(REL_CLOSE).ok_or_else(error)?;
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return Err(error());
        }
        Ok(Self::Relation {
            subject: subject.to_owned(),
            predicate: predicate.to_owned(),
            object: object.to_owned(),
        })
    }
}

/// Converts buckets into typed objects in a fixed order: goals, constraints, known facts,
/// hypotheses, unknowns, relations.
pub fn semantic_objects_from_buckets(
    buckets: &SemanticBuckets,
) -> Result<Vec<SemanticObject>, SemanticConversionError> {
    let mut out = Vec::new();
    out.extend(buckets.goals.iter().cloned().map(SemanticObject::Goal));
    out.extend(
        buckets
            .hard_constraints
            .iter()
            .cloned()
            .map(SemanticObject::Constraint),
    );
    out.extend(buckets.known.iter().cloned().map(SemanticObject::KnownFact));
    out.extend(
        buckets
            .hypotheses
            .iter()
            .cloned()
            .map(SemanticObject::Hypothesis),
    );
    out.extend(
        buckets
            .unknowns
            .iter()
            .cloned()
            .map(SemanticObject::Unknown),
    );
    for relation in &buckets.relations {
        out.push(SemanticObject::parse_relation(relation)?);
    }
    Ok(out)
}

/// Converts typed objects back into buckets, preserving the order within each kind.
pub fn semantic_buckets_from_objects(
    objects: &[SemanticObject],
) -> Result<SemanticBuckets, SemanticConversionError> {
    let mut buckets = SemanticBuckets::default();
    for object in objects {
        match object {
            SemanticObject::Goal(text) => buckets.goals.push(text.clone()),
            SemanticObject::Constraint(text) => buckets.hard_constraints.push(text.clone()),
            SemanticObject::KnownFact(text) => buckets.known.push(text.clone()),
            SemanticObject::Hypothesis(text) => buckets.hypotheses.push(text.clone()),
            SemanticObject::Unknown(text) => buckets.unknowns.push(text.clone()),
            SemanticObject::Relation {
                subject,
                predicate,
                object,
            } => buckets
                .relations
                .push(SemanticObject::format_relation(subject, predicate, object)?),
        }
    }
    Ok(buckets)
}

impl SemanticCapsule {
    /// The capsule's buckets as typed objects.
    pub fn typed_objects(&self) -> Result<Vec<SemanticObject>, SemanticConversionError> {
        semantic_objects_from_buckets(&SemanticBuckets {
            goals: self.goals.clone(),
            hard_constraints: self.hard_constraints.clone(),
            known: self.known.clone(),
            hypotheses: self.hypotheses.clone(),
            unknowns: self.unknowns.clone(),
            relations: self.relations.clone(),
        })
    }

    /// Replaces the six buckets with the given typed objects. On error the capsule is unchanged.
    pub fn set_typed_objects(
        &mut self,
        objects: &[SemanticObject],
    ) -> Result<(), SemanticConversionError> {
        let buckets = semantic_buckets_from_objects(objects)?;
        self.goals = buckets.goals;
        self.hard_constraints = buckets.hard_constraints;
        self.known = buckets.known;
        self.hypotheses = buckets.hypotheses;
        self.unknowns = buckets.unknowns;
        self.relations = buckets.relations;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_round_trips_even_with_delimiter_like_object_text() {
        let text = SemanticObject::format_relation("a", "p", "b -[c]-> d").unwrap();
        assert_eq!(
            SemanticObject::parse_relation(&text),
            Ok(SemanticObject::Relation {
                subject: "a".into(),
                predicate: "p".into(),
                object: "b -[c]-> d".into()
            })
        );
    }

    #[test]
    fn malformed_and_unrepresentable_relations_are_typed_errors() {
        assert_eq!(
            SemanticObject::parse_relation("just text"),
            Err(SemanticConversionError::RelationFormat {
                raw: "just text".into()
            })
        );
        assert!(SemanticObject::parse_relation("a -[p]-> ").is_err());
        assert_eq!(
            SemanticObject::format_relation("a -[x", "p", "o"),
            Err(SemanticConversionError::UnrepresentableRelation { field: "subject" })
        );
        assert_eq!(
            SemanticObject::format_relation("a", "p]-> q", "o"),
            Err(SemanticConversionError::UnrepresentableRelation { field: "predicate" })
        );
    }
}
