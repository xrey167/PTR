//! Loading operator-routing v1 from its TSV twin.
//!
//! Each line is `id, label code, R, B, comma-separated tokens, facts`, facts as
//! `role,e,c,validity,entity;...` in codebook codes (see
//! benchmarks/operator-routing/README.md). Loading refuses to proceed when the
//! bytes are not the pinned ones or when any label disagrees with this binary's
//! own implementation of the rule.

use crate::rng::{fnv1a64, fnv1a64_start};
use crate::rule::{self, RuleFact};
use ptr_types::{Codebook, CognitiveType, EpistemicState, SemanticRole, Validity};
use std::path::Path;

pub const SLOTS: usize = 6;

/// The eight splits, in the order their bytes enter the data FNV.
pub const SPLITS: [&str; 8] = [
    "train",
    "val",
    "test_iid",
    "ood_compose_epi",
    "ood_compose_regime",
    "ood_distractors",
    "ood_validity",
    "ood_payload",
];

/// The six splits a trained arm is scored on.
pub const TEST_SPLITS: [&str; 6] = [
    "test_iid",
    "ood_compose_epi",
    "ood_compose_regime",
    "ood_distractors",
    "ood_validity",
    "ood_payload",
];

#[derive(Clone, Debug)]
pub struct Fact {
    pub role: SemanticRole,
    pub epistemic: EpistemicState,
    pub confidence: f32,
    pub bucket: usize,
    pub validity: Validity,
    pub entity: u32,
}

#[derive(Clone, Debug)]
pub struct Example {
    pub label: usize,
    pub tokens: Vec<i64>,
    pub facts: Vec<Fact>,
}

pub struct Split {
    pub name: &'static str,
    pub examples: Vec<Example>,
    /// FNV-1a-64 of one lowercase hex digit per gold label, in file order.
    pub label_fnv64: u64,
}

pub struct Dataset {
    pub fnv64: u64,
    pub splits: Vec<Split>,
}

impl Dataset {
    /// Borrow a split by exact name, panicking if the dataset does not contain it.
    pub fn split(&self, name: &str) -> &Split {
        self.splits
            .iter()
            .find(|split| split.name == name)
            .unwrap_or_else(|| panic!("no split {name}"))
    }
}

/// Codebook code -> member, for one family, built from `code_of` over its members.
fn by_code<T: CognitiveType>(members: &[T]) -> Vec<T> {
    let mut table: Vec<Option<T>> = vec![None; members.len()];
    for member in members {
        let code = usize::from(Codebook::V1.code_of(*member).expect("v1 member").index());
        table[code] = Some(*member);
    }
    table
        .into_iter()
        .map(|member| member.expect("every code assigned"))
        .collect()
}

pub struct Tables {
    pub roles: Vec<SemanticRole>,
    pub states: Vec<EpistemicState>,
    pub validities: Vec<Validity>,
}

/// Build role, epistemic-state, and validity lookup tables indexed by v1 codes.
pub fn tables() -> Tables {
    use EpistemicState as E;
    use SemanticRole as R;
    Tables {
        roles: by_code(&[
            R::Goal,
            R::Constraint,
            R::Claim,
            R::Evidence,
            R::Resource,
            R::Capability,
            R::Relation,
            R::Procedure,
            R::Action,
        ]),
        states: by_code(&[
            E::Unknown,
            E::Assumed,
            E::Hypothesis,
            E::Observed,
            E::Inferred,
            E::Verified,
        ]),
        validities: by_code(&[
            Validity::Live,
            Validity::Superseded,
            Validity::Revoked,
            Validity::Disputed,
        ]),
    }
}

/// Read a zero-based field, returning an error with the line number if absent.
fn field<'a>(fields: &[&'a str], index: usize, line: usize) -> Result<&'a str, String> {
    fields
        .get(index)
        .copied()
        .ok_or_else(|| format!("line {line}: missing field {index}"))
}

/// Parse a field, reporting its name and line number on conversion failure.
fn number<T: std::str::FromStr>(text: &str, what: &str, line: usize) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("line {line}: {what} is not a number: {text:?}"))
}

/// Parse TSV examples, re-derive gold labels, and compute their label digest.
///
/// Returns an error for wrong field or fact counts, numeric conversion failures,
/// invalid regime, budget, label, or fact-table indices, or label disagreement.
/// Token ranges and entity-table bounds are not validated here.
fn parse_split(name: &'static str, text: &str, tables: &Tables) -> Result<Split, String> {
    let mut examples = Vec::new();
    let mut labels = String::new();
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let fields: Vec<&str> = raw.split('\t').collect();
        if fields.len() != 6 {
            return Err(format!(
                "{name} line {line}: expected 6 fields, got {}",
                fields.len()
            ));
        }
        let id = field(&fields, 0, line)?;
        let label: usize = number(field(&fields, 1, line)?, "label", line)?;
        let regime: usize = number(field(&fields, 2, line)?, "regime", line)?;
        let budget: usize = number(field(&fields, 3, line)?, "budget", line)?;
        if regime > 3 || budget > 2 || label >= rule::OPERATORS {
            return Err(format!("{name} {id}: regime, budget or label out of range"));
        }
        let tokens = field(&fields, 4, line)?
            .split(',')
            .map(|token| number::<i64>(token, "token", line))
            .collect::<Result<Vec<_>, _>>()?;
        let mut facts = Vec::with_capacity(SLOTS);
        for part in field(&fields, 5, line)?.split(';') {
            let parts: Vec<&str> = part.split(',').collect();
            if parts.len() != 5 {
                return Err(format!("{name} {id}: a fact needs 5 fields: {part:?}"));
            }
            let role: usize = number(parts[0], "role", line)?;
            let state: usize = number(parts[1], "epistemic", line)?;
            let confidence: f64 = number(parts[2], "confidence", line)?;
            let validity: usize = number(parts[3], "validity", line)?;
            let entity: u32 = number(parts[4], "entity", line)?;
            let bucket = (5.0 * confidence) as usize;
            if role >= tables.roles.len()
                || state >= tables.states.len()
                || validity >= tables.validities.len()
                || bucket > 4
            {
                return Err(format!(
                    "{name} {id}: a fact field is out of range: {part:?}"
                ));
            }
            facts.push(Fact {
                role: tables.roles[role],
                epistemic: tables.states[state],
                confidence: confidence as f32,
                bucket,
                validity: tables.validities[validity],
                entity,
            });
        }
        if facts.len() != SLOTS {
            return Err(format!(
                "{name} {id}: expected {SLOTS} facts, got {}",
                facts.len()
            ));
        }
        let rule_facts: Vec<RuleFact> = facts
            .iter()
            .map(|fact| RuleFact {
                role: fact.role,
                epistemic: fact.epistemic,
                bucket: fact.bucket,
                validity: fact.validity,
            })
            .collect();
        let derived = rule::label(&rule_facts, regime, budget);
        if derived != label {
            return Err(format!(
                "{name} {id}: the file says label {label}, this binary's rule says {derived}; refusing to train on labels the two implementations disagree on"
            ));
        }
        labels.push(char::from_digit(label as u32, 16).expect("a hex digit"));
        examples.push(Example {
            label,
            tokens,
            facts,
        });
    }
    Ok(Split {
        name,
        examples,
        label_fnv64: fnv1a64(labels.as_bytes(), fnv1a64_start()),
    })
}

/// Load every split, check the pinned FNV-1a-64 over the eight TSV files, and
/// re-derive every label.
///
/// # Errors
///
/// Returns an error for file I/O, invalid UTF-8, a split rejected by the parser,
/// or a combined digest that differs from `expected_fnv64`.
pub fn load(directory: &Path, expected_fnv64: u64) -> Result<Dataset, String> {
    let tables = tables();
    let mut fnv = fnv1a64_start();
    let mut splits = Vec::with_capacity(SPLITS.len());
    for name in SPLITS {
        let path = directory.join(format!("{name}.tsv"));
        let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        fnv = fnv1a64(&bytes, fnv);
        let text = String::from_utf8(bytes).map_err(|_| format!("{name}.tsv is not UTF-8"))?;
        splits.push(parse_split(name, &text, &tables)?);
    }
    if fnv != expected_fnv64 {
        return Err(format!(
            "the data FNV-1a-64 is {fnv:016x}, but {expected_fnv64:016x} is pinned: these are not the frozen splits"
        ));
    }
    Ok(Dataset { fnv64: fnv, splits })
}
