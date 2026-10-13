//! The one table of finding codes the task verifiers emit, and the repair role each maps to.
//!
//! A code is stable: it is what [`ptr_router`'s `FindingRouter`] routes on, what the trace
//! records, and what a repair Pod is registered for. Each code is either *hard* (a deterministic
//! contradiction of the candidate: status `Fail`, the router sends it to the repair role) or
//! *soft* (the verifier could not judge, or notes a limit: it never turns a report into a repair
//! request; an `Unknown` report is handled by the router's fallback policy). A learned score has
//! no entry here and cannot override a hard code (invariant 11).
//!
//! | code | hard | repair role | meaning |
//! |------|------|-------------|---------|
//! | `arith.format` | yes | `format-repair` | candidate is not an integer or `a/b` rational |
//! | `arith.oversize` | yes | `format-repair` | candidate longer than the bound |
//! | `arith.value` | yes | `value-repair` | candidate parses but is not exactly the task's value |
//! | `arith.bounds` | yes | `value-repair` | candidate magnitude beyond the bound |
//! | `arith.task` | no | none | task unparsable, oversize, overflowing or dividing by zero (`Unknown`) |
//! | `json.parse` | yes | `format-repair` | candidate is not valid JSON |
//! | `json.oversize` | yes | `format-repair` | candidate longer than the bound |
//! | `json.depth` | yes | `format-repair` | candidate nested deeper than the bound |
//! | `json.type` | yes | `schema-repair` | `type` violated at a pointer |
//! | `json.required` | yes | `schema-repair` | a required property is missing |
//! | `json.enum` | yes | `schema-repair` | value is not one of the `enum` members |
//! | `json.range` | yes | `schema-repair` | `minimum` / `maximum` violated |
//! | `json.length` | yes | `schema-repair` | `minLength` / `maxLength` violated |
//! | `json.extra` | yes | `schema-repair` | property not allowed by `additionalProperties: false` |
//! | `json.truncated` | no | none | more violations than the report holds; the rest are not listed |
//! | `json.schema` | no | none | the task's schema is invalid or outside the pinned subset (`Unknown`) |
//! | `code.syntax` | yes | `code-repair` | candidate does not parse, or lacks the entry function |
//! | `code.oversize` | yes | `format-repair` | candidate source longer than the bound |
//! | `code.test_failed` | yes | `code-repair` | a task test returned another value, or trapped |
//! | `code.timeout` | yes | `resource-repair` | a test exhausted its step (CPU) budget |
//! | `code.resource` | yes | `resource-repair` | a test exceeded the call-depth, memory or output bound |
//! | `code.task` | no | none | the task has no tests, or too many (`Unknown`) |
//! | `fact.contradicted` | yes | `fact-repair` | the snapshot holds another value for the claim |
//! | `fact.malformed` | yes | `format-repair` | a claim has an empty or ill-formed field |
//! | `fact.oversize` | yes | `format-repair` | too many claims, or a field beyond the bound |
//! | `fact.unknown` | no | none | the snapshot holds nothing for the claim (`Unknown`, not `Fail`) |
//! | `fact.stale_snapshot` | no | none | the snapshot is not the live revision (`Unknown`) |
//!
//! Notes on individual rows:
//!
//! * `code.oversize` routes to `format-repair`, not to `code-repair`: the defect is the size of
//!   the text (the bound is on bytes, before anything is parsed), which is the same repair as an
//!   oversize number or document (rewrite it shorter), and a code-repair Pod would be asked to
//!   fix behaviour that was never run. The code is stable once released.
//! * `code.test_failed` is hard and routed to `code-repair` for a wrong value, a wrong printed
//!   output and a runtime trap alike. Its message names the test id only unless the task marks
//!   the test visible (`TestVisibility`), so a repair Pod cannot read held-out answers from it.
//! * `json.truncated` follows the capped violations only if a further violation exists.
//! * `fact.*` messages name the snapshot revision; the structured revision is
//!   `FactVerifier::verify_bound`, because `VerificationReport` has no field for it.
//!
//! `ptr-router` builds its code routes from this table
//! (`RepairRoleAssignments` in `ptr-router`).

use crate::Finding;

/// What kind of repair Pod a hard code is routed to.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RepairRole {
    /// Rewrite the candidate so it is well-formed (parse, size, nesting).
    Format,
    /// Recompute the value.
    Value,
    /// Make a well-formed document satisfy the schema.
    Schema,
    /// Fix the program (syntax or behaviour).
    Code,
    /// Make the program terminate within its resource bounds.
    Resource,
    /// Replace a claim the snapshot contradicts.
    Fact,
}

impl RepairRole {
    /// Every role, in a fixed order.
    pub const ALL: [RepairRole; 6] = [
        Self::Format,
        Self::Value,
        Self::Schema,
        Self::Code,
        Self::Resource,
        Self::Fact,
    ];

    /// The stable role name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Format => "format-repair",
            Self::Value => "value-repair",
            Self::Schema => "schema-repair",
            Self::Code => "code-repair",
            Self::Resource => "resource-repair",
            Self::Fact => "fact-repair",
        }
    }
}

/// One row of the code table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodeSpec {
    pub code: &'static str,
    /// A hard code is a deterministic contradiction (`Fail`); a soft one is never repaired.
    pub hard: bool,
    /// The repair role of a hard code; soft codes have none.
    pub role: Option<RepairRole>,
}

const fn hard(code: &'static str, role: RepairRole) -> CodeSpec {
    CodeSpec {
        code,
        hard: true,
        role: Some(role),
    }
}

const fn soft(code: &'static str) -> CodeSpec {
    CodeSpec {
        code,
        hard: false,
        role: None,
    }
}

/// The table, in the order of the module documentation.
pub const CODE_TABLE: &[CodeSpec] = &[
    hard("arith.format", RepairRole::Format),
    hard("arith.oversize", RepairRole::Format),
    hard("arith.value", RepairRole::Value),
    hard("arith.bounds", RepairRole::Value),
    soft("arith.task"),
    hard("json.parse", RepairRole::Format),
    hard("json.oversize", RepairRole::Format),
    hard("json.depth", RepairRole::Format),
    hard("json.type", RepairRole::Schema),
    hard("json.required", RepairRole::Schema),
    hard("json.enum", RepairRole::Schema),
    hard("json.range", RepairRole::Schema),
    hard("json.length", RepairRole::Schema),
    hard("json.extra", RepairRole::Schema),
    soft("json.truncated"),
    soft("json.schema"),
    hard("code.syntax", RepairRole::Code),
    hard("code.oversize", RepairRole::Format),
    hard("code.test_failed", RepairRole::Code),
    hard("code.timeout", RepairRole::Resource),
    hard("code.resource", RepairRole::Resource),
    soft("code.task"),
    hard("fact.contradicted", RepairRole::Fact),
    hard("fact.malformed", RepairRole::Format),
    hard("fact.oversize", RepairRole::Format),
    soft("fact.unknown"),
    soft("fact.stale_snapshot"),
];

/// The row of `code`, if it is one of the table's codes.
pub fn code_spec(code: &str) -> Option<&'static CodeSpec> {
    CODE_TABLE.iter().find(|spec| spec.code == code)
}

/// The repair role a hard `code` is routed to. `None` for a soft or unknown code.
pub fn repair_role_for(code: &str) -> Option<RepairRole> {
    code_spec(code).and_then(|spec| spec.role)
}

/// The hard codes routed to `role`, in table order.
pub fn codes_for_role(role: RepairRole) -> Vec<&'static str> {
    CODE_TABLE
        .iter()
        .filter(|spec| spec.role == Some(role))
        .map(|spec| spec.code)
        .collect()
}

/// A finding whose hardness comes from the table, so a verifier cannot disagree with it.
pub(crate) fn finding(code: &'static str, message: String) -> Finding {
    let spec = code_spec(code);
    debug_assert!(spec.is_some(), "finding code {code} is not in the table");
    Finding {
        code: code.into(),
        message,
        hard: spec.is_some_and(|spec| spec.hard),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_short_and_consistent() {
        let mut seen = std::collections::BTreeSet::new();
        for spec in CODE_TABLE {
            assert!(seen.insert(spec.code), "duplicate {}", spec.code);
            assert!(spec.code.len() <= 32);
            assert_eq!(spec.hard, spec.role.is_some(), "{}", spec.code);
        }
    }

    #[test]
    fn module_documentation_lists_exactly_the_table() {
        let source = include_str!("codes.rs");
        for spec in CODE_TABLE {
            let row = format!(
                "//! | `{}` | {} | ",
                spec.code,
                if spec.hard { "yes" } else { "no" }
            );
            assert!(
                source.contains(&row),
                "{} missing from the doc table",
                spec.code
            );
            let role = spec
                .role
                .map_or("none".to_owned(), |r| r.as_str().to_owned());
            let cell = format!("| `{role}` |");
            let cell = if spec.role.is_none() {
                "| none |".to_owned()
            } else {
                cell
            };
            assert!(source.contains(&cell), "{} role cell", spec.code);
        }
        let rows = source
            .lines()
            .filter(|line| {
                line.starts_with("//! | `") && line.contains("` | yes |")
                    || line.starts_with("//! | `") && line.contains("` | no |")
            })
            .count();
        assert_eq!(rows, CODE_TABLE.len());
    }

    #[test]
    fn every_role_has_a_code_and_soft_codes_have_none() {
        for role in RepairRole::ALL {
            assert!(!codes_for_role(role).is_empty(), "{role:?}");
        }
        assert_eq!(repair_role_for("fact.unknown"), None);
        assert_eq!(repair_role_for("nope"), None);
        assert_eq!(repair_role_for("json.type"), Some(RepairRole::Schema));
    }
}
