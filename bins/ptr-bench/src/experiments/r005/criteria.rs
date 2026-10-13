//! Tests that pin `criteria.toml` (the rules `aggregate.py` applies) to this harness: every
//! rule is one the harness itself judges, in the harness's order, every counter a rule names
//! resolves in the harness's JSON, and a rule evaluated under the file's documented grammar
//! agrees with the harness's own verdict on a reduced run.

use super::metrics::{Check, Json, Status};

const CRITERIA: &str =
    include_str!("../../../../../experiments/runtime/R005-exact-prefix-reuse/criteria.toml");

#[derive(Debug, PartialEq)]
enum Operand<'a> {
    Int(i64),
    Decimal(f64),
    Reference(&'a str),
}

fn operand(token: &str) -> Operand<'_> {
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    let unsigned = token.strip_prefix('-').unwrap_or(token);
    match unsigned.split_once('.') {
        Some((whole, fraction)) if digits(whole) && digits(fraction) => {
            Operand::Decimal(token.parse().expect("a decimal literal"))
        }
        None if digits(unsigned) => Operand::Int(token.parse().expect("an integer literal")),
        _ => Operand::Reference(token),
    }
}

/// The harness JSON at the dotted path, or `None`.
fn resolve<'a>(root: &'a Json, path: &str) -> Option<&'a Json> {
    let mut at = root;
    for key in path.split('.') {
        match at {
            Json::Obj(pairs) => at = &pairs.iter().find(|(name, _)| name == key)?.1,
            _ => return None,
        }
    }
    Some(at)
}

/// A rule's verdict under the grammar of `criteria.toml`, or why it cannot be read.
fn evaluate(root: &Json, rule: &str) -> Result<bool, String> {
    let tokens: Vec<&str> = rule.split(' ').collect();
    let [left, op, right] = tokens[..] else {
        return Err(format!("not three tokens: {rule:?}"));
    };
    if !matches!(op, ">=" | "<=" | "==") {
        return Err(format!("unknown operator in {rule:?}"));
    }
    #[derive(Debug)]
    enum Value {
        Int(i64),
        Num(f64),
        Text(String),
    }
    let value = |token: &str, decimal: bool| -> Result<Value, String> {
        match operand(token) {
            Operand::Int(number) => Ok(Value::Int(number)),
            Operand::Decimal(number) => Ok(Value::Num(number)),
            Operand::Reference(path) => match resolve(root, path) {
                Some(Json::Int(number)) => Ok(Value::Int(*number)),
                Some(Json::Float(number)) if decimal => Ok(Value::Num(*number)),
                Some(Json::Str(text)) if op == "==" => Ok(Value::Text(text.clone())),
                other => Err(format!("{path} in {rule:?} is {other:?}")),
            },
        }
    };
    let decimal = matches!(operand(left), Operand::Decimal(_))
        || matches!(operand(right), Operand::Decimal(_));
    let (a, b) = (value(left, decimal)?, value(right, decimal)?);
    let ordering = match (a, b) {
        (Value::Int(a), Value::Int(b)) if !decimal => a.cmp(&b),
        (Value::Int(a), Value::Num(b)) => (a as f64).partial_cmp(&b).ok_or("NaN")?,
        (Value::Num(a), Value::Int(b)) => a.partial_cmp(&(b as f64)).ok_or("NaN")?,
        (Value::Num(a), Value::Num(b)) => a.partial_cmp(&b).ok_or("NaN")?,
        (Value::Text(a), Value::Text(b)) => a.cmp(&b),
        (a, b) => return Err(format!("types differ in {rule:?}: {a:?} and {b:?}")),
    };
    Ok(match op {
        ">=" => ordering.is_ge(),
        "<=" => ordering.is_le(),
        _ => ordering.is_eq(),
    })
}

/// The criteria rule a harness rule is written as: the counters that live under `members` and
/// `internal` in the JSON are named by their path, and the hit rate names its counter.
fn as_criteria(rule: &str) -> String {
    let mut text = rule.to_string();
    if text.starts_with(">= ") {
        text = format!("hit_rate {text}");
    }
    for (flat, group) in [
        ("member_cancelled_forks.", "cancelled_forks"),
        ("member_race_cases.", "race_cases"),
    ] {
        if let Some(rest) = text.strip_prefix(flat) {
            let (member, tail) = rest.split_once(' ').expect("a rule");
            text = format!("members.{member}.{group} {tail}");
        }
    }
    if text.starts_with("semantic_runtime_segment_")
        || text.starts_with("semantic_lifecycle_identity_")
    {
        text = format!("internal.{text}");
    }
    text.replace(
        ">= revoked_pinned_page_bytes_total",
        ">= internal.revoked_pinned_page_bytes_total",
    )
}

fn criteria() -> toml::Table {
    CRITERIA.parse().expect("criteria.toml is TOML")
}

fn rules(table: &toml::Table, key: &str) -> Vec<String> {
    table[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} is an array"))
        .iter()
        .map(|rule| rule.as_str().expect("a rule is a string").to_string())
        .collect()
}

fn judged(checks: &[Check]) -> Vec<&Check> {
    checks
        .iter()
        .filter(|check| check.status != Status::Unsupported)
        .collect()
}

#[test]
fn the_grammar_reads_literals_and_references_unambiguously() {
    assert_eq!(operand("4096"), Operand::Int(4096));
    assert_eq!(operand("-1"), Operand::Int(-1));
    assert_eq!(operand("0.4"), Operand::Decimal(0.4));
    assert_eq!(operand("stale_hits"), Operand::Reference("stale_hits"));
    assert_eq!(operand("a.b"), Operand::Reference("a.b"));
    assert_eq!(operand("1."), Operand::Reference("1."));
    let root = Json::obj([
        ("a", Json::Int(2)),
        ("f", Json::Float(0.5)),
        ("o", Json::obj([("n", Json::Int(3))])),
        ("s", Json::str("x")),
        ("t", Json::str("x")),
        ("b", Json::Bool(true)),
    ]);
    assert_eq!(evaluate(&root, "o.n >= a"), Ok(true));
    assert_eq!(evaluate(&root, "a == 3"), Ok(false));
    assert_eq!(evaluate(&root, "f >= 0.4"), Ok(true));
    assert_eq!(evaluate(&root, "s == t"), Ok(true));
    assert!(
        evaluate(&root, "f >= 1").is_err(),
        "a float needs a decimal literal"
    );
    assert!(evaluate(&root, "b >= 1").is_err());
    assert!(evaluate(&root, "missing >= 1").is_err());
    assert!(evaluate(&root, "a > 1").is_err());
    assert!(evaluate(&root, "a >=  1").is_err());
    assert!(evaluate(&root, "s >= t").is_err(), "strings only under ==");
}

#[test]
fn the_rules_are_the_rules_the_harness_judges_in_its_order() {
    let report = super::tests::reduced_report();
    let table = criteria();
    for (key, checks) in [
        ("coverage", &report.coverage),
        ("falsifiers", &report.falsifiers),
    ] {
        let expected: Vec<String> = judged(checks)
            .iter()
            .map(|check| as_criteria(&check.rule))
            .collect();
        assert_eq!(rules(&table, key), expected, "{key}");
    }
}

#[test]
fn every_counter_a_rule_names_resolves_and_the_rule_agrees_with_the_harness() {
    let report = super::tests::reduced_report();
    let table = criteria();
    for (key, checks) in [
        ("coverage", &report.coverage),
        ("falsifiers", &report.falsifiers),
    ] {
        let rules = rules(&table, key);
        for (rule, check) in rules.iter().zip(judged(checks)) {
            let held =
                evaluate(&report.json, rule).unwrap_or_else(|reason| panic!("{key}: {reason}"));
            assert_eq!(held, check.status == Status::Pass, "{key}: {rule}");
        }
    }
}

#[test]
fn the_static_evidence_is_the_one_the_manifest_and_config_name() {
    let table = criteria();
    let commands = rules(&table, "static_evidence");
    assert_eq!(commands.len(), 2);
    let config: toml::Table = super::params::CONFIG_TOML.parse().expect("config.toml");
    let pinned: Vec<String> = config["preregistration"]["static_evidence"]
        .as_array()
        .expect("static_evidence")
        .iter()
        .map(|item| item.as_str().expect("a command").to_string())
        .collect();
    assert_eq!(commands, pinned);
}
