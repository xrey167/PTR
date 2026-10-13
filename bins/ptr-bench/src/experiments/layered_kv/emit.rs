//! The emitter shared by the R009 and R010 harnesses: one-line JSON, the counters of a run,
//! and the judgement of the run's own output under the rule grammar of `criteria.toml`.
//!
//! Every metric the manifest names has exactly one place in the output: a number, an object
//! (a grouped metric), a string (a digest) or `null` together with a reason under
//! `unsupported`. A metric the harness neither filled nor declared unsupported is a harness
//! gap: it prints as `null`, is listed under `harness_gaps` and fails the run, so nothing is
//! dropped silently.
//!
//! Fractions are never floats. A fraction is an object of integer numerators and
//! denominators with an integer `cross_difference` (`n * pd - pn * d`), and a rule compares
//! that integer with `0` (owner decision 6).
//!
//! Rules are strings of exactly three tokens, `<left> <op> <right>` with `op` one of `>=`,
//! `<=`, `==`. A token that matches `-?[0-9]+` is an integer literal; any other token is a
//! path of object keys from the root of the printed object, separated by `.`. The same
//! grammar is what the experiment's `criteria.toml` and `aggregate.py` read, and the harness
//! judges its own output with [`evaluate`], so a rule cannot mean one thing here and another
//! there.

use std::collections::BTreeMap;

/// A JSON value, rendered on one line.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn str(text: impl Into<String>) -> Self {
        Self::Str(text.into())
    }

    pub fn obj<const N: usize>(pairs: [(&str, Json); N]) -> Self {
        Self::Obj(
            pairs
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    pub fn strs<S: AsRef<str>>(items: &[S]) -> Self {
        Self::Arr(items.iter().map(|item| Self::str(item.as_ref())).collect())
    }

    pub fn ints(items: impl IntoIterator<Item = i64>) -> Self {
        Self::Arr(items.into_iter().map(Self::Int).collect())
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
            Self::Int(number) => out.push_str(&number.to_string()),
            Self::Float(number) if number.is_finite() => {
                // `{:?}` keeps a decimal point, so a float stays a float.
                out.push_str(&format!("{number:?}"));
            }
            Self::Float(_) => out.push_str("null"),
            Self::Str(text) => out.push_str(&super::super::json::json_string(text)),
            Self::Arr(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Self::Obj(pairs) => {
                out.push('{');
                for (index, (key, value)) in pairs.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&super::super::json::json_string(key));
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// An exact fraction `numerator / denominator` next to the one it is predicted to equal. The
/// two are equal iff `cross_difference` is `0`; nothing here is a float.
pub fn fraction_json(
    numerator: i64,
    denominator: i64,
    predicted_numerator: i64,
    predicted_denominator: i64,
) -> Json {
    Json::obj([
        ("numerator", Json::Int(numerator)),
        ("denominator", Json::Int(denominator)),
        ("predicted_numerator", Json::Int(predicted_numerator)),
        ("predicted_denominator", Json::Int(predicted_denominator)),
        (
            "cross_difference",
            Json::Int(numerator * predicted_denominator - predicted_numerator * denominator),
        ),
    ])
}

/// The metrics a run neither produced nor declared unsupported, each with its reason.
pub type Gaps = Vec<(String, String)>;

/// What a metric list looks like to a [`Tally`].
#[derive(Clone, Copy, Debug)]
pub struct MetricList {
    /// The manifest's metric names, in its order.
    pub names: &'static [&'static str],
    /// Metrics printed as an object built from the dotted counters under their name.
    pub grouped: &'static [&'static str],
}

/// The counters of one run. Names are the manifest's metric names; a dotted name is one
/// member of a grouped metric. Anything else is an internal counter, printed under
/// `internal`.
#[derive(Debug)]
pub struct Tally {
    list: MetricList,
    ints: BTreeMap<String, i64>,
    json: BTreeMap<String, Json>,
    unsupported: BTreeMap<String, String>,
    gaps: BTreeMap<String, String>,
}

impl Tally {
    pub fn new(list: MetricList) -> Self {
        Self {
            list,
            ints: BTreeMap::new(),
            json: BTreeMap::new(),
            unsupported: BTreeMap::new(),
            gaps: BTreeMap::new(),
        }
    }

    pub fn add(&mut self, name: &str, amount: i64) {
        *self.ints.entry(name.to_string()).or_insert(0) += amount;
    }

    pub fn set(&mut self, name: &str, value: i64) {
        self.ints.insert(name.to_string(), value);
    }

    /// Every integer counter, in name order (a harness that runs a case on its own tally
    /// hands the counters back to the run's).
    #[cfg_attr(not(feature = "ledger-experiments"), allow(dead_code))]
    pub fn counters(&self) -> Vec<(String, i64)> {
        self.ints
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect()
    }

    /// Add the counters of another tally to this one.
    #[cfg_attr(not(feature = "ledger-experiments"), allow(dead_code))]
    pub fn absorb(&mut self, counters: &[(String, i64)]) {
        for (name, value) in counters {
            self.add(name, *value);
        }
    }

    pub fn set_json(&mut self, name: &str, value: Json) {
        self.json.insert(name.to_string(), value);
    }

    /// Declare a metric or case the code under test cannot honestly produce.
    pub fn unsupported(&mut self, name: &str, reason: &str) {
        self.unsupported
            .insert(name.to_string(), reason.to_string());
    }

    /// Declare a metric the harness could not produce yet: unlike `unsupported`, this fails
    /// the run (it is listed under `harness_gaps` with its reason).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn gap(&mut self, name: &str, reason: &str) {
        self.gaps.insert(name.to_string(), reason.to_string());
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn get(&self, name: &str) -> i64 {
        self.ints.get(name).copied().unwrap_or(0)
    }

    pub fn json_or_null(&self, name: &str) -> Json {
        self.json.get(name).cloned().unwrap_or(Json::Null)
    }

    /// The members `name.<member>` of a grouped metric, in name order.
    pub fn group(&self, name: &str) -> Vec<(String, i64)> {
        let prefix = format!("{name}.");
        self.ints
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&prefix)
                    .map(|rest| (rest.to_string(), *value))
            })
            .collect()
    }

    /// Make sure every member of a grouped metric exists (at zero), so a cell that never ran
    /// prints a zero rather than missing.
    pub fn seed_group<S: AsRef<str>>(&mut self, name: &str, members: &[S]) {
        for member in members {
            self.ints
                .entry(format!("{name}.{}", member.as_ref()))
                .or_insert(0);
        }
    }

    fn render_metric(&self, name: &str) -> Option<Json> {
        if let Some(value) = self.json.get(name) {
            return Some(value.clone());
        }
        if self.list.grouped.contains(&name) {
            let members = self.group(name);
            if !members.is_empty() {
                return Some(Json::Obj(
                    members
                        .into_iter()
                        .map(|(member, value)| (member, Json::Int(value)))
                        .collect(),
                ));
            }
        }
        self.ints.get(name).map(|value| Json::Int(*value))
    }

    /// The metrics of the manifest as `(name, value)` in manifest order, plus the gaps: the
    /// names the harness neither filled nor declared unsupported, each with its reason.
    pub fn metrics(&self) -> (Vec<(String, Json)>, Gaps) {
        let mut printed = Vec::new();
        let mut gaps = Vec::new();
        for name in self.list.names {
            if let Some(reason) = self.gaps.get(*name) {
                printed.push((name.to_string(), Json::Null));
                gaps.push((name.to_string(), reason.clone()));
            } else if self.unsupported.contains_key(*name) {
                printed.push((name.to_string(), Json::Null));
            } else if let Some(value) = self.render_metric(name) {
                printed.push((name.to_string(), value));
            } else {
                printed.push((name.to_string(), Json::Null));
                gaps.push((name.to_string(), "no value and no declared reason".into()));
            }
        }
        (printed, gaps)
    }

    /// Counters that are no manifest metric, for the reader of a dry run. A dotted name nests
    /// (`a.b` is the member `b` of the object `a`), so a rule can name it by path.
    pub fn internal(&self) -> Json {
        let named = |key: &str| {
            let base = key.split('.').next().unwrap_or(key);
            self.list.names.contains(&base)
        };
        let mut root: Vec<(String, Json)> = Vec::new();
        for (key, value) in self.ints.iter().filter(|(key, _)| !named(key)) {
            let parts: Vec<&str> = key.split('.').collect();
            nest(&mut root, &parts, *value);
        }
        Json::Obj(root)
    }

    pub fn unsupported_json(&self) -> Json {
        Json::Obj(
            self.unsupported
                .iter()
                .map(|(name, reason)| (name.clone(), Json::str(reason.clone())))
                .collect(),
        )
    }

    pub fn unsupported_names(&self) -> Vec<String> {
        self.unsupported.keys().cloned().collect()
    }
}

fn nest(level: &mut Vec<(String, Json)>, parts: &[&str], value: i64) {
    let Some((first, rest)) = parts.split_first() else {
        return;
    };
    if rest.is_empty() {
        level.push(((*first).to_string(), Json::Int(value)));
        return;
    }
    let position = level.iter().position(|(name, _)| name == first);
    let index = position.unwrap_or_else(|| {
        level.push(((*first).to_string(), Json::Obj(Vec::new())));
        level.len() - 1
    });
    if let Json::Obj(inner) = &mut level[index].1 {
        nest(inner, rest, value);
    }
}

/// Whether a prerequisite or falsifier holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    /// Coverage: the prerequisite was met. Falsifier: the criterion held.
    Pass,
    /// Coverage: unmet (the run is vacuous). Falsifier: tripped (the experiment is falsified).
    Fail,
    /// The code under test cannot produce the value, so nothing is judged.
    Unsupported,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Unsupported => "unsupported",
        }
    }
}

/// One judged rule.
#[derive(Clone, Debug)]
pub struct Check {
    pub rule: String,
    pub observed: Json,
    pub status: Status,
}

impl Check {
    pub fn json(&self) -> Json {
        Json::obj([
            ("rule", Json::str(self.rule.clone())),
            ("observed", self.observed.clone()),
            ("status", Json::str(self.status.name())),
        ])
    }
}

/// How many checks have each status.
pub fn count(checks: &[Check], status: Status) -> usize {
    checks.iter().filter(|check| check.status == status).count()
}

/// The harness JSON at the dotted path, or `None`.
pub fn resolve<'a>(root: &'a Json, path: &str) -> Option<&'a Json> {
    let mut at = root;
    for key in path.split('.') {
        match at {
            Json::Obj(pairs) => at = &pairs.iter().find(|(name, _)| name == key)?.1,
            _ => return None,
        }
    }
    Some(at)
}

fn is_literal(token: &str) -> bool {
    let digits = token.strip_prefix('-').unwrap_or(token);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

/// One side of a rule: an integer, or a digest string (only under `==`).
#[derive(Debug, PartialEq)]
enum Operand {
    Int(i64),
    Text(String),
}

fn operand(root: &Json, token: &str, op: &str) -> Result<Operand, String> {
    if is_literal(token) {
        return token
            .parse()
            .map(Operand::Int)
            .map_err(|_| format!("{token} is not an integer"));
    }
    match resolve(root, token) {
        Some(Json::Int(number)) => Ok(Operand::Int(*number)),
        Some(Json::Str(text)) if op == "==" => Ok(Operand::Text(text.clone())),
        other => Err(format!("{token} is {other:?}")),
    }
}

/// A rule's verdict under the grammar above, or why it cannot be read.
pub fn evaluate(root: &Json, rule: &str) -> Result<bool, String> {
    let tokens: Vec<&str> = rule.split(' ').collect();
    let [left, op, right] = tokens[..] else {
        return Err(format!("not three tokens: {rule:?}"));
    };
    if !matches!(op, ">=" | "<=" | "==") {
        return Err(format!("unknown operator in {rule:?}"));
    }
    let (a, b) = (operand(root, left, op)?, operand(root, right, op)?);
    match (a, b) {
        (Operand::Int(a), Operand::Int(b)) => Ok(match op {
            ">=" => a >= b,
            "<=" => a <= b,
            _ => a == b,
        }),
        (Operand::Text(a), Operand::Text(b)) => Ok(a == b),
        (a, b) => Err(format!("types differ in {rule:?}: {a:?} and {b:?}")),
    }
}

/// Whether `path` names a value the harness declared unsupported (the declared name is the
/// path, a prefix of it, or a member under it).
fn excused(unsupported: &[String], path: &str) -> bool {
    unsupported.iter().any(|name| {
        path == name
            || path.starts_with(&format!("{name}."))
            || name.starts_with(&format!("{path}."))
    })
}

/// Judge every rule against `root`. A rule that cannot be read because one of its operands is
/// a declared-unsupported metric is `Unsupported`; one that cannot be read otherwise fails.
pub fn judge(root: &Json, unsupported: &[String], rules: &[String]) -> Vec<Check> {
    rules
        .iter()
        .map(|rule| {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let observed = Json::Arr(
                tokens
                    .iter()
                    .enumerate()
                    .filter(|(index, token)| index % 2 == 0 && !is_literal(token))
                    .map(|(_, token)| resolve(root, token).cloned().unwrap_or(Json::Null))
                    .collect(),
            );
            let status = match evaluate(root, rule) {
                Ok(true) => Status::Pass,
                Ok(false) => Status::Fail,
                Err(_)
                    if tokens
                        .iter()
                        .step_by(2)
                        .any(|token| !is_literal(token) && excused(unsupported, token)) =>
                {
                    Status::Unsupported
                }
                Err(_) => Status::Fail,
            };
            Check {
                rule: rule.clone(),
                observed,
                status,
            }
        })
        .collect()
}

/// The rules of a criteria list that are judged (not unsupported), in order: what
/// `criteria.toml` holds.
#[cfg_attr(not(test), allow(dead_code))]
pub fn judged_rules(checks: &[Check]) -> Vec<String> {
    checks
        .iter()
        .filter(|check| check.status != Status::Unsupported)
        .map(|check| check.rule.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIST: MetricList = MetricList {
        names: &["a", "g", "u", "gap"],
        grouped: &["g"],
    };

    #[test]
    fn json_renders_on_one_line_and_escapes() {
        let value = Json::obj([
            ("a", Json::Int(-3)),
            ("b", Json::Float(0.5)),
            ("c", Json::str("x\"y\n")),
            ("d", Json::Arr(vec![Json::Null, Json::Bool(true)])),
            ("e", Json::Float(f64::NAN)),
        ]);
        assert_eq!(
            value.render(),
            r#"{"a":-3,"b":0.5,"c":"x\"y\n","d":[null,true],"e":null}"#
        );
    }

    #[test]
    fn a_fraction_is_integers_and_equal_iff_the_cross_difference_is_zero() {
        let equal = fraction_json(128, 256, 1, 2);
        assert_eq!(resolve(&equal, "cross_difference"), Some(&Json::Int(0)));
        let unequal = fraction_json(129, 256, 1, 2);
        assert_eq!(resolve(&unequal, "cross_difference"), Some(&Json::Int(2)));
        assert!(!equal.render().contains('.'));
    }

    #[test]
    fn unfilled_unsupported_and_gap_metrics_are_told_apart() {
        let mut tally = Tally::new(LIST);
        tally.add("g.x", 2);
        tally.unsupported("u", "because");
        tally.gap("gap", "not merged");
        let (printed, gaps) = tally.metrics();
        assert_eq!(printed.len(), 4);
        assert_eq!(
            gaps,
            vec![
                (
                    "a".to_string(),
                    "no value and no declared reason".to_string()
                ),
                ("gap".to_string(), "not merged".to_string())
            ]
        );
        assert_eq!(printed[1].1.render(), r#"{"x":2}"#);
        assert!(tally.unsupported_json().render().contains("because"));
        assert!(tally.internal().render() == "{}");
    }

    #[test]
    fn the_grammar_reads_literals_and_references_and_refuses_the_rest() {
        let root = Json::obj([
            ("a", Json::Int(2)),
            ("o", Json::obj([("n", Json::Int(3))])),
            ("s", Json::str("x")),
            ("t", Json::str("x")),
            ("f", Json::Float(0.5)),
        ]);
        assert_eq!(evaluate(&root, "o.n >= a"), Ok(true));
        assert_eq!(evaluate(&root, "a == 3"), Ok(false));
        assert_eq!(evaluate(&root, "a >= -1"), Ok(true));
        assert_eq!(evaluate(&root, "s == t"), Ok(true));
        assert!(evaluate(&root, "f >= 1").is_err(), "a float is no operand");
        assert!(evaluate(&root, "s >= t").is_err(), "strings only under ==");
        assert!(evaluate(&root, "missing >= 1").is_err());
        assert!(evaluate(&root, "a > 1").is_err());
        assert!(evaluate(&root, "a >=  1").is_err());
        assert!(evaluate(&root, "a == s").is_err());
    }

    #[test]
    fn judging_marks_unsupported_operands_and_fails_unreadable_ones() {
        let root = Json::obj([("a", Json::Int(0)), ("u", Json::Null)]);
        let rules = vec![
            "a <= 0".to_string(),
            "a >= 1".to_string(),
            "u >= 1".to_string(),
            "zzz >= 1".to_string(),
        ];
        let checks = judge(&root, &["u".to_string()], &rules);
        let status: Vec<Status> = checks.iter().map(|check| check.status).collect();
        assert_eq!(
            status,
            [
                Status::Pass,
                Status::Fail,
                Status::Unsupported,
                Status::Fail
            ]
        );
        assert_eq!(
            judged_rules(&checks),
            [rules[0].clone(), rules[1].clone(), rules[3].clone()]
        );
    }
}
