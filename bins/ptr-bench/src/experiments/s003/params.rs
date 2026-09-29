//! The preregistered parameters of S003, compiled in, and the canonical text
//! of the table they come from.
//!
//! The table is `[preregistration]` in the experiment's `config.toml`. Its
//! canonical text is what `json.dumps(table, sort_keys=True,
//! separators=(",", ":"))` prints in Python, which every result echoes so that
//! the aggregator can compare it with the frozen table byte for byte. Every
//! value is an integer, a list of integers, a printable ASCII string or a list
//! of them, so this module reproduces that text without a JSON library.
//!
//! `low_cells` is the one key the pilot fills, and it is not compiled in: the
//! harness echoes the table it finds in the file, and only the aggregator
//! reads the list. The drift test fails when a compiled constant differs from
//! any other key of the file.

use crate::experiments::json::json_string;

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/semdb/S003-certified-branches/config.toml");

pub const SCHEMA: u64 = 1;
pub const HARNESS: &str = "certified-branches";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];
pub const PILOT_SEEDS: [u64; 3] = [1, 2, 3];
pub const CASES_PER_SEED: usize = 48;
pub const TASKS_PER_CASE: usize = 128;
pub const AGENTS: [usize; 4] = [2, 4, 8, 16];
pub const GROUPS_LADDER: [usize; 6] = [8, 16, 32, 64, 128, 256];
pub const GROUPS_LADDER_FALLBACK: [usize; 6] = [4, 8, 16, 64, 256, 1024];
pub const ITEMS_PER_GROUP: usize = 8;
pub const INSERT_CAP: usize = 12;
pub const COUNTERS: usize = 16;
pub const COUNTER_START: i64 = 50;
pub const SETS: usize = 8;
pub const SET_MEMBERS: usize = 16;
pub const POLICIES: usize = 4;
pub const TICK_US: u64 = 10_000;
pub const THINK_MIN_TICKS: u64 = 20;
pub const THINK_MAX_TICKS: u64 = 60;
pub const STEP_TICKS: u64 = 1;
pub const MERGE_TICKS: u64 = 1;
pub const MAX_ATTEMPTS: usize = 8;
pub const REVIEW_MIN_TICKS: u64 = 20;
pub const REVIEW_MAX_TICKS: u64 = 60;
pub const PROGRAMS: [&str; 9] = [
    "rmw",
    "write_skew",
    "insert_capped",
    "remove_extra",
    "audit",
    "group_total",
    "counter_add",
    "set_op",
    "guarded_decrement",
];
pub const PROGRAM_WEIGHTS_PERMILLE: [u64; 9] = [250, 50, 100, 50, 100, 100, 200, 100, 50];
pub const RMW_DELTA_MAX: i64 = 5;
pub const COUNTER_ADD_MAX: i64 = 10;
pub const GUARD_AMOUNT: i64 = 10;
pub const RELY_PERMILLE: u64 = 200;
pub const INGRESS_PERMILLE_PER_TICK: u64 = 20;
pub const EXTERNAL_WRITE_PERMILLE_PER_TICK: u64 = 10;
pub const LIFECYCLE_PERMILLE_PER_TICK: u64 = 2;
pub const REWIRE_PERMILLE_PER_TICK: u64 = 10;
pub const REWIRE_SWAP_MIN_TICKS: u64 = 5;
pub const REWIRE_SWAP_MAX_TICKS: u64 = 30;
pub const REQUIRED_VERIFICATION: &str = "deterministic";
pub const VERIFIERS: [&str; 2] = ["s003-domain/v1", "s003-diff/v1"];
pub const HOST_COUNTER_JUMP_MAX: i64 = 100;
pub const AUTO_POLICY: &str = "s003-auto/v1";
pub const AUTO_THRESHOLD_PERMILLE: u64 = 0;
pub const AUTO_CALIBRATION_PERMILLE: u64 = 0;
pub const REVIEW_POLICY: &str = "s003-review/v1";
pub const REVIEW_THRESHOLD_PERMILLE: u64 = 500;
pub const REVIEW_CALIBRATION_PERMILLE: u64 = 100;
pub const CALIBRATION_SEED: u64 = 7919;
pub const REVIEWER: &str = "s003-reviewer";
pub const CONFLICT_THRESHOLD_PERMILLE: u64 = 100;
pub const EFFICIENCY_FLOOR_PERMILLE: u64 = 500;
pub const BOOTSTRAP_RESAMPLES: usize = 2000;
pub const BOOTSTRAP_SEED: u64 = 20_260_926;
pub const BOOTSTRAP_INTERVAL_PERMILLE: u64 = 950;
pub const MIN_HAZARD_TRIALS_PER_CLASS_PER_SEED: u64 = 30;
pub const MIN_CELLS_PER_SIDE: usize = 6;
pub const MERGE_WALL_BUDGET_US_P99: u64 = 10_000;
pub const DURABLE_ROUNDTRIP_EVERY_CASES: usize = 8;
pub const NONDETERMINISM_RERUN_CASE: usize = 0;

/// A value of the table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    Int(i64),
    Ints(Vec<i64>),
    Str(String),
    Strs(Vec<String>),
}

fn sizes(values: &[usize]) -> Value {
    Value::Ints(values.iter().map(|value| *value as i64).collect())
}

fn strs(values: &[&str]) -> Value {
    Value::Strs(values.iter().map(|value| (*value).to_string()).collect())
}

fn text(value: &str) -> Value {
    Value::Str(value.to_string())
}

/// Every key of the table but `low_cells`, as the harness compiles it.
pub fn compiled() -> Vec<(&'static str, Value)> {
    let int = |value: i64| Value::Int(value);
    vec![
        ("schema", int(SCHEMA as i64)),
        ("harness", text(HARNESS)),
        (
            "seeds",
            Value::Ints(SEEDS.iter().map(|seed| *seed as i64).collect()),
        ),
        (
            "pilot_seeds",
            Value::Ints(PILOT_SEEDS.iter().map(|seed| *seed as i64).collect()),
        ),
        ("cases_per_seed", int(CASES_PER_SEED as i64)),
        ("tasks_per_case", int(TASKS_PER_CASE as i64)),
        ("agents", sizes(&AGENTS)),
        ("groups_ladder", sizes(&GROUPS_LADDER)),
        ("groups_ladder_fallback", sizes(&GROUPS_LADDER_FALLBACK)),
        ("items_per_group", int(ITEMS_PER_GROUP as i64)),
        ("insert_cap", int(INSERT_CAP as i64)),
        ("counters", int(COUNTERS as i64)),
        ("counter_start", int(COUNTER_START)),
        ("sets", int(SETS as i64)),
        ("set_members", int(SET_MEMBERS as i64)),
        ("policies", int(POLICIES as i64)),
        ("tick_us", int(TICK_US as i64)),
        ("think_min_ticks", int(THINK_MIN_TICKS as i64)),
        ("think_max_ticks", int(THINK_MAX_TICKS as i64)),
        ("step_ticks", int(STEP_TICKS as i64)),
        ("merge_ticks", int(MERGE_TICKS as i64)),
        ("max_attempts", int(MAX_ATTEMPTS as i64)),
        ("review_min_ticks", int(REVIEW_MIN_TICKS as i64)),
        ("review_max_ticks", int(REVIEW_MAX_TICKS as i64)),
        ("programs", strs(&PROGRAMS)),
        (
            "program_weights_permille",
            Value::Ints(
                PROGRAM_WEIGHTS_PERMILLE
                    .iter()
                    .map(|weight| *weight as i64)
                    .collect(),
            ),
        ),
        ("rmw_delta_max", int(RMW_DELTA_MAX)),
        ("counter_add_max", int(COUNTER_ADD_MAX)),
        ("guard_amount", int(GUARD_AMOUNT)),
        ("rely_permille", int(RELY_PERMILLE as i64)),
        (
            "ingress_permille_per_tick",
            int(INGRESS_PERMILLE_PER_TICK as i64),
        ),
        (
            "external_write_permille_per_tick",
            int(EXTERNAL_WRITE_PERMILLE_PER_TICK as i64),
        ),
        (
            "lifecycle_permille_per_tick",
            int(LIFECYCLE_PERMILLE_PER_TICK as i64),
        ),
        (
            "rewire_permille_per_tick",
            int(REWIRE_PERMILLE_PER_TICK as i64),
        ),
        ("rewire_swap_min_ticks", int(REWIRE_SWAP_MIN_TICKS as i64)),
        ("rewire_swap_max_ticks", int(REWIRE_SWAP_MAX_TICKS as i64)),
        ("required_verification", text(REQUIRED_VERIFICATION)),
        ("verifiers", strs(&VERIFIERS)),
        ("host_counter_jump_max", int(HOST_COUNTER_JUMP_MAX)),
        ("auto_policy", text(AUTO_POLICY)),
        (
            "auto_threshold_permille",
            int(AUTO_THRESHOLD_PERMILLE as i64),
        ),
        (
            "auto_calibration_permille",
            int(AUTO_CALIBRATION_PERMILLE as i64),
        ),
        ("review_policy", text(REVIEW_POLICY)),
        (
            "review_threshold_permille",
            int(REVIEW_THRESHOLD_PERMILLE as i64),
        ),
        (
            "review_calibration_permille",
            int(REVIEW_CALIBRATION_PERMILLE as i64),
        ),
        ("calibration_seed", int(CALIBRATION_SEED as i64)),
        ("reviewer", text(REVIEWER)),
        (
            "conflict_threshold_permille",
            int(CONFLICT_THRESHOLD_PERMILLE as i64),
        ),
        (
            "efficiency_floor_permille",
            int(EFFICIENCY_FLOOR_PERMILLE as i64),
        ),
        ("bootstrap_resamples", int(BOOTSTRAP_RESAMPLES as i64)),
        ("bootstrap_seed", int(BOOTSTRAP_SEED as i64)),
        (
            "bootstrap_interval_permille",
            int(BOOTSTRAP_INTERVAL_PERMILLE as i64),
        ),
        (
            "min_hazard_trials_per_class_per_seed",
            int(MIN_HAZARD_TRIALS_PER_CLASS_PER_SEED as i64),
        ),
        ("min_cells_per_side", int(MIN_CELLS_PER_SIDE as i64)),
        (
            "merge_wall_budget_us_p99",
            int(MERGE_WALL_BUDGET_US_P99 as i64),
        ),
        (
            "durable_roundtrip_every_cases",
            int(DURABLE_ROUNDTRIP_EVERY_CASES as i64),
        ),
        (
            "nondeterminism_rerun_case",
            int(NONDETERMINISM_RERUN_CASE as i64),
        ),
    ]
}

/// Read the `[preregistration]` table of a configuration file: one
/// `key = value` per line, where a value is an integer, a printable ASCII
/// string without a quote or backslash, or a list of either, and a `#` outside
/// a string starts a comment. The order of the file is kept.
pub fn parse_table(source: &str) -> Result<Vec<(String, Value)>, String> {
    let mut in_table = false;
    let mut found = false;
    let mut table: Vec<(String, Value)> = Vec::new();
    for (number, raw) in source.lines().enumerate() {
        let line = strip_comment(raw).trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && !line.contains('=') {
            in_table = line == "[preregistration]";
            found |= in_table;
            continue;
        }
        if !in_table {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("line {}: a table line is `key = value`", number + 1))?;
        let key = key.trim();
        if key.is_empty()
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        {
            return Err(format!(
                "line {}: `{key}` is not a key of lowercase letters, digits and `_`",
                number + 1
            ));
        }
        if table.iter().any(|(existing, _)| existing == key) {
            return Err(format!("line {}: `{key}` appears twice", number + 1));
        }
        let value = parse_value(value.trim())
            .map_err(|error| format!("line {}: `{key}`: {error}", number + 1))?;
        table.push((key.to_string(), value));
    }
    if !found {
        return Err("no [preregistration] table".to_string());
    }
    Ok(table)
}

/// The line without a trailing comment, where a `#` inside a string is text.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (index, character) in line.char_indices() {
        match character {
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..index],
            _ => {}
        }
    }
    line
}

fn parse_value(value: &str) -> Result<Value, String> {
    if let Some(inner) = value.strip_prefix('[') {
        let inner = inner
            .strip_suffix(']')
            .ok_or("a list is closed by `]`")?
            .trim();
        if inner.is_empty() {
            return Ok(Value::Ints(Vec::new()));
        }
        let elements = split_elements(inner)?;
        if elements.iter().all(|element| element.starts_with('"')) {
            return elements
                .iter()
                .map(|element| parse_string(element))
                .collect::<Result<_, _>>()
                .map(Value::Strs);
        }
        if elements.iter().all(|element| !element.starts_with('"')) {
            return elements
                .iter()
                .map(|element| parse_int(element))
                .collect::<Result<_, _>>()
                .map(Value::Ints);
        }
        return Err("a list holds integers or strings, not both".to_string());
    }
    if value.starts_with('"') {
        return parse_string(value).map(Value::Str);
    }
    parse_int(value).map(Value::Int)
}

fn split_elements(inner: &str) -> Result<Vec<String>, String> {
    let mut elements = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    for character in inner.chars() {
        match character {
            '"' => {
                in_string = !in_string;
                current.push(character);
            }
            ',' if !in_string => {
                elements.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(character),
        }
    }
    if in_string {
        return Err("a string is not closed".to_string());
    }
    elements.push(current.trim().to_string());
    if elements.iter().any(String::is_empty) {
        return Err("a list has an empty element".to_string());
    }
    Ok(elements)
}

fn parse_string(element: &str) -> Result<String, String> {
    let inner = element
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .ok_or("a string is `\"...\"`")?;
    if inner.contains('"')
        || inner.contains('\\')
        || !inner.bytes().all(|byte| (0x20..0x7f).contains(&byte))
    {
        return Err(format!(
            "`{inner}` is not printable ASCII without a quote or backslash"
        ));
    }
    Ok(inner.to_string())
}

fn parse_int(element: &str) -> Result<i64, String> {
    let digits = element.strip_prefix('-').unwrap_or(element);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("`{element}` is not an integer"));
    }
    element
        .parse::<i64>()
        .map_err(|error| format!("`{element}`: {error}"))
}

/// The canonical text of a table: `json.dumps(table, sort_keys=True,
/// separators=(",", ":"))`, with keys in the order of their bytes.
pub fn canonical_text(table: &[(String, Value)]) -> String {
    let mut sorted: Vec<&(String, Value)> = table.iter().collect();
    sorted.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let mut out = String::from("{");
    for (index, (key, value)) in sorted.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&json_string(key));
        out.push(':');
        out.push_str(&value_text(value));
    }
    out.push('}');
    out
}

fn value_text(value: &Value) -> String {
    match value {
        Value::Int(number) => number.to_string(),
        Value::Ints(numbers) => {
            let items: Vec<String> = numbers.iter().map(i64::to_string).collect();
            format!("[{}]", items.join(","))
        }
        Value::Str(string) => json_string(string),
        Value::Strs(strings) => {
            let items: Vec<String> = strings.iter().map(|string| json_string(string)).collect();
            format!("[{}]", items.join(","))
        }
    }
}

/// The table of the configuration file this binary was built with.
pub fn frozen_table() -> Result<Vec<(String, Value)>, String> {
    parse_table(CONFIG_TOML)
}

/// The canonical text every result echoes: that of the table in the file,
/// `low_cells` included.
pub fn echo() -> Result<String, String> {
    frozen_table().map(|table| canonical_text(&table))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_ledger::integrity::sha256;

    fn table(pairs: &[(&str, Value)]) -> Vec<(String, Value)> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    #[test]
    fn the_compiled_parameters_are_the_frozen_preregistration() {
        let frozen = frozen_table().expect("the preregistration table parses");
        let compiled = compiled();
        for (key, value) in &frozen {
            if key == "low_cells" {
                continue;
            }
            let ours = compiled.iter().find(|(name, _)| name == key);
            assert_eq!(
                ours.map(|(_, value)| value),
                Some(value),
                "{key} differs from the frozen table"
            );
        }
        for (name, _) in &compiled {
            assert!(
                frozen.iter().any(|(key, _)| key == name),
                "{name} is not in the frozen table"
            );
        }
        assert_eq!(
            compiled.len() + 1,
            frozen.len(),
            "the frozen table has keys nothing compiles"
        );
        let low = frozen
            .iter()
            .find(|(key, _)| key == "low_cells")
            .expect("low_cells is in the table");
        match &low.1 {
            Value::Strs(cells) => assert!(cells.iter().all(|cell| {
                let mut parts = cell.strip_prefix('L').map(|rest| rest.splitn(2, 'N'));
                parts
                    .as_mut()
                    .map(|parts| {
                        parts.all(|part| {
                            !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())
                        })
                    })
                    .unwrap_or(false)
            })),
            Value::Ints(cells) => assert!(
                cells.is_empty(),
                "low_cells holds L<level>N<agents> strings"
            ),
            other => panic!("low_cells is a list of strings, not {other:?}"),
        }
    }

    #[test]
    fn the_echoed_canonical_text_is_the_python_canonical_text() {
        // Each expected text below is what Python's `json.dumps(table,
        // sort_keys=True, separators=(",", ":"))` prints for the same table.
        let fixture = table(&[
            ("b", Value::Int(1)),
            ("a", Value::Ints(vec![3, 1, 2])),
            ("c", Value::Str("x y".to_string())),
            ("d", Value::Strs(vec!["q".to_string(), "r/s".to_string()])),
            ("e", Value::Ints(Vec::new())),
            ("f", Value::Int(-7)),
            ("g", Value::Str("a\"b\\c".to_string())),
            ("B", Value::Int(0)),
            ("a_1", Value::Strs(Vec::new())),
        ]);
        assert_eq!(
            canonical_text(&fixture),
            r#"{"B":0,"a":[3,1,2],"a_1":[],"b":1,"c":"x y","d":["q","r/s"],"e":[],"f":-7,"g":"a\"b\\c"}"#
        );
        assert_eq!(canonical_text(&[]), "{}");
        let echoed = echo().expect("the preregistration table parses");
        assert!(
            echoed.starts_with(r#"{"agents":[2,4,8,16],"auto_calibration_permille":0,"#),
            "{echoed}"
        );
        assert!(
            echoed.ends_with(r#""verifiers":["s003-domain/v1","s003-diff/v1"]}"#),
            "{echoed}"
        );
        assert!(!echoed.contains(' ') && !echoed.contains('\n'));
        assert!(echoed.contains(r#""low_cells":["#));
    }

    #[test]
    fn the_canonical_text_of_the_frozen_table_matches_the_python_digest() {
        // `scripts/tests/test_s003_preregistration.py` holds the same digest
        // of what Python's `preregistration_canonical` makes of the table
        // without `low_cells` (which the pilot fills), so the two sides agree
        // for as long as both tests pass.
        let mut frozen = frozen_table().expect("the preregistration table parses");
        frozen.retain(|(key, _)| key != "low_cells");
        let digest: String = sha256(canonical_text(&frozen).as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(
            digest,
            "2511af8d2f9708fef79a0f37d811820e7cfcbce3bf9ef849204f2d8e345a267a"
        );
    }

    #[test]
    fn a_table_line_is_read_as_the_python_parser_reads_it() {
        let source = "\
version = 1\n\
[other]\n\
a = 1\n\
\n\
# a comment\n\
[preregistration]\n\
n = -12   # trailing comment\n\
s = \"with # inside\"\n\
xs = [1, 2 ,3]\n\
ws = [\"a\", \"b,c\"]\n\
none = []\n\
[after]\n\
z = 9\n";
        let parsed = parse_table(source).expect("parses");
        assert_eq!(
            parsed,
            table(&[
                ("n", Value::Int(-12)),
                ("s", Value::Str("with # inside".to_string())),
                ("xs", Value::Ints(vec![1, 2, 3])),
                ("ws", Value::Strs(vec!["a".to_string(), "b,c".to_string()])),
                ("none", Value::Ints(Vec::new())),
            ])
        );
    }

    #[test]
    fn a_table_that_cannot_be_read_exactly_is_refused() {
        for (source, why) in [
            ("a = 1\n", "no [preregistration] table"),
            ("[preregistration]\na = 1\na = 2\n", "appears twice"),
            ("[preregistration]\nA = 1\n", "is not a key"),
            ("[preregistration]\na = 1.5\n", "is not an integer"),
            ("[preregistration]\na = 1_000\n", "is not an integer"),
            ("[preregistration]\na = +1\n", "is not an integer"),
            (
                "[preregistration]\na = [1, \"x\"]\n",
                "integers or strings, not both",
            ),
            ("[preregistration]\na = [1,,2]\n", "empty element"),
            ("[preregistration]\na = [1, 2\n", "closed by"),
            ("[preregistration]\na = \"x\\ny\"\n", "printable ASCII"),
            ("[preregistration]\na = \"caf\u{e9}\"\n", "printable ASCII"),
            ("[preregistration]\na = \"open\n", "a string is"),
            ("[preregistration]\na\n", "key = value"),
        ] {
            let error = parse_table(source).expect_err(source);
            assert!(error.contains(why), "{source:?}: {error}");
        }
    }
}
