//! `ptrctl trace`: summarize a span file written by `ptr_observe::OtlpJsonFileExporter`.
//!
//! `ptr-observe` has no reader, so this is a minimal bounded reader for exactly the shape the
//! exporter writes: one JSON document per line with
//! `resourceSpans[].scopeSpans[].spans[]` carrying `traceId`, `spanId`, optional
//! `parentSpanId`, `name`, `startTimeUnixNano`, `endTimeUnixNano`, `events` and `status`.
//! It is not a general OTLP reader. Times are the exporter's ticks plus its epoch offset, so
//! durations are meaningful and absolute times may not be wall-clock.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::cli::{is_help, usage_error, EXIT_CODES_HELP, EXIT_INPUT, EXIT_OK};
use crate::json::{self, Json};

const USAGE: &str = "usage: ptrctl trace <otlp.jsonl>";
/// Largest file read.
const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
/// Most spans summarized.
const MAX_SPANS: usize = 100_000;
/// Deepest indentation drawn; spans below it carry a `[depth N]` marker instead.
const MAX_INDENT_DEPTH: usize = 64;

fn help() -> String {
    format!(
        "{USAGE}

Summarize an OTLP/JSON span file written by ptr-observe's OtlpJsonFileExporter: one span
tree per trace with durations (ns), status and event counts. Reads at most
{MAX_FILE_BYTES} bytes and {MAX_SPANS} spans.

{EXIT_CODES_HELP}"
    )
}

struct SpanRow {
    id: String,
    parent: Option<String>,
    name: String,
    start: u64,
    end: u64,
    events: usize,
    status: String,
    error: bool,
}

pub(crate) fn command(args: Vec<String>) -> i32 {
    if args.first().is_some_and(|arg| is_help(arg)) {
        println!("{}", help());
        return EXIT_OK;
    }
    let [path] = args.as_slice() else {
        return usage_error("trace", "exactly one file is required", USAGE);
    };
    match summarize(Path::new(path)) {
        Ok(text) => {
            print!("{text}");
            EXIT_OK
        }
        Err(message) => {
            eprintln!("ptrctl trace: {message}");
            EXIT_INPUT
        }
    }
}

fn summarize(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(format!(
            "{} is longer than {MAX_FILE_BYTES} bytes",
            path.display()
        ));
    }
    let text = String::from_utf8(bytes).map_err(|_| "file is not valid UTF-8".to_owned())?;

    let mut traces: BTreeMap<usize, (String, Vec<SpanRow>)> = BTreeMap::new();
    let mut order: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = |message: String| format!("line {}: {message}", index + 1);
        let document = json::parse(line).map_err(|e| at(e.to_string()))?;
        for (trace_id, row) in spans_of(&document).map_err(at)? {
            total += 1;
            if total > MAX_SPANS {
                return Err(format!("more than {MAX_SPANS} spans"));
            }
            let next = order.len();
            let slot = *order.entry(trace_id.clone()).or_insert(next);
            traces
                .entry(slot)
                .or_insert_with(|| (trace_id, Vec::new()))
                .1
                .push(row);
        }
    }
    if total == 0 {
        return Err("no spans found (expected OTLP/JSON lines with resourceSpans)".to_owned());
    }
    let errors = traces
        .values()
        .flat_map(|(_, rows)| rows)
        .filter(|row| row.error)
        .count();
    let mut out = format!(
        "{}: {} trace(s), {total} span(s), {errors} error span(s)\n",
        path.display(),
        traces.len()
    );
    for (trace_id, rows) in traces.values() {
        render_trace(&mut out, trace_id, rows);
    }
    Ok(out)
}

fn text_field<'a>(object: &'a Json, key: &str) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Json::as_str)
        .ok_or_else(|| format!("span field {key:?} is missing or not a string"))
}

fn nano_field(object: &Json, key: &str) -> Result<u64, String> {
    text_field(object, key)?
        .parse()
        .map_err(|_| format!("span field {key:?} is not an unsigned integer string"))
}

fn array<'a>(value: &'a Json, key: &str) -> Result<&'a [Json], String> {
    value
        .get(key)
        .and_then(Json::as_array)
        .ok_or_else(|| format!("{key:?} is missing or not an array"))
}

/// Every span of one exporter line, with its trace id.
fn spans_of(document: &Json) -> Result<Vec<(String, SpanRow)>, String> {
    let mut out = Vec::new();
    for resource in array(document, "resourceSpans")? {
        for scope in array(resource, "scopeSpans")? {
            for span in array(scope, "spans")? {
                out.push((text_field(span, "traceId")?.to_owned(), span_row(span)?));
            }
        }
    }
    Ok(out)
}

fn span_row(span: &Json) -> Result<SpanRow, String> {
    let status = span.get("status");
    let code = status
        .and_then(|s| s.get("code"))
        .and_then(Json::as_u64)
        .ok_or("span status has no numeric code")?;
    let message = status.and_then(|s| s.get("message")).and_then(Json::as_str);
    let (label, error) = match code {
        0 => ("UNSET".to_owned(), false),
        1 => ("OK".to_owned(), false),
        2 => (
            match message {
                Some(message) => format!("ERROR {message:?}"),
                None => "ERROR".to_owned(),
            },
            true,
        ),
        other => (format!("UNKNOWN({other})"), false),
    };
    Ok(SpanRow {
        id: text_field(span, "spanId")?.to_owned(),
        parent: span
            .get("parentSpanId")
            .and_then(Json::as_str)
            .map(str::to_owned),
        name: text_field(span, "name")?.to_owned(),
        start: nano_field(span, "startTimeUnixNano")?,
        end: nano_field(span, "endTimeUnixNano")?,
        events: array(span, "events")?.len(),
        status: label,
        error,
    })
}

fn render_trace(out: &mut String, trace_id: &str, rows: &[SpanRow]) {
    let first = rows.iter().map(|r| r.start).min().unwrap_or(0);
    let last = rows.iter().map(|r| r.end).max().unwrap_or(0);
    out.push_str(&format!(
        "trace {trace_id}: {} span(s), {} ns\n",
        rows.len(),
        last.saturating_sub(first)
    ));
    let by_id: HashMap<&str, usize> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| (row.id.as_str(), index))
        .collect();
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut roots = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        match row.parent.as_deref().and_then(|p| by_id.get(p)) {
            Some(parent) if *parent != index => children.entry(*parent).or_default().push(index),
            _ => roots.push(index),
        }
    }
    let by_start = |list: &mut Vec<usize>| list.sort_by_key(|&i| (rows[i].start, i));
    by_start(&mut roots);
    children.values_mut().for_each(by_start);

    let mut seen = vec![false; rows.len()];
    // Depth-first with an explicit stack so a deep chain cannot overflow the call stack.
    let mut stack: Vec<(usize, usize)> = roots.iter().rev().map(|&i| (i, 1)).collect();
    while let Some((index, depth)) = stack.pop() {
        seen[index] = true;
        let row = &rows[index];
        let orphan = match row.parent.as_deref() {
            Some(parent) if !by_id.contains_key(parent) => {
                format!(" (parent {parent} not in file)")
            }
            _ => String::new(),
        };
        // Indentation is capped so a deep chain costs O(1) per line, not O(depth).
        let marker = if depth > MAX_INDENT_DEPTH {
            format!("[depth {depth}] ")
        } else {
            String::new()
        };
        out.push_str(&format!(
            "{}{marker}{} [{} ns] {} events={} span={}{orphan}\n",
            "  ".repeat(depth.min(MAX_INDENT_DEPTH)),
            row.name.escape_debug(),
            row.end.saturating_sub(row.start),
            row.status,
            row.events,
            row.id
        ));
        if let Some(kids) = children.get(&index) {
            stack.extend(kids.iter().rev().map(|&kid| (kid, depth + 1)));
        }
    }
    for (_, row) in rows.iter().enumerate().filter(|(i, _)| !seen[*i]) {
        out.push_str(&format!(
            "  unreachable (parent cycle): {} span={}\n",
            row.name.escape_debug(),
            row.id
        ));
    }
}
