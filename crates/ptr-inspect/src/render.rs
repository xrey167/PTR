//! Deterministic text, single-line and JSON renderings of an inspection tree.

use crate::node::{fingerprint_label, BytesPreview, InspectNode, RedactionHint};

/// Nesting at which [`render_json`] stops and writes `{"truncated":true}`.
pub const JSON_MAX_DEPTH: usize = 128;

/// Hard ceiling on [`RenderOptions::max_depth`]: the tree and single-line renderers recurse, so a
/// larger caller-supplied depth (for example `usize::MAX`) is clamped to this before traversal.
pub const MAX_RENDER_DEPTH: usize = 64;

/// Largest [`RenderOptions::max_text_width`] honoured; larger values are clamped.
pub const MAX_TEXT_WIDTH: usize = 65_536;

/// Largest [`RenderOptions::max_items`] honoured; larger values are clamped.
pub const MAX_RENDER_ITEMS: usize = 65_536;

/// Largest per-level indentation honoured by [`render_tree`]; larger `indent` values are clamped.
pub const MAX_INDENT: usize = 32;

/// Bounds and layout for [`render_tree`] and [`render_compact`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderOptions {
    /// Spaces per nesting level in the tree renderer, clamped to [`MAX_INDENT`].
    pub indent: usize,
    /// Longest text shown, in characters, clamped to [`MAX_TEXT_WIDTH`]; longer text ends in an
    /// ellipsis.
    pub max_text_width: usize,
    /// Containers nested deeper than this show `…` instead of their children; clamped to
    /// [`MAX_RENDER_DEPTH`].
    pub max_depth: usize,
    /// Children shown per container before `… N more`, clamped to [`MAX_RENDER_ITEMS`].
    pub max_items: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            indent: 2,
            max_text_width: 80,
            max_depth: 16,
            max_items: 64,
        }
    }
}

/// Render a multi-line indented tree. The output ends without a trailing newline.
pub fn render_tree(node: &InspectNode, options: &RenderOptions) -> String {
    let mut lines = Vec::new();
    tree_node(&mut lines, node, options, 0, "");
    lines.join("\n")
}

/// Render the whole tree on one line, for example `Foo{a:1,b:[2,3]}`.
pub fn render_compact(node: &InspectNode, options: &RenderOptions) -> String {
    let mut out = String::new();
    compact_node(&mut out, node, options, 0);
    out
}

/// Render JSON. Structs are `{"struct":name,"fields":{..}}`, enums
/// `{"enum":name,"variant":v,"fields":{..}}`, maps `{"map":[[key,value],..]}`, bytes
/// `{"bytes":hex_prefix,"len":n,"fingerprint":..}`, redactions `{"redacted":true,..}`.
/// Non-finite floats are the strings `"NaN"`, `"inf"` and `"-inf"`. Integers are written as
/// JSON numbers at full precision, which a consumer using binary64 may round.
pub fn render_json(node: &InspectNode) -> String {
    let mut out = String::new();
    json_node(&mut out, node, 0);
    out
}

/// Label for a map key: the compact rendering with generous but finite bounds.
pub(crate) fn key_label(key: &InspectNode) -> String {
    let options = RenderOptions {
        indent: 0,
        max_text_width: 4096,
        max_depth: 8,
        max_items: 1024,
    };
    render_compact(key, &options)
}

fn number_text(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_owned()
    } else if value.is_infinite() {
        if value > 0.0 { "inf" } else { "-inf" }.to_owned()
    } else {
        format!("{value:?}")
    }
}

fn escape_into(out: &mut String, ch: char) {
    match ch {
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"),
        '\t' => out.push_str("\\t"),
        c if c.is_control() => out.push_str(&format!("\\u{{{:x}}}", u32::from(c))),
        c => out.push(c),
    }
}

fn quote(text: &str, max_width: usize) -> String {
    let mut out = String::from("\"");
    let mut chars = text.chars();
    for ch in chars.by_ref().take(max_width) {
        escape_into(&mut out, ch);
    }
    if chars.next().is_some() {
        out.push('…');
    }
    out.push('"');
    out
}

fn label(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        escape_into(&mut out, ch);
    }
    out
}

fn bytes_text(bytes: &BytesPreview) -> String {
    let ellipsis = if bytes.is_truncated() { "…" } else { "" };
    format!(
        "bytes(len={}, {}, 0x{}{ellipsis})",
        bytes.len(),
        fingerprint_label(bytes.fingerprint()),
        bytes.head_hex()
    )
}

fn hint_text(hint: &RedactionHint) -> String {
    let mut out = format!("<redacted type={}", hint.type_name());
    if let Some(class) = hint.length_class() {
        out.push_str(&format!(" len={}", class.label()));
    }
    out.push('>');
    out
}

/// Text of a leaf, or `None` for a container.
fn scalar_text(node: &InspectNode, options: &RenderOptions) -> Option<String> {
    Some(match node {
        InspectNode::Null => "null".to_owned(),
        InspectNode::Bool(value) => value.to_string(),
        InspectNode::Number(value) => number_text(*value),
        InspectNode::Int(value) => value.to_string(),
        InspectNode::UInt(value) => value.to_string(),
        InspectNode::Text(value) => quote(value, options.max_text_width.min(MAX_TEXT_WIDTH)),
        InspectNode::Bytes(bytes) => bytes_text(bytes),
        InspectNode::Redacted => "<redacted>".to_owned(),
        InspectNode::RedactedWith(hint) => hint_text(hint),
        InspectNode::Struct { .. }
        | InspectNode::List(_)
        | InspectNode::Map(_)
        | InspectNode::Enum { .. } => return None,
    })
}

/// Indentation for `level`; `indent` is clamped to [`MAX_INDENT`] and the product saturates.
fn padding(options: &RenderOptions, level: usize) -> String {
    " ".repeat(options.indent.min(MAX_INDENT).saturating_mul(level))
}

fn tree_node(
    lines: &mut Vec<String>,
    node: &InspectNode,
    options: &RenderOptions,
    level: usize,
    prefix: &str,
) {
    let pad = padding(options, level);
    if let Some(text) = scalar_text(node, options) {
        lines.push(format!("{pad}{prefix}{text}"));
        return;
    }
    let (head, child_count) = match node {
        InspectNode::Struct { name, fields } => (label(name), fields.len()),
        InspectNode::Enum {
            name,
            variant,
            fields,
        } => (format!("{}::{}", label(name), label(variant)), fields.len()),
        InspectNode::List(items) => (format!("List({})", items.len()), items.len()),
        InspectNode::Map(entries) => (format!("Map({})", entries.len()), entries.len()),
        _ => return,
    };
    lines.push(format!("{pad}{prefix}{head}"));
    if child_count == 0 {
        return;
    }
    let child_pad = padding(options, level.saturating_add(1));
    if level >= options.max_depth.min(MAX_RENDER_DEPTH) {
        lines.push(format!("{child_pad}…"));
        return;
    }
    let shown = child_count.min(options.max_items.min(MAX_RENDER_ITEMS));
    match node {
        InspectNode::Struct { fields, .. } | InspectNode::Enum { fields, .. } => {
            for (name, value) in fields.iter().take(shown) {
                tree_node(
                    lines,
                    value,
                    options,
                    level + 1,
                    &format!("{}: ", label(name)),
                );
            }
        }
        InspectNode::List(items) => {
            for (index, value) in items.iter().take(shown).enumerate() {
                tree_node(lines, value, options, level + 1, &format!("[{index}] "));
            }
        }
        InspectNode::Map(entries) => {
            for (key, value) in entries.iter().take(shown) {
                let prefix = format!("{} => ", render_compact(key, options));
                tree_node(lines, value, options, level + 1, &prefix);
            }
        }
        _ => {}
    }
    if child_count > shown {
        lines.push(format!("{child_pad}… {} more", child_count - shown));
    }
}

fn compact_node(out: &mut String, node: &InspectNode, options: &RenderOptions, depth: usize) {
    if let Some(text) = scalar_text(node, options) {
        out.push_str(&text);
        return;
    }
    match node {
        InspectNode::Struct { name, fields } => {
            out.push_str(&label(name));
            compact_fields(out, fields, options, depth, true);
        }
        InspectNode::Enum {
            name,
            variant,
            fields,
        } => {
            out.push_str(&label(name));
            out.push_str("::");
            out.push_str(&label(variant));
            compact_fields(out, fields, options, depth, false);
        }
        InspectNode::List(items) => {
            out.push('[');
            compact_seq(out, items.len(), options, depth, |out, index| {
                compact_node(out, &items[index], options, depth + 1);
            });
            out.push(']');
        }
        InspectNode::Map(entries) => {
            out.push('{');
            compact_seq(out, entries.len(), options, depth, |out, index| {
                compact_node(out, &entries[index].0, options, depth + 1);
                out.push_str("=>");
                compact_node(out, &entries[index].1, options, depth + 1);
            });
            out.push('}');
        }
        _ => {}
    }
}

fn compact_fields(
    out: &mut String,
    fields: &[(String, InspectNode)],
    options: &RenderOptions,
    depth: usize,
    always_braces: bool,
) {
    if fields.is_empty() && !always_braces {
        return;
    }
    out.push('{');
    compact_seq(out, fields.len(), options, depth, |out, index| {
        out.push_str(&label(&fields[index].0));
        out.push(':');
        compact_node(out, &fields[index].1, options, depth + 1);
    });
    out.push('}');
}

/// Write up to `max_items` comma-separated entries, or `…` past the depth limit.
fn compact_seq(
    out: &mut String,
    len: usize,
    options: &RenderOptions,
    depth: usize,
    mut entry: impl FnMut(&mut String, usize),
) {
    if len == 0 {
        return;
    }
    if depth >= options.max_depth.min(MAX_RENDER_DEPTH) {
        out.push('…');
        return;
    }
    let shown = len.min(options.max_items.min(MAX_RENDER_ITEMS));
    for index in 0..shown {
        if index > 0 {
            out.push(',');
        }
        entry(out, index);
    }
    if len > shown {
        out.push_str(&format!(",…{} more", len - shown));
    }
}

fn json_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_fields(out: &mut String, fields: &[(String, InspectNode)], depth: usize) {
    out.push('{');
    for (index, (name, value)) in fields.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        json_string(out, name);
        out.push(':');
        json_node(out, value, depth + 1);
    }
    out.push('}');
}

fn json_node(out: &mut String, node: &InspectNode, depth: usize) {
    if depth > JSON_MAX_DEPTH {
        out.push_str("{\"truncated\":true}");
        return;
    }
    match node {
        InspectNode::Null => out.push_str("null"),
        InspectNode::Bool(value) => out.push_str(&value.to_string()),
        InspectNode::Number(value) if value.is_finite() => out.push_str(&format!("{value:?}")),
        InspectNode::Number(value) => json_string(out, &number_text(*value)),
        InspectNode::Int(value) => out.push_str(&value.to_string()),
        InspectNode::UInt(value) => out.push_str(&value.to_string()),
        InspectNode::Text(value) => json_string(out, value),
        InspectNode::Bytes(bytes) => {
            out.push_str("{\"bytes\":");
            json_string(out, &bytes.head_hex());
            out.push_str(&format!(",\"len\":{},\"fingerprint\":", bytes.len()));
            json_string(out, &fingerprint_label(bytes.fingerprint()));
            out.push('}');
        }
        InspectNode::Struct { name, fields } => {
            out.push_str("{\"struct\":");
            json_string(out, name);
            out.push_str(",\"fields\":");
            json_fields(out, fields, depth);
            out.push('}');
        }
        InspectNode::Enum {
            name,
            variant,
            fields,
        } => {
            out.push_str("{\"enum\":");
            json_string(out, name);
            out.push_str(",\"variant\":");
            json_string(out, variant);
            out.push_str(",\"fields\":");
            json_fields(out, fields, depth);
            out.push('}');
        }
        InspectNode::List(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                json_node(out, item, depth + 1);
            }
            out.push(']');
        }
        InspectNode::Map(entries) => {
            out.push_str("{\"map\":[");
            for (index, (key, value)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push('[');
                json_node(out, key, depth + 1);
                out.push(',');
                json_node(out, value, depth + 1);
                out.push(']');
            }
            out.push_str("]}");
        }
        InspectNode::Redacted => out.push_str("{\"redacted\":true}"),
        InspectNode::RedactedWith(hint) => {
            out.push_str("{\"redacted\":true,\"type\":");
            json_string(out, hint.type_name());
            if let Some(class) = hint.length_class() {
                out.push_str(",\"length_class\":");
                json_string(out, class.label());
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> InspectNode {
        InspectNode::new_struct(
            "Job",
            vec![
                ("id", InspectNode::UInt(7)),
                (
                    "tags",
                    InspectNode::List(vec![InspectNode::text("a"), InspectNode::Null]),
                ),
                (
                    "meta",
                    InspectNode::Map(vec![(InspectNode::text("k"), InspectNode::Bool(true))]),
                ),
                ("state", InspectNode::unit_variant("State", "Ready")),
                ("pw", InspectNode::Redacted),
            ],
        )
    }

    #[test]
    fn tree_is_indented_and_deterministic() {
        let expected = "Job\n  id: 7\n  tags: List(2)\n    [0] \"a\"\n    [1] null\n  meta: Map(1)\n    \"k\" => true\n  state: State::Ready\n  pw: <redacted>";
        let options = RenderOptions::default();
        assert_eq!(render_tree(&sample(), &options), expected);
        assert_eq!(render_tree(&sample(), &options), expected);
    }

    #[test]
    fn compact_is_single_line() {
        let text = render_compact(&sample(), &RenderOptions::default());
        assert_eq!(
            text,
            "Job{id:7,tags:[\"a\",null],meta:{\"k\"=>true},state:State::Ready,pw:<redacted>}"
        );
        assert!(!text.contains('\n'));
    }

    #[test]
    fn long_text_gets_an_ellipsis_and_newlines_are_escaped() {
        let options = RenderOptions {
            max_text_width: 4,
            ..RenderOptions::default()
        };
        assert_eq!(
            render_compact(&InspectNode::text("abcdefgh"), &options),
            "\"abcd…\""
        );
        assert_eq!(
            render_compact(&InspectNode::text("a\nb"), &options),
            "\"a\\nb\""
        );
    }

    #[test]
    fn depth_and_item_limits_truncate_visibly() {
        let nested = InspectNode::List(vec![InspectNode::List(vec![InspectNode::Int(1)])]);
        let options = RenderOptions {
            max_depth: 1,
            ..RenderOptions::default()
        };
        assert_eq!(render_compact(&nested, &options), "[[…]]");
        assert_eq!(
            render_tree(&nested, &options),
            "List(1)\n  [0] List(1)\n    …"
        );
        let many = InspectNode::List((0..5).map(InspectNode::Int).collect());
        let options = RenderOptions {
            max_items: 2,
            ..RenderOptions::default()
        };
        assert_eq!(render_compact(&many, &options), "[0,1,…3 more]");
        assert!(render_tree(&many, &options).ends_with("… 3 more"));
    }

    #[test]
    fn huge_width_and_item_limits_are_clamped() {
        let long = InspectNode::text("x".repeat(MAX_TEXT_WIDTH + 10));
        let options = RenderOptions {
            max_text_width: usize::MAX,
            max_items: usize::MAX,
            ..RenderOptions::default()
        };
        let out = render_compact(&long, &options);
        assert_eq!(out.chars().count(), MAX_TEXT_WIDTH + 3);
        let list = InspectNode::List(vec![InspectNode::Null; MAX_RENDER_ITEMS + 5]);
        assert_eq!(
            render_compact(&list, &options).matches("null").count(),
            MAX_RENDER_ITEMS
        );
    }

    #[test]
    fn huge_indent_is_clamped_instead_of_allocating() {
        let nested = InspectNode::List(vec![InspectNode::List(vec![InspectNode::Int(1)])]);
        for indent in [usize::MAX, usize::MAX / 2, 1 << 40] {
            let options = RenderOptions {
                indent,
                ..RenderOptions::default()
            };
            let text = render_tree(&nested, &options);
            assert!(
                text.len() <= 3 * (2 * MAX_INDENT + 16),
                "{} bytes",
                text.len()
            );
            assert!(text.contains(&format!("\n{}[0] List(1)", " ".repeat(MAX_INDENT))));
        }
    }

    #[test]
    fn json_escapes_controls_quotes_and_keeps_unicode() {
        let node = InspectNode::text("q\"\\\n\r\t\u{8}\u{c}\u{1}\u{1f}é😀");
        assert_eq!(
            render_json(&node),
            "\"q\\\"\\\\\\n\\r\\t\\b\\f\\u0001\\u001fé😀\""
        );
    }

    #[test]
    fn json_writes_non_finite_floats_as_strings() {
        let list = InspectNode::List(vec![
            InspectNode::Number(f64::NAN),
            InspectNode::Number(f64::INFINITY),
            InspectNode::Number(f64::NEG_INFINITY),
            InspectNode::Number(1.5),
            InspectNode::Number(2.0),
        ]);
        assert_eq!(render_json(&list), "[\"NaN\",\"inf\",\"-inf\",1.5,2.0]");
    }

    #[test]
    fn json_keeps_integers_lossless() {
        let list = InspectNode::List(vec![
            InspectNode::Int(i128::MIN),
            InspectNode::UInt(u128::MAX),
        ]);
        assert_eq!(
            render_json(&list),
            "[-170141183460469231731687303715884105728,340282366920938463463374607431768211455]"
        );
    }

    #[test]
    fn json_shapes_for_containers() {
        assert_eq!(
            render_json(&sample()),
            "{\"struct\":\"Job\",\"fields\":{\"id\":7,\"tags\":[\"a\",null],\"meta\":{\"map\":[[\"k\",true]]},\"state\":{\"enum\":\"State\",\"variant\":\"Ready\",\"fields\":{}},\"pw\":{\"redacted\":true}}}"
        );
    }
}
