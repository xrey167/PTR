//! Deterministic, human-readable Markdown mirrors of memory contents.
//!
//! Output depends only on store content (sorted, no timestamps from the wall clock). Secrets
//! never reach the mirror: a record flagged `sensitive` has its body replaced by `[REDACTED]`,
//! and raw events are represented by byte length and digest only, never by content, tool
//! arguments or tool output (Invariant 13). Free text is escaped so a value cannot start a new
//! line or heading of its own.

use crate::knowledge::KnowledgeStore;
use crate::memory_store::DurableMemoryStore;
use crate::record::{class_name, class_tag, MemoryRecord, SupportKind};
use crate::MemoryClass;
use std::fmt::Write;

/// Text that replaces a sensitive body.
pub const REDACTED: &str = "[REDACTED]";

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => {
                let _ = write!(out, "\\u{{{:x}}}", u32::from(ch));
            }
            ch => out.push(ch),
        }
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn list<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let joined: Vec<String> = items.map(escape).collect();
    if joined.is_empty() {
        "-".to_owned()
    } else {
        joined.join(", ")
    }
}

fn write_record(out: &mut String, record: &MemoryRecord) {
    let _ = writeln!(out, "### {}", escape(&record.id.0));
    let _ = writeln!(out, "- state: {:?}", record.state);
    let _ = writeln!(out, "- pinned: {}", record.pinned);
    let body = if record.sensitive {
        REDACTED.to_owned()
    } else {
        escape(&record.body)
    };
    let _ = writeln!(out, "- body: {body}");
    let mut supports: Vec<String> = record
        .supports
        .iter()
        .map(|support| match support.kind {
            SupportKind::Observation { revision } => {
                format!("{} (observation r{})", escape(&support.source), revision.0)
            }
            SupportKind::Derived => format!("{} (derived)", escape(&support.source)),
            SupportKind::SearchHit => format!("{} (search)", escape(&support.source)),
        })
        .collect();
    supports.sort();
    let supports = if supports.is_empty() {
        "-".to_owned()
    } else {
        supports.join(", ")
    };
    let _ = writeln!(out, "- supports: {supports}");
    let mut derived: Vec<&str> = record.derived_from.iter().map(|id| id.0.as_str()).collect();
    derived.sort_unstable();
    let _ = writeln!(out, "- derived_from: {}", list(derived.into_iter()));
    let _ = writeln!(
        out,
        "- revoked_ancestors: {}",
        list(record.revoked_ancestors.iter().map(|id| id.0.as_str()))
    );
    let _ = writeln!(out, "- created_at: {}", record.created_at);
    let _ = writeln!(out, "- updated_at: {}", record.updated_at);
    out.push('\n');
}

/// Renders every record of `store`, grouped by class and sorted by id.
pub fn render_memory_mirror(store: &DurableMemoryStore) -> String {
    let mut out = String::new();
    out.push_str("# PTR memory mirror\n\n");
    let _ = writeln!(
        out,
        "- committed_revision: {}",
        store.committed_revision().0
    );
    let _ = writeln!(out, "- logical_clock: {}", store.clock());
    let _ = writeln!(out, "- records: {}\n", store.len());
    let mut classes = [
        MemoryClass::Semantic,
        MemoryClass::Episodic,
        MemoryClass::Procedural,
        MemoryClass::Epistemic,
    ];
    classes.sort_by_key(|class| class_tag(*class));
    for class in classes {
        let _ = writeln!(out, "## {}\n", class_name(class));
        let mut any = false;
        for record in store.records_of(class) {
            any = true;
            write_record(&mut out, record);
        }
        if !any {
            out.push_str("(none)\n\n");
        }
    }
    out
}

/// Renders a [`KnowledgeStore`]: raw events by digest and size only, objects with their
/// effective lifecycle, sorted by key.
pub fn render_knowledge_mirror(store: &KnowledgeStore) -> String {
    let export = store.export();
    let mut out = String::new();
    out.push_str("# PTR knowledge mirror\n\n");
    let _ = writeln!(out, "- raw_events: {}", export.raw.len());
    let _ = writeln!(out, "- object_generations: {}\n", export.objects.len());
    out.push_str("## raw events\n\n");
    for event in &export.raw {
        let _ = writeln!(
            out,
            "- {} role={:?} session={} bytes={} sha256={} tool_call={}",
            escape(&event.id.0),
            event.role,
            escape(&event.session.0),
            event.content.len(),
            hex(&event.digest),
            event
                .tool_call
                .as_ref()
                .map_or_else(|| "-".to_owned(), |c| escape(&c.name)),
        );
    }
    out.push_str("\n## objects\n\n");
    for object in &export.objects {
        let key = crate::retention::KnowledgeKey {
            logical_id: object.id.clone(),
            generation: object.generation,
        };
        let effective = store
            .object_at(&key)
            .map_or(object.lifecycle, |o| o.lifecycle);
        let current = store
            .object(&object.id)
            .is_some_and(|o| o.generation == object.generation);
        let _ = writeln!(
            out,
            "### {} generation {}",
            escape(&object.id.0),
            object.generation.0
        );
        let _ = writeln!(out, "- current: {current}");
        let _ = writeln!(out, "- lifecycle: {effective:?}");
        let _ = writeln!(out, "- conflict: {:?}", object.conflict);
        let _ = writeln!(out, "- freshness: {:?}", object.freshness);
        let _ = writeln!(out, "- confidence: {:.4}", object.confidence.get());
        let _ = writeln!(
            out,
            "- sources: {}",
            list(object.sources.iter().map(|s| s.0.as_str()))
        );
        let dependencies: Vec<String> = object
            .dependencies
            .iter()
            .map(|d| format!("{}@{}", d.logical_id.0, d.generation.0))
            .collect();
        let _ = writeln!(
            out,
            "- dependencies: {}",
            list(dependencies.iter().map(String::as_str))
        );
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_neutralises_newlines_and_controls() {
        assert_eq!(escape("a\n# h\u{7}\\"), "a\\n# h\\u{7}\\\\");
    }

    #[test]
    fn hex_is_lowercase_fixed_width() {
        assert_eq!(hex(&[0, 15, 255]), "000fff");
    }
}
