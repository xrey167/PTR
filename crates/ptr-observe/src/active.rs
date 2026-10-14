//! Active-tree diagnostics: a snapshot of currently open spans as a parent/child tree with ages
//! relative to the tracer clock, plus stuck-span detection.
//!
//! This is the std-only stand-in for async-backtrace: it shows which instrumented spans are open
//! and for how long, not the await-points of arbitrary futures. Only spans the program opened
//! through [`Tracer`] are visible.

use crate::{SpanContext, SpanId, Tracer};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSpan {
    pub context: SpanContext,
    pub parent: Option<SpanId>,
    pub name: String,
    pub start_tick: u64,
    /// Snapshot tick minus `start_tick`.
    pub age: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StuckSpan {
    pub context: SpanContext,
    pub age: u64,
    /// Span names from the outermost reported ancestor down to the stuck span. Bounded by
    /// [`STUCK_PATH_MAX_DEPTH`] entries and [`STUCK_PATH_MAX_BYTES`] bytes of ancestor names;
    /// when ancestors were left out, the outermost ones are the ones omitted.
    pub path: Vec<String>,
    /// `true` when `path` omits outer open ancestors because of the bounds above.
    pub path_truncated: bool,
}

/// Maximum number of names (the stuck span included) reported in [`StuckSpan::path`].
pub const STUCK_PATH_MAX_DEPTH: usize = 16;
/// Maximum total bytes of ancestor names reported in [`StuckSpan::path`]; the stuck span's own
/// name is always reported and does not count against it.
pub const STUCK_PATH_MAX_BYTES: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveTree {
    pub at_tick: u64,
    /// Open spans ordered by `(start_tick, span id)`.
    pub spans: Vec<ActiveSpan>,
}

fn format_ticks(ns: u64) -> String {
    match ns {
        0..=999 => format!("{ns}ns"),
        1_000..=999_999 => format!("{}.{:03}us", ns / 1_000, ns % 1_000),
        1_000_000..=999_999_999 => format!("{}.{:03}ms", ns / 1_000_000, (ns / 1_000) % 1_000),
        _ => format!("{}.{:03}s", ns / 1_000_000_000, (ns / 1_000_000) % 1_000),
    }
}

impl Tracer {
    /// Snapshots the currently open spans.
    pub fn active_tree(&self) -> ActiveTree {
        let (at_tick, views) = self.open_snapshot();
        let mut spans: Vec<ActiveSpan> = views
            .into_iter()
            .map(|view| ActiveSpan {
                context: SpanContext {
                    trace_id: view.trace_id,
                    span_id: view.id,
                },
                parent: view.parent,
                name: view.name,
                start_tick: view.start_tick,
                age: at_tick.saturating_sub(view.start_tick),
            })
            .collect();
        spans.sort_by_key(|span| (span.start_tick, span.context.span_id));
        ActiveTree { at_tick, spans }
    }
}

impl ActiveTree {
    fn find(&self, id: SpanId) -> Option<&ActiveSpan> {
        self.spans.iter().find(|span| span.context.span_id == id)
    }

    fn is_root(&self, span: &ActiveSpan) -> bool {
        span.parent.is_none_or(|parent| self.find(parent).is_none())
    }

    /// Depth-first rendering, children ordered by start. A span whose parent already finished is
    /// shown as a root and marked `(parent closed)`.
    pub fn render_text(&self) -> String {
        let mut out = format!(
            "active spans at tick {}: {}\n",
            self.at_tick,
            self.spans.len()
        );
        let mut stack: Vec<(&ActiveSpan, usize)> = self
            .spans
            .iter()
            .rev()
            .filter(|span| self.is_root(span))
            .map(|span| (span, 0))
            .collect();
        while let Some((span, depth)) = stack.pop() {
            let orphan = depth == 0 && span.parent.is_some();
            out.push_str(&format!(
                "{}- {} [{}] open {}{}\n",
                "  ".repeat(depth),
                span.name,
                span.context.span_id,
                format_ticks(span.age),
                if orphan { " (parent closed)" } else { "" }
            ));
            for child in self
                .spans
                .iter()
                .rev()
                .filter(|child| child.parent == Some(span.context.span_id))
            {
                stack.push((child, depth + 1));
            }
        }
        out
    }

    /// Spans open for at least `threshold` ticks, oldest first.
    pub fn stuck(&self, threshold: u64) -> Vec<StuckSpan> {
        self.spans
            .iter()
            .filter(|span| span.age >= threshold)
            .map(|span| {
                let mut path = vec![span.name.clone()];
                let mut ancestor_bytes = 0usize;
                let mut path_truncated = false;
                let mut cursor = span.parent;
                while let Some(parent) = cursor.and_then(|id| self.find(id)) {
                    if path.len() >= STUCK_PATH_MAX_DEPTH
                        || ancestor_bytes.saturating_add(parent.name.len()) > STUCK_PATH_MAX_BYTES
                    {
                        path_truncated = true;
                        break;
                    }
                    ancestor_bytes += parent.name.len();
                    path.push(parent.name.clone());
                    cursor = parent.parent;
                }
                path.reverse();
                StuckSpan {
                    context: span.context,
                    age: span.age,
                    path,
                    path_truncated,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_formatting_picks_unit() {
        assert_eq!(format_ticks(999), "999ns");
        assert_eq!(format_ticks(1_500), "1.500us");
        assert_eq!(format_ticks(2_250_000), "2.250ms");
        assert_eq!(format_ticks(3_000_000_000), "3.000s");
    }
}
