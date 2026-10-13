//! Shared helpers for the integration tests (no test annotations here).
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ptr_events::{BusRecord, EventClass, NewRecord};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique directory under the system temp dir, removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(label: &str) -> Self {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("ptr-events-{label}-{}-{id}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    /// Segment files in offset order.
    pub fn segments(&self) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.0)
            .map(|dir| {
                dir.filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("segment-"))
                    })
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        files
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn projection(key: &str) -> NewRecord {
    NewRecord {
        class: EventClass::Projection,
        topic: "capsule.committed".into(),
        key: key.into(),
        payload: key.as_bytes().to_vec(),
    }
}

pub fn telemetry(key: &str) -> NewRecord {
    NewRecord {
        class: EventClass::Telemetry,
        topic: "execution.trace".into(),
        key: key.into(),
        payload: key.as_bytes().to_vec(),
    }
}

pub fn keys(records: &[BusRecord]) -> Vec<String> {
    records.iter().map(|r| r.key.clone()).collect()
}
