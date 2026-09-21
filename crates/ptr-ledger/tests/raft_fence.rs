//! A fence that excludes a writer which is **not** running the protocol.
//!
//! This is the distinction `31-cluster-integrity.md` insists on. Raft's rules are
//! a safety argument among members that run the protocol correctly: a deposed
//! leader may append, it cannot gather a majority, and the entry only it held is
//! overwritten. That argument says nothing about a writer that does not
//! participate at all — one partitioned from its peers but *not* from the storage
//! they share, or one resumed from a stale image. A local advisory file lock does
//! not help either: it makes one writer per file on one machine and says nothing
//! about a second machine.
//!
//! What excludes such a writer is a fencing token. The storage records the highest
//! term it has ever accepted, and a write presenting a lower term is refused —
//! read from disk on every write, because a stale writer's in-memory copy of the
//! token is its own stale copy.
//!
//! Every test here opens **two** `FileRaftStorage` handles on one directory. That
//! is the shared-storage case in miniature: two writers, one set of files, neither
//! aware of the other except through what is on disk.
#![cfg(feature = "raft-rs-backend")]

use ptr_ledger::FileRaftStorage;
use raft::prelude::{Entry, HardState};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-raft-fence-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const VOTERS: [u64; 3] = [1, 2, 3];

fn at_term(term: u64) -> HardState {
    HardState {
        term,
        vote: 0,
        commit: 0,
    }
}

fn entry(index: u64, term: u64) -> Entry {
    let mut entry = Entry::default();
    entry.index = index;
    entry.term = term;
    entry.data = b"an entry a fenced writer must not land".to_vec().into();
    entry
}

/// The storage's two files, for asserting a refusal left nothing behind.
fn files(temp: &Temp) -> (Vec<u8>, Vec<u8>) {
    let read = |name: &str| std::fs::read(temp.0.join(name)).unwrap_or_default();
    (read("state"), read("log"))
}

#[test]
fn a_writer_at_a_superseded_term_is_refused_by_the_storage_it_shares() {
    let temp = Temp::new("shared");

    // Two processes, one directory. `stale` opens first and never hears anything
    // again — it is partitioned from its peers, not from their disk.
    let stale = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();
    let current = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();

    current.wl().set_hard_state(&at_term(7)).unwrap();

    let before = files(&temp);
    let error = stale
        .wl()
        .set_hard_state(&at_term(2))
        .expect_err("a writer at a superseded term must be refused");
    assert!(
        error.to_string().contains("PTR_RAFT_FENCED"),
        "the refusal must name the fence, got: {error}"
    );
    assert!(
        error.to_string().contains("term 7"),
        "and the term that claimed the storage, got: {error}"
    );
    assert_eq!(
        files(&temp),
        before,
        "a fenced write must leave the files exactly as they were"
    );
}

#[test]
fn a_fenced_writer_cannot_append_entries_either() {
    let temp = Temp::new("append");
    let stale = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();
    let current = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();
    current.wl().set_hard_state(&at_term(9)).unwrap();

    let before = files(&temp);
    let error = stale
        .wl()
        .append(&[entry(1, 2)])
        .expect_err("an append is a write to the same shared storage");
    assert!(error.to_string().contains("PTR_RAFT_FENCED"), "{error}");
    assert_eq!(
        files(&temp),
        before,
        "the log must not move for a fenced writer"
    );
}

/// The control.
///
/// Both refusals above would hold against a storage that refused *every* second
/// writer, which would be a lock rather than a fence and would break a legitimate
/// restart. So the same handle, once it is no longer behind, must be able to write.
#[test]
fn the_same_writer_proceeds_once_it_is_no_longer_behind() {
    let temp = Temp::new("control");
    let behind = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();
    let current = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();
    current.wl().set_hard_state(&at_term(7)).unwrap();

    assert!(
        behind.wl().set_hard_state(&at_term(2)).is_err(),
        "behind the fence, refused"
    );

    // Learning the higher term is exactly what a member that rejoins the protocol
    // does. The fence excludes a writer that is behind, not a second writer.
    behind
        .wl()
        .set_hard_state(&at_term(7))
        .expect("a writer that has caught up is not fenced");
    behind
        .wl()
        .set_hard_state(&at_term(8))
        .expect("nor is one that moves the term on");
}

#[test]
fn the_fence_survives_reopening_the_storage() {
    let temp = Temp::new("durable");
    {
        let current = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();
        current.wl().set_hard_state(&at_term(11)).unwrap();
    }

    // A process resumed from a stale image: it believes an old term, and the
    // storage it reaches has moved on. Reopening does not clear the token,
    // because the token is in the file rather than in the process.
    let resumed = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();
    assert_eq!(
        resumed.wl().hard_state().term,
        11,
        "reopening recovers the recorded term"
    );

    resumed.wl().set_hard_state(&at_term(3)).expect_err(
        "a resumed writer that believes an older term must not be able to write \
         beneath the recorded one",
    );
}

#[test]
fn an_unclaimed_directory_is_not_fenced() {
    let temp = Temp::new("fresh");
    let first = FileRaftStorage::open(&temp.0, &VOTERS, &[]).unwrap();
    // Nothing has claimed this storage, so there is nothing to be fenced by.
    // Without this, a fresh member could never make its first write.
    first
        .wl()
        .set_hard_state(&at_term(1))
        .expect("a first writer is not fenced by an empty directory");
}
