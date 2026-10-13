//! The drop-tracked payload every arm queues.
//!
//! A [`Tracked`] owns a body, carries the id the ledger knows it by, and reports its creation
//! and its drop to a shared [`DropLog`]. The log answers three questions no queue can fake: how
//! many payloads are alive (and how many bytes they hold), whether a payload was dropped while
//! a queue was meant to own it, and whether each payload was dropped exactly once.

use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::experiments::rng::Rng;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// FNV-1a over `bytes`, continuing from `state`. A fingerprint, not a security hash.
pub fn fnv(state: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(state, |acc, byte| {
        (acc ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

pub const FNV_START: u64 = FNV_OFFSET;

/// The fingerprint of a payload: its id and its body.
pub fn payload_digest(id: u32, body: &[u8]) -> u64 {
    fnv(fnv(FNV_OFFSET, &id.to_le_bytes()), body)
}

/// Creations and drops of payloads, shared by every `Tracked` of one run.
pub struct DropLog {
    live: AtomicI64,
    live_bytes: AtomicI64,
    drops: Box<[AtomicU8]>,
    recent: Mutex<Vec<u32>>,
}

impl DropLog {
    pub fn new(ids: usize) -> Arc<Self> {
        Arc::new(Self {
            live: AtomicI64::new(0),
            live_bytes: AtomicI64::new(0),
            drops: (0..ids).map(|_| AtomicU8::new(0)).collect(),
            recent: Mutex::new(Vec::new()),
        })
    }

    fn created(&self, bytes: usize) {
        self.live.fetch_add(1, Ordering::SeqCst);
        self.live_bytes.fetch_add(bytes as i64, Ordering::SeqCst);
    }

    fn dropped(&self, id: u32, bytes: usize) {
        self.live.fetch_sub(1, Ordering::SeqCst);
        self.live_bytes.fetch_sub(bytes as i64, Ordering::SeqCst);
        if let Some(slot) = self.drops.get(id as usize) {
            slot.fetch_add(1, Ordering::SeqCst);
        }
        self.recent
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(id);
    }

    /// Payloads created and not yet dropped.
    pub fn live(&self) -> i64 {
        self.live.load(Ordering::SeqCst)
    }

    /// Body bytes of the payloads created and not yet dropped.
    pub fn live_bytes(&self) -> i64 {
        self.live_bytes.load(Ordering::SeqCst)
    }

    /// How many times a payload of this id has been dropped.
    pub fn drops_of(&self, id: u32) -> u8 {
        self.drops
            .get(id as usize)
            .map_or(0, |slot| slot.load(Ordering::SeqCst))
    }

    /// The ids dropped since the last call, in drop order.
    pub fn take_recent(&self) -> Vec<u32> {
        std::mem::take(&mut *self.recent.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// A message: an id, a body, and a stamp the threaded arm reads for its wall-clock latency.
pub struct Tracked {
    id: u32,
    body: Box<[u8]>,
    stamp_ns: u64,
    log: Arc<DropLog>,
}

impl Tracked {
    pub fn new(log: &Arc<DropLog>, id: u32, body: Box<[u8]>, stamp_ns: u64) -> Self {
        log.created(body.len());
        Self {
            id,
            body,
            stamp_ns,
            log: Arc::clone(log),
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn stamp_ns(&self) -> u64 {
        self.stamp_ns
    }

    pub fn digest(&self) -> u64 {
        payload_digest(self.id, &self.body)
    }

    /// Alter the body in place (only a faulty queue of the negative controls does this).
    pub fn corrupt(&mut self) {
        if let Some(first) = self.body.first_mut() {
            *first ^= 0xFF;
        }
    }

    /// A second payload with this one's id and body, as a duplicating queue would deliver.
    pub fn forge_copy(&self) -> Self {
        Self::new(&self.log, self.id, self.body.clone(), self.stamp_ns)
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.log.dropped(self.id, self.body.len());
    }
}

/// The bodies of one run's messages, generated once from the seed, and their fingerprints.
pub struct Corpus {
    bodies: Vec<Box<[u8]>>,
    digests: Vec<u64>,
}

impl Corpus {
    pub fn generate(seed: u64, ids: usize, body_bytes: usize) -> Self {
        let mut bodies = Vec::with_capacity(ids);
        let mut digests = Vec::with_capacity(ids);
        for id in 0..ids {
            let mut rng = Rng::new(seed ^ (id as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
            let mut body = vec![0u8; body_bytes];
            for chunk in body.chunks_mut(8) {
                let word = rng.next_u64().to_le_bytes();
                chunk.copy_from_slice(&word[..chunk.len()]);
            }
            digests.push(payload_digest(id as u32, &body));
            bodies.push(body.into_boxed_slice());
        }
        Self { bodies, digests }
    }

    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn digests(&self) -> &[u64] {
        &self.digests
    }

    pub fn make(&self, log: &Arc<DropLog>, id: u32, stamp_ns: u64) -> Tracked {
        Tracked::new(log, id, self.bodies[id as usize].clone(), stamp_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_counts_live_payloads_bytes_and_drops_per_id() {
        let log = DropLog::new(4);
        let corpus = Corpus::generate(1, 4, 16);
        let a = corpus.make(&log, 2, 0);
        let b = corpus.make(&log, 3, 0);
        assert_eq!((log.live(), log.live_bytes()), (2, 32));
        drop(a);
        assert_eq!((log.live(), log.live_bytes(), log.drops_of(2)), (1, 16, 1));
        assert_eq!(log.take_recent(), [2]);
        assert!(log.take_recent().is_empty());
        drop(b);
        assert_eq!((log.live(), log.live_bytes()), (0, 0));
    }

    #[test]
    fn a_forged_copy_has_the_same_digest_and_counts_as_a_second_payload() {
        let log = DropLog::new(2);
        let corpus = Corpus::generate(7, 2, 64);
        let original = corpus.make(&log, 1, 0);
        let copy = original.forge_copy();
        assert_eq!(original.digest(), copy.digest());
        assert_eq!(original.digest(), corpus.digests()[1]);
        drop(original);
        drop(copy);
        assert_eq!(log.drops_of(1), 2);
    }

    #[test]
    fn corrupting_a_body_changes_the_digest() {
        let log = DropLog::new(1);
        let corpus = Corpus::generate(7, 1, 8);
        let mut payload = corpus.make(&log, 0, 0);
        let before = payload.digest();
        payload.corrupt();
        assert_ne!(payload.digest(), before);
        assert_eq!(before, corpus.digests()[0]);
    }

    #[test]
    fn the_corpus_is_a_function_of_the_seed_with_distinct_bodies() {
        let first = Corpus::generate(17, 50, 64);
        let second = Corpus::generate(17, 50, 64);
        let other = Corpus::generate(18, 50, 64);
        assert_eq!(first.digests(), second.digests());
        assert_ne!(first.digests(), other.digests());
        let mut distinct = first.digests().to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 50);
        assert_eq!(first.len(), 50);
    }

    #[test]
    fn fnv_matches_the_published_vectors() {
        assert_eq!(fnv(FNV_START, b""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv(FNV_START, b"a"), 0xAF63_DC4C_8601_EC8C);
        assert_eq!(fnv(FNV_START, b"foobar"), 0x8594_4171_F739_67E8);
    }
}
