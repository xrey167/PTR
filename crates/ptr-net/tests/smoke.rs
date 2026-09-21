use ptr_net::{ALPN_BLOB, ALPN_EVENTS, ALPN_EXEC, ALPN_MODEL, ALPN_PODWIRE, ALPN_RAFT};

#[test]
fn every_alpn_is_distinct() {
    // Two protocols sharing an ALPN is how a request meant for one gets parsed by the
    // other, and the parse that succeeds by accident is the dangerous one. So this
    // holds every declared ALPN to account rather than one pair of them.
    let alpns = [
        ALPN_RAFT,
        ALPN_PODWIRE,
        ALPN_MODEL,
        ALPN_BLOB,
        ALPN_EVENTS,
        ALPN_EXEC,
    ];
    let total = alpns.len();
    let mut seen: Vec<&[u8]> = alpns.to_vec();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), total, "two protocols share an ALPN");
}
