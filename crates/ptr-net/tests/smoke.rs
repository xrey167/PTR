use ptr_net::{ALPN_PODWIRE, ALPN_RAFT};
#[test]
fn alpns_are_distinct() {
    assert_ne!(ALPN_RAFT, ALPN_PODWIRE);
}
