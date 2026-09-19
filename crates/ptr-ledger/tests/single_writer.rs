use ptr_ledger::{FileLedger, LedgerEvent};
use ptr_types::Generation;
use std::io::ErrorKind;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROBE: &str = "PTR_LEDGER_LOCK_TEST_PATH";

#[test]
fn lock_probe_child() {
    let Some(path) = std::env::var_os(PROBE) else {
        return;
    };
    let error = FileLedger::open(path)
        .err()
        .expect("another process owns the ledger");
    assert_eq!(error.kind(), ErrorKind::WouldBlock);
}

#[test]
fn one_writer_owns_recovery_and_append_across_processes() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "ptr-single-writer-{}-{nonce}.log",
        std::process::id()
    ));
    let mut first = FileLedger::open(&path).unwrap();
    first
        .append_durable(LedgerEvent::Revoked {
            subject: "a".into(),
            generation: Generation(1),
        })
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let error = FileLedger::open(&path)
        .err()
        .expect("a second handle must fail");
    assert_eq!(error.kind(), ErrorKind::WouldBlock);
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "lock_probe_child", "--nocapture"])
        .env(PROBE, &path)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    drop(first);
    let mut reopened = FileLedger::open(&path).unwrap();
    assert_eq!(reopened.events().len(), 1);
    assert_eq!(
        reopened
            .append_durable(LedgerEvent::Revoked {
                subject: "b".into(),
                generation: Generation(2)
            })
            .unwrap()
            .0,
        2
    );
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}
