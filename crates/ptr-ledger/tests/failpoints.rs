use fail::FailScenario;
use ptr_ledger::{FileLedger, LedgerEvent};
use ptr_types::Generation;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ptr-ledger-failpoint-{nonce}.log"))
}

#[test]
fn panic_after_length_prefix_recovers_without_phantom_event() {
    let path = temp_path();
    let scenario = FailScenario::setup();

    let mut ledger = FileLedger::open(&path).unwrap();
    fail::cfg("ledger.after_length_before_payload", "panic").unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        ledger
            .append_durable(LedgerEvent::Revoked {
                subject: "capsule:a".into(),
                generation: Generation(7),
            })
            .unwrap();
    }));
    assert!(result.is_err());

    drop(ledger);
    scenario.teardown();

    let recovered = FileLedger::open(&path).unwrap();
    assert!(recovered.events().is_empty());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);

    std::fs::remove_file(path).unwrap();
}
