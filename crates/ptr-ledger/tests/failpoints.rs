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

    assert!(FileLedger::open(&path).is_err());
    let recovered =
        FileLedger::recover_unacknowledged_tail(&path, ptr_ledger::integrity::LogAnchor::empty())
            .unwrap();
    assert!(recovered.events().is_empty());
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        ptr_ledger::integrity::LOG_MAGIC.len() as u64
    );

    drop(recovered);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn failed_append_poison_survives_unwind_until_reopen() {
    for (point, durable_count) in [
        ("ledger.before_record_write", 0),
        ("ledger.after_length_before_payload", 0),
        ("ledger.after_payload_before_sync", 1),
        ("ledger.after_sync_before_memory", 1),
    ] {
        let path = temp_path();
        let scenario = FailScenario::setup();
        let event = LedgerEvent::Revoked {
            subject: "capsule:poison".into(),
            generation: Generation(9),
        };
        let mut ledger = FileLedger::open(&path).unwrap();
        fail::cfg(point, "panic").unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| ledger.append_durable(event.clone()))).is_err());
        fail::remove(point);
        let length = std::fs::metadata(&path).unwrap().len();
        assert!(
            ledger.append_durable(event.clone()).is_err(),
            "{point} must poison the handle"
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            length,
            "poisoned writes must not touch disk"
        );
        assert!(ledger.events().is_empty());
        drop(ledger);
        scenario.teardown();
        let mut recovered = if point == "ledger.after_length_before_payload" {
            assert!(FileLedger::open(&path).is_err());
            FileLedger::recover_unacknowledged_tail(
                &path,
                ptr_ledger::integrity::LogAnchor::empty(),
            )
            .unwrap()
        } else {
            FileLedger::open(&path).unwrap()
        };
        // These are unwind tests, not power-loss tests: a fully written record
        // is visible to reopen even when the original call never acknowledged it.
        assert_eq!(recovered.events().len(), durable_count);
        let index = recovered.append_durable(event).unwrap();
        assert_eq!(index.0, durable_count as u64 + 1);
        drop(recovered);
        std::fs::remove_file(path).unwrap();
    }
}
