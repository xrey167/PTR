use ptr_ledger::{InMemoryLedger,Ledger,LedgerEvent}; use ptr_types::Generation;
#[test] fn append_assigns_monotonic_index(){ let mut l=InMemoryLedger::default(); let a=l.append(LedgerEvent::Revoked{subject:"x".into(),generation:Generation(1)}); let b=l.append(LedgerEvent::Revoked{subject:"y".into(),generation:Generation(1)}); assert!(b.0>a.0); }
