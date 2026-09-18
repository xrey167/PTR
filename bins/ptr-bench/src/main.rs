use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_exec::Mailbox;
use ptr_ledger::{FileLedger, LedgerEvent};
use ptr_runtime::PtrRuntime;
use ptr_semdb::{SemanticDelta, SemanticHost};
use ptr_types::{CapabilityId, CapsuleId, Effect, Generation, ProjectId, TypeId};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let command = args.get(1).map(String::as_str).unwrap_or("all");

    match command {
        "semdb" => bench_semdb(parse_usize(&args, 2, 10_000)),
        "mailbox" => bench_mailbox(parse_usize(&args, 2, 10_000)),
        "ledger-recovery" => {
            bench_ledger_recovery(parse_usize(&args, 2, 10_000), parse_u64(&args, 3, 17))
        }
        "ledger-process-crash" => {
            bench_ledger_process_crash(parse_usize(&args, 2, 100), parse_u64(&args, 3, 17))
        }
        "ledger-crash-child" => {
            let path = args.get(2).expect("ledger-crash-child requires path");
            let subject = args.get(3).expect("ledger-crash-child requires subject");
            let partial_len = parse_usize(&args, 4, 7);
            crash_child(Path::new(path), subject, partial_len);
        }
        "all" => {
            let iterations = parse_usize(&args, 2, 10_000);
            bench_semdb(iterations);
            bench_mailbox(iterations);
        }
        _ => {
            eprintln!(
                "usage: ptr-bench [all|semdb|mailbox|ledger-recovery|ledger-process-crash] [iterations] [seed]"
            );
            std::process::exit(2);
        }
    }
}

fn parse_usize(args: &[String], index: usize, default: usize) -> usize {
    args.get(index)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_u64(args: &[String], index: usize, default: u64) -> u64 {
    args.get(index)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn bench_semdb(iterations: usize) {
    let mut host = SemanticHost::default();
    for i in 0..128 {
        host.dependencies_mut()
            .depends_on(format!("derived:{i}"), format!("input:{i}"));
    }

    let start = Instant::now();
    let mut affected_total = 0usize;
    for i in 0..iterations {
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert(format!("input:{}", i % 128), i.to_string());
        let (_, affected) = host.apply_delta(delta);
        affected_total += affected.len();
    }
    let elapsed = start.elapsed();
    println!(
        r#"{{"benchmark":"semdb","iterations":{},"elapsed_ns":{},"ns_per_op":{},"affected_total":{}}}"#,
        iterations,
        elapsed.as_nanos(),
        elapsed.as_nanos() / iterations.max(1) as u128,
        affected_total
    );
}

fn bench_mailbox(iterations: usize) {
    let mailbox = Mailbox::bounded(1);
    let start = Instant::now();
    for i in 0..iterations {
        mailbox.try_send(i).expect("capacity available");
        let value = mailbox.try_recv().expect("message available");
        std::hint::black_box(value);
    }
    let elapsed = start.elapsed();
    println!(
        r#"{{"benchmark":"mailbox","iterations":{},"elapsed_ns":{},"ns_per_roundtrip":{}}}"#,
        iterations,
        elapsed.as_nanos(),
        elapsed.as_nanos() / iterations.max(1) as u128
    );
}

fn bench_ledger_recovery(iterations: usize, seed: u64) {
    let start = Instant::now();
    let mut rng = seed;
    let mut false_accepts = 0usize;
    let mut recovery_errors = 0usize;
    let mut tail_trim_errors = 0usize;

    for i in 0..iterations {
        let subject = format!("capsule:{seed}:{i}");
        let path = std::env::temp_dir().join(format!(
            "ptr-ledger-recovery-{}-{seed}-{i}.log",
            std::process::id()
        ));

        {
            let mut ledger = FileLedger::open(&path).expect("open reference ledger");
            ledger
                .append_durable(LedgerEvent::CapsuleCommitted {
                    project: ProjectId::from("l001"),
                    capsule: CapsuleId(subject.clone()),
                    generation: Generation(1),
                })
                .expect("commit capsule");
            ledger
                .append_durable(LedgerEvent::Revoked {
                    subject: subject.clone(),
                    generation: Generation(1),
                })
                .expect("commit revocation");
        }

        let durable_len = std::fs::metadata(&path)
            .expect("durable ledger metadata")
            .len();

        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let partial_len = (rng % 31) as usize;
        {
            let mut file = OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append crash tail");
            file.write_all(&128_u32.to_le_bytes())
                .expect("write incomplete record prefix");
            file.write_all(&vec![0xA5; partial_len])
                .expect("write incomplete crash tail");
            file.flush().expect("flush simulated crash tail");
        }

        let reopened = FileLedger::open(&path).expect("recover reference ledger");
        if reopened.events().len() != 2 {
            recovery_errors += 1;
        }
        if std::fs::metadata(&path).expect("recovered metadata").len() != durable_len {
            tail_trim_errors += 1;
        }

        if stale_action_accepted(&subject, reopened.events()) {
            false_accepts += 1;
        }

        drop(reopened);
        std::fs::remove_file(&path).expect("cleanup recovery ledger");
    }

    let elapsed = start.elapsed();
    println!(
        r#"{{"benchmark":"ledger-recovery","iterations":{},"seed":{},"elapsed_ns":{},"false_accepts":{},"recovery_errors":{},"tail_trim_errors":{}}}"#,
        iterations,
        seed,
        elapsed.as_nanos(),
        false_accepts,
        recovery_errors,
        tail_trim_errors
    );
}

fn bench_ledger_process_crash(iterations: usize, seed: u64) {
    let start = Instant::now();
    let mut rng = seed;
    let mut false_accepts = 0usize;
    let mut recovery_errors = 0usize;
    let mut tail_trim_errors = 0usize;
    let mut child_exit_errors = 0usize;
    let exe = std::env::current_exe().expect("locate ptr-bench executable");

    for i in 0..iterations {
        let subject = format!("process-capsule:{seed}:{i}");
        let path = std::env::temp_dir().join(format!(
            "ptr-ledger-process-crash-{}-{seed}-{i}.log",
            std::process::id()
        ));

        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let partial_len = (rng % 31) as usize;

        let status = Command::new(&exe)
            .arg("ledger-crash-child")
            .arg(&path)
            .arg(&subject)
            .arg(partial_len.to_string())
            .status()
            .expect("spawn crash child");
        if status.success() {
            child_exit_errors += 1;
        }

        let before_recovery = std::fs::metadata(&path)
            .expect("crash child created ledger")
            .len();
        let reopened = FileLedger::open(&path).expect("recover process-crashed ledger");
        let after_recovery = std::fs::metadata(&path)
            .expect("recovered process ledger metadata")
            .len();

        if reopened.events().len() != 2 {
            recovery_errors += 1;
        }
        if after_recovery >= before_recovery {
            tail_trim_errors += 1;
        }
        if stale_action_accepted(&subject, reopened.events()) {
            false_accepts += 1;
        }

        drop(reopened);
        std::fs::remove_file(&path).expect("cleanup process-crash ledger");
    }

    let elapsed = start.elapsed();
    println!(
        r#"{{"benchmark":"ledger-process-crash","iterations":{},"seed":{},"elapsed_ns":{},"false_accepts":{},"recovery_errors":{},"tail_trim_errors":{},"child_exit_errors":{}}}"#,
        iterations,
        seed,
        elapsed.as_nanos(),
        false_accepts,
        recovery_errors,
        tail_trim_errors,
        child_exit_errors
    );
}

fn crash_child(path: &Path, subject: &str, partial_len: usize) -> ! {
    {
        let mut ledger = FileLedger::open(path).expect("open crash-child ledger");
        ledger
            .append_durable(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("l001-process"),
                capsule: CapsuleId(subject.to_owned()),
                generation: Generation(1),
            })
            .expect("commit capsule before process crash");
        ledger
            .append_durable(LedgerEvent::Revoked {
                subject: subject.to_owned(),
                generation: Generation(1),
            })
            .expect("commit revocation before process crash");
    }

    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("append incomplete crash record");
    file.write_all(&128_u32.to_le_bytes())
        .expect("write incomplete record length");
    file.write_all(&vec![0x5A; partial_len])
        .expect("write incomplete record bytes");
    file.flush().expect("flush crash tail to OS before abort");
    std::process::abort();
}

fn stale_action_accepted(subject: &str, events: &[ptr_ledger::CommittedEvent]) -> bool {
    let mut runtime = PtrRuntime::replay(PtrConfig::default(), events).expect("replay ledger");
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("memory.write"));
    runtime.permissions_mut().allow_mutation = true;

    let action = ActionIr {
        operation: "update".into(),
        target: subject.to_owned(),
        capability: CapabilityId::from("memory.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("SemanticCapsule"),
        generation: Generation(1),
        revision: runtime.revision(),
        payload: vec![],
    };

    runtime.authorize_action(&action).is_ok()
}
