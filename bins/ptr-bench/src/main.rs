use ptr_exec::Mailbox;
use ptr_semdb::{SemanticDelta, SemanticHost};
use std::time::Instant;

fn main() {
    let command = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let iterations = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10_000);

    match command.as_str() {
        "semdb" => bench_semdb(iterations),
        "mailbox" => bench_mailbox(iterations),
        "all" => {
            bench_semdb(iterations);
            bench_mailbox(iterations);
        }
        _ => {
            eprintln!("usage: ptr-bench [all|semdb|mailbox] [iterations]");
            std::process::exit(2);
        }
    }
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
        "{{"benchmark":"semdb","iterations":{},"elapsed_ns":{},"ns_per_op":{},"affected_total":{}}}",
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
        "{{"benchmark":"mailbox","iterations":{},"elapsed_ns":{},"ns_per_roundtrip":{}}}",
        iterations,
        elapsed.as_nanos(),
        elapsed.as_nanos() / iterations.max(1) as u128
    );
}
