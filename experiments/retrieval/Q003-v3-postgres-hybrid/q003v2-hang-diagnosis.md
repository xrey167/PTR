# Q003-v2 seed 29: the snapshot taken when the run was stopped, and what it shows

## The snapshot (verbatim, as taken by the head on 2026-10-11)

```
seed 29 started 2026-10-10 20:03 UTC; ptr-bench pid 9695 cpu 6 s; threads:
  task 19425 wchan=anon_pipe_read syscall=0 0x5
  task 19432 wchan=anon_pipe_read syscall=0 0x7
  task 19440 wchan=anon_pipe_read syscall=0 0x9
  task 9695 wchan=futex_do_wait syscall=202 0x56287eaabbc8
ptr-bench fds: 0 socket, 1 and 2 pipes (the runner's), 3 and 11 epoll, 10 eventfd, 12 to 16 sockets (the PostgreSQL connections),
  4 pipe (stdin of the reference driver), 5 pipe (its stdout), 6 pipe (stdin of the BM25 driver), 7 pipe (its stdout),
  8 pipe (stdin of the strong driver), 9 pipe (its stdout)
child 19424: python3 research/baselines/rag_reference/q003_driver.py      wchan=anon_pipe_read syscall=0 0x0   (fd 0 = pipe 9583577)
child 19431: python3 experiments/retrieval/Q003-v2-postgres-hybrid/bm25_driver.py   wchan=anon_pipe_read syscall=0 0x0   (fd 0 = pipe 9583579)
child 19439: <strong interpreter> research/baselines/strong_retrieval/driver.py --reranker bge --models-root ...   wchan=0 syscall=running
```

The start time above is the committed record's (20:03 UTC on 2026-10-10, as in `SUPERSEDED.md` and section 12 of the preregistration). The snapshot as first transcribed read 22:13, which does not match the record and is probably a local-time reading; it is corrected here and nothing depends on it.

First reading: a pipe-protocol deadlock, "ptr-bench waited on the BM25 driver's stdout while the driver waited on its stdin".

**Correction, stated plainly.** The head's first reading (a pipe deadlock) was wrong. The snapshot is consistent with an extremely slow
strong arm under torch thread oversubscription on the shared 4-core host. That is not proven for seed 29 itself (one sample, no
CPU time of the strong driver). The v2 harness additionally had real hang paths (listed below) that Q003-v3 closes.

## What the snapshot shows

1. The three `ptr-bench` threads in `anon_pipe_read` are the **reader threads** of the three bridges (`bridge.rs`: one thread per
   driver, blocked in `read` on that driver's stdout for as long as the driver lives). They sit there in a healthy run, all the
   time, whatever the main thread is doing. Fds 0x5, 0x7 and 0x9 are the three drivers' stdout pipes. Their state is no evidence of
   a request in flight on any one of them.
2. The reference and BM25 drivers blocked on their own **stdin** is their idle state: for each query the harness asks the product, then the reference, then BM25 and
   only then the strong arm (`runner.rs`, `queries`), each request synchronous; while the strong arm works they wait for the next request.
3. The main thread (`task 9695`) is in `futex_do_wait` (syscall 202), the state of `Receiver::recv_timeout` (the bridge's wait for a
   reply). It is **not** in `write` (a pipe-full deadlock would show `pipe_write`) and not in `epoll_wait`/`futex` of the
   PostgreSQL runtime (`block_on` on the current-thread runtime parks in `epoll_wait`). So it was waiting for a driver's reply.
4. Of the three drivers only one was not waiting for input: the **strong driver was running** (`syscall=running`), that is,
   executing, not blocked. A driver the harness is waiting on that is on the CPU is computing its answer. The one driver whose
   state would fit a deadlock (BM25 waiting on stdin while the harness waits on its stdout) is the one the main thread could not
   have been waiting on, since the harness sends it a request and reads its reply before it asks the strong arm.
5. `ptr-bench`'s CPU time of 6 s over 13 hours is what a thread that only waits uses. It says nothing about the drivers, whose CPU
   time was not recorded; the strong driver's would have told slow from stuck.

So the snapshot is **consistent with a slow strong arm and not with the pipe deadlock it was first read as**. It does not prove
slowness either: one sample, no CPU time of the strong driver, no progress line (the harness's stderr is held by the runner until
the run ends).

## Evidence that "slow" is what this host does

Seed 17 of Q003-v2 completed (exit 0, 30,693 s). 30,350 s of it were the strong arm, for 200 queries that the preregistration
prices at about 8 CPU-seconds each (about 1,600 s). Its progress lines (in its record) give the strong arm's seconds so far at each
50-query window of the second checkpoint: 1,275, 1,378, 9,694, 24,294, 30,349, then unchanged. About ten of every 50 queries are
strong queries, so one window cost 8,300 s (about 830 s a query) and the next 14,600 s (about 1,460 s a query): a 100- to 180-fold
slowdown of the cross-encoder against its price, on a host (4 logical CPUs, 16 GB) that was shared with other jobs (load average
8 to 10 when seed 29 was stopped; torch runs 4 threads by default, so a loaded host oversubscribes them). A run that has spent 8.5 hours on 200
queries and 13 hours on the next seed is, on this evidence, a run on an overloaded host.

## What was wrong with the v2 harness, whichever it was

* **No way to tell slow from stuck.** No heartbeat, no phase, no record of the slowest request; the runner holds the harness's
  standard error until it exits, so a 13-hour run showed nothing.
* **Limits that could not fire.** The only bound was one hour per reply (`REPLY_TIMEOUT`), which a driver that keeps answering
  slowly never meets, and the bound started only after the request line was written.
* **An unbounded write.** The request line was written on the calling thread (`write_all`, then `flush`) before the timeout
  started: a driver that stops reading its input while a request larger than the pipe's buffer (64 KiB) is being written blocked
  the harness for as long as the driver lived. This is a genuine hang path of the v2 code (an `upsert_many` line is 0.6 MB for the reference and about 2 MB for the strong arm); it was not what the snapshot shows (the main thread was not in `write`).
* **No bound on the whole run**, and a driver's failure at the end (a non-zero exit after `quit`) was ignored.
* **An environment-dependent buffer.** A Python driver that does not flush its replies hangs a harness that waits for them only
  where Python is not unbuffered; an operator shell that sets `PYTHONUNBUFFERED` (as the one used for this channel's development
  does) hides the defect. The v2 drivers do flush; nothing tested that the harness would notice if one did not.

## What Q003-v3 changes (the harness only)

See section 12 of `PREREGISTRATION.md`: the request line is written by a thread of its own and the exchange (write, flush and
reply) is bounded by one request limit (7,200 s) that kills the driver when it passes, with the typed reason `request-timeout`;
a driver that exits or closes its output is `driver-exited` with its exit status, and one that ends badly after `quit` is a stop
too; a watchdog ends a run that passes its budget (86,400 s) with `run-deadline`; every 600 s the run prints its phase, the request
in flight and the slowest request; the harness removes `PYTHONUNBUFFERED` from the drivers' environment so that a missing flush
shows the same way everywhere. Every stop is exit status 2 and the line `harness stop [<reason>]` on standard error. Each path has
a unit test and a planted mutation (`tests/mutations.toml`).

What this does not do: it does not make the strong arm fast. On a loaded host a seed can still take many hours, and a seed that
a limit stops is spent (a listed experiment runs each seed once). The confirmatory seeds are to be run one after another, and the
runner's command pins the strong arm to ONE torch thread (`--strong-threads 1`; `PREREGISTRATION.md` section 12).

## Measured afterwards: the thread oversubscription

Same pinned workload, same loaded 4-core host (load average 10 to 15): at the default 4 torch threads (pilot 1001) a strong search took
57 to 877 s, the strong driver used more than 11,000 CPU-seconds for about 30 strong queries (priced: 8 CPU-seconds each) and its
involuntary context switches ran to 17 million in 3 hours (OpenMP-style spin contention; CPU time is inflated, so it is not a wait);
at 1 thread (pilot 1002) a strong search took about 15 s, the slowest 19 s, and the whole pinned seed 3,365 s.
