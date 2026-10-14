//! The confinement probe and the seccomp allowlist through the public API. Linux only, feature
//! `worker-confine` (the Cargo manifest requires it).
//!
//! The filter is installed in a child process: the test re-executes this test binary with an
//! environment variable and `--exact confine_child_helper`, so no `fork` happens in the
//! multithreaded test harness and a filter can never confine the harness itself. Where seccomp
//! is not available the tests print why and pass; under `CI=true` they fail instead, because
//! the CI runners support it.
#![cfg(all(feature = "worker-confine", target_os = "linux"))]

use std::ffi::c_long;
use std::os::fd::{AsFd, FromRawFd, OwnedFd};
use std::os::unix::process::ExitStatusExt;
use std::process::{Command, Stdio};

use ptr_fabric::{
    install_worker_seccomp, install_worker_seccomp_with, Channel, ConfineError, ConfinementProbe,
    DenyAction, ShmExpectation, ShmExport, ShmMapping,
};
use ptr_protocol::content_digest;

const MODE_VAR: &str = "PTR_FABRIC_CONFINE_CHILD";
const SOCKET_VAR: &str = "PTR_FABRIC_CONFINE_FD";
const EXIT_SKIP: i32 = 77;

/// Held while a child is spawned: the descriptor handed to one child is not close-on-exec and
/// must not be inherited by a child another test spawns at the same time.
static SPAWN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The payload of the descriptor the parent sends to the confined child.
fn payload() -> Vec<u8> {
    (0..20_000u32).map(|index| (index % 249) as u8).collect()
}

fn skip_or_fail(reason: &str) {
    assert!(
        std::env::var("CI").as_deref() != Ok("true"),
        "seccomp is required under CI=true but unavailable: {reason}"
    );
    eprintln!("SKIPPED: {reason}");
}

// --- the probe ----------------------------------------------------------------------------

fn status_field(name: &str) -> String {
    let status = std::fs::read_to_string("/proc/self/status").expect("status");
    status
        .lines()
        .find(|line| line.starts_with(name))
        .unwrap_or_default()
        .to_owned()
}

fn process_state() -> (String, String, String, std::io::Result<std::path::PathBuf>) {
    (
        status_field("Seccomp:"),
        status_field("NoNewPrivs:"),
        status_field("Uid:"),
        std::fs::read_link("/proc/self/ns/net"),
    )
}

#[test]
fn detecting_changes_nothing_about_the_calling_process() {
    let before = process_state();
    let probe = ConfinementProbe::detect();
    let again = ConfinementProbe::detect();
    let after = process_state();
    assert_eq!(before.0, after.0, "seccomp mode");
    assert_eq!(before.1, after.1, "no_new_privs");
    assert_eq!(before.2, after.2, "user ids");
    assert_eq!(
        before.3.as_ref().ok(),
        after.3.as_ref().ok(),
        "network namespace"
    );
    assert_eq!(probe, again, "the probe is stable");
    // The tests below still fork and run: the process is as it was.
    eprintln!("probe: {probe:#?}");
}

#[test]
fn every_missing_mechanism_comes_with_a_reason() {
    let probe = ConfinementProbe::detect();
    for availability in [
        &probe.seccomp,
        &probe.own_uid,
        &probe.cgroup_v2,
        &probe.net_namespace,
    ] {
        if let ptr_fabric::Availability::Unavailable(reason) = availability {
            assert!(!reason.is_empty());
        }
    }
    let confinement = probe.best_confinement();
    let applied = probe.available();
    assert_eq!(
        matches!(confinement, ptr_fabric::Confinement::Full),
        applied.seccomp && applied.own_uid && applied.cgroup_v2 && applied.net_namespace
    );
}

// --- the filter, in a child ----------------------------------------------------------------

/// A system call made directly, so the C library cannot change what the kernel sees.
fn raw(number: c_long, args: [usize; 6]) -> Result<c_long, i32> {
    // SAFETY: every call in the table below either carries null or zero arguments, which the
    // kernel rejects or which the filter denies before the kernel looks at them; none can
    // touch memory of this process.
    let result =
        unsafe { libc::syscall(number, args[0], args[1], args[2], args[3], args[4], args[5]) };
    if result == -1 {
        Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
    } else {
        Ok(result)
    }
}

fn fail(message: &str) -> ! {
    eprintln!("CHILD FAIL: {message}");
    std::process::exit(1);
}

const NONE: [usize; 6] = [0; 6];

/// Calls the allowlist must deny with `EPERM`: everything doc 42 names, plus the ways to make
/// a process or leave a namespace.
fn denied_calls() -> Vec<(&'static str, c_long, [usize; 6])> {
    let thread_flags = (libc::CLONE_VM
        | libc::CLONE_FS
        | libc::CLONE_FILES
        | libc::CLONE_SIGHAND
        | libc::CLONE_THREAD) as usize;
    let mut calls = vec![
        ("socket", libc::SYS_socket, [1, 1, 0, 0, 0, 0]),
        ("socketpair", libc::SYS_socketpair, [1, 1, 0, 0, 0, 0]),
        ("connect", libc::SYS_connect, NONE),
        ("bind", libc::SYS_bind, NONE),
        ("listen", libc::SYS_listen, NONE),
        ("accept", libc::SYS_accept, NONE),
        ("accept4", libc::SYS_accept4, NONE),
        (
            "io_uring_setup",
            libc::SYS_io_uring_setup,
            [1, 0, 0, 0, 0, 0],
        ),
        ("io_uring_enter", libc::SYS_io_uring_enter, NONE),
        ("io_uring_register", libc::SYS_io_uring_register, NONE),
        ("pidfd_getfd", libc::SYS_pidfd_getfd, NONE),
        ("pidfd_open", libc::SYS_pidfd_open, [1, 0, 0, 0, 0, 0]),
        ("ptrace", libc::SYS_ptrace, NONE),
        ("process_vm_readv", libc::SYS_process_vm_readv, NONE),
        ("process_vm_writev", libc::SYS_process_vm_writev, NONE),
        ("execve", libc::SYS_execve, NONE),
        ("execveat", libc::SYS_execveat, NONE),
        ("openat", libc::SYS_openat, NONE),
        ("kill", libc::SYS_kill, NONE),
        ("unshare", libc::SYS_unshare, NONE),
        ("setns", libc::SYS_setns, NONE),
        // `clone` that makes a process: SIGCHLD as the exit signal, no CLONE_THREAD.
        (
            "clone (process)",
            libc::SYS_clone,
            [libc::SIGCHLD as usize, 0, 0, 0, 0, 0],
        ),
        // `clone` that shares memory but is not a thread.
        (
            "clone (CLONE_VM only)",
            libc::SYS_clone,
            [libc::CLONE_VM as usize, 0, 0, 0, 0, 0],
        ),
        // A thread that also asks for a new network namespace.
        (
            "clone (thread with CLONE_NEWNET)",
            libc::SYS_clone,
            [thread_flags | libc::CLONE_NEWNET as usize, 0, 0, 0, 0, 0],
        ),
        // `ioctl` other than the three allowed requests.
        ("ioctl (TIOCSTI)", libc::SYS_ioctl, [0, 0x5412, 0, 0, 0, 0]),
        // `prlimit64` that would set a limit (non-null new limit) or reach another process.
        ("prlimit64 (set)", libc::SYS_prlimit64, [0, 7, 8, 0, 0, 0]),
        (
            "prlimit64 (other pid)",
            libc::SYS_prlimit64,
            [1, 7, 0, 0, 0, 0],
        ),
        // `tgkill` of another process: only the worker's own thread group may be signalled.
        ("tgkill (pid 1)", libc::SYS_tgkill, [1, 1, 0, 0, 0, 0]),
        // `fcntl` commands that are not needed: F_SETOWN, F_SETSIG, F_SETOWN_EX.
        ("fcntl (F_SETOWN)", libc::SYS_fcntl, [0, 8, 1, 0, 0, 0]),
        ("fcntl (F_SETSIG)", libc::SYS_fcntl, [0, 10, 1, 0, 0, 0]),
        ("fcntl (F_SETOWN_EX)", libc::SYS_fcntl, [0, 15, 0, 0, 0, 0]),
    ];
    #[cfg(target_arch = "x86_64")]
    {
        calls.push(("fork", libc::SYS_fork, NONE));
        calls.push(("vfork", libc::SYS_vfork, NONE));
    }
    calls
}

fn check_denials() {
    for (name, number, args) in denied_calls() {
        match raw(number, args) {
            Err(libc::EPERM) => {}
            Err(errno) => fail(&format!("{name}: errno {errno}, wanted EPERM")),
            Ok(value) => fail(&format!("{name} was allowed and returned {value}")),
        }
    }
    // `clone3` is answered ENOSYS so the C library falls back to `clone`.
    match raw(libc::SYS_clone3, NONE) {
        Err(libc::ENOSYS) => {}
        other => fail(&format!("clone3: {other:?}, wanted ENOSYS")),
    }
    // Through the C library: what an exploit in the worker would actually call.
    // SAFETY: `socket` and `fork` take plain integers; both must fail under the filter. If the
    // filter were wrong and `fork` succeeded, the grandchild exits at once.
    unsafe {
        if libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) != -1
            || std::io::Error::last_os_error().raw_os_error() != Some(libc::EPERM)
        {
            fail("socket() was not refused with EPERM");
        }
        let pid = libc::fork();
        if pid == 0 {
            libc::_exit(99);
        }
        if pid != -1 || std::io::Error::last_os_error().raw_os_error() != Some(libc::EPERM) {
            fail("fork() was not refused with EPERM");
        }
    }
}

fn check_compute(pipe: [i32; 2]) {
    // A thread still works (glibc tries clone3, gets ENOSYS, falls back to the checked clone).
    let sum = std::thread::spawn(|| (1..=100u64).sum::<u64>())
        .join()
        .unwrap_or_else(|_| fail("the thread panicked"));
    if sum != 5050 {
        fail("the thread computed the wrong value");
    }
    // tgkill of the own thread group is allowed (signal 0 only checks), and so is the fcntl
    // that reads a descriptor's flags.
    // SAFETY: `getpid` and `gettid` take no arguments.
    let (pid, tid) = unsafe { (libc::getpid(), libc::gettid()) };
    if raw(libc::SYS_tgkill, [pid as usize, tid as usize, 0, 0, 0, 0]) != Ok(0) {
        fail("tgkill of the own thread group was refused");
    }
    if raw(
        libc::SYS_fcntl,
        [pipe[0] as usize, libc::F_GETFD as usize, 0, 0, 0, 0],
    )
    .is_err()
    {
        fail("fcntl(F_GETFD) was refused");
    }
    // An inherited pipe works.
    let message = b"through the pipe";
    // SAFETY: `pipe` holds the two ends created before the filter; the buffers are valid for
    // the lengths passed.
    let (written, read, received) = unsafe {
        let written = libc::write(pipe[1], message.as_ptr().cast(), message.len());
        let mut buffer = [0u8; 32];
        let read = libc::read(pipe[0], buffer.as_mut_ptr().cast(), buffer.len());
        (written, read, buffer)
    };
    if written != message.len() as isize
        || read != message.len() as isize
        || &received[..message.len()] != message
    {
        fail("the inherited pipe did not carry the bytes");
    }
    // A memfd round trip: create, seal, map, verify, unmap.
    let bytes = payload();
    let export =
        ShmExport::create(&bytes).unwrap_or_else(|error| fail(&format!("export: {error}")));
    let fd = export
        .fd()
        .try_clone_to_owned()
        .unwrap_or_else(|error| fail(&format!("dup: {error}")));
    let expected = ShmExpectation {
        byte_len: bytes.len() as u64,
        offset: 0,
        content_digest: content_digest(&bytes),
    };
    match ShmMapping::map(fd, &expected) {
        Ok(mapping) if mapping.as_bytes() == bytes.as_slice() => {}
        Ok(_) => fail("the memfd mapped other bytes"),
        Err(error) => fail(&format!("memfd map under the filter: {error}")),
    }
}

/// Over the inherited control socket: receive a frame with a sealed memfd, map and verify it,
/// answer. Every call is `recvmsg`, `sendmsg`, `fstat`, `fcntl`, `mmap`, `munmap` and `close`.
fn check_control_socket(channel: &Channel) {
    let frame = channel
        .recv()
        .unwrap_or_else(|error| fail(&format!("recv on the control socket: {error}")));
    if frame.body != b"export" || frame.descriptors.len() != 1 {
        fail("the frame from the supervisor is not the expected one");
    }
    let bytes = payload();
    let expected = ShmExpectation {
        byte_len: bytes.len() as u64,
        offset: 0,
        content_digest: content_digest(&bytes),
    };
    let fd = frame
        .descriptors
        .into_iter()
        .next()
        .unwrap_or_else(|| fail("no descriptor"));
    match ShmMapping::map(fd, &expected) {
        Ok(mapping) if mapping.as_bytes() == bytes.as_slice() => {}
        Ok(_) => fail("the received region held other bytes"),
        Err(error) => fail(&format!("map of the received region: {error}")),
    }
    channel
        .send(b"verified", &[])
        .unwrap_or_else(|error| fail(&format!("send on the control socket: {error}")));
}

/// The child. Does nothing when run as an ordinary test.
#[test]
fn confine_child_helper() {
    let Ok(mode) = std::env::var(MODE_VAR) else {
        return;
    };
    // Everything that needs other system calls happens before the filter.
    let mut pipe = [0i32; 2];
    // SAFETY: `pipe` is a valid two-element array.
    if unsafe { libc::pipe(pipe.as_mut_ptr()) } != 0 {
        fail("pipe");
    }
    let channel = std::env::var(SOCKET_VAR).ok().map(|number| {
        let number: i32 = number.parse().unwrap_or_else(|_| fail("bad socket number"));
        // SAFETY: the parent passed this descriptor to us and nothing else owns it.
        let fd = unsafe { OwnedFd::from_raw_fd(number) };
        Channel::from_fd(fd).unwrap_or_else(|error| fail(&format!("control socket: {error}")))
    });

    if mode == "kill" {
        match install_worker_seccomp_with(DenyAction::KillProcess) {
            Ok(()) => {}
            Err(ConfineError::Unsupported(why)) => {
                eprintln!("UNSUPPORTED: {why}");
                std::process::exit(EXIT_SKIP);
            }
            Err(error) => fail(&format!("install: {error}")),
        }
        // SAFETY: plain integers; the filter must kill the process before this returns.
        unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
        eprintln!("CHILD FAIL: survived a forbidden call under KillProcess");
        std::process::exit(2);
    }

    match install_worker_seccomp() {
        Ok(()) => {}
        Err(ConfineError::Unsupported(why)) => {
            eprintln!("UNSUPPORTED: {why}");
            std::process::exit(EXIT_SKIP);
        }
        Err(error) => fail(&format!("install: {error}")),
    }
    check_denials();
    check_compute(pipe);
    if let Some(channel) = &channel {
        check_control_socket(channel);
    }
    println!("CHILD OK");
    std::process::exit(0);
}

fn child_command(mode: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().expect("test binary"));
    command
        .args([
            "--exact",
            "confine_child_helper",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(MODE_VAR, mode)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn seccomp_available() -> bool {
    let probe = ConfinementProbe::detect();
    match &probe.seccomp {
        ptr_fabric::Availability::Available => true,
        ptr_fabric::Availability::Unavailable(reason) => {
            skip_or_fail(reason);
            false
        }
    }
}

#[test]
fn a_filtered_child_is_denied_what_doc_42_names_and_still_computes() {
    if !seccomp_available() {
        return;
    }
    let (parent, child_end) = Channel::pair().expect("socket pair");
    // The child's end, without close-on-exec, so the child inherits it. The spawns of the two
    // child tests are serialized so this descriptor cannot leak into the other child.
    let child = {
        let _spawning = SPAWN
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let inheritable = rustix::io::dup(child_end.as_fd()).expect("dup");
        let mut command = child_command("full");
        command.env(
            SOCKET_VAR,
            std::os::fd::AsRawFd::as_raw_fd(&inheritable).to_string(),
        );
        let child = command.spawn().expect("spawn the child");
        drop(inheritable);
        child
    };
    // Only the child holds the far end now, so its death shows as `PeerClosed`.
    drop(child_end);

    let bytes = payload();
    let export = ShmExport::create(&bytes).expect("export");
    parent
        .send(b"export", &[export.fd()])
        .expect("send to the child");
    let reply = parent.recv();
    let output = child.wait_with_output().expect("child output");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.code() == Some(EXIT_SKIP) {
        skip_or_fail(&format!("the child could not install the filter: {stderr}"));
        return;
    }
    assert!(
        output.status.success(),
        "child failed: {:?}\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
        output.status
    );
    assert!(stdout.contains("CHILD OK"), "{stdout}");
    assert_eq!(reply.expect("reply from the child").body, b"verified");
}

#[test]
fn kill_mode_ends_the_process_on_the_first_forbidden_call() {
    if !seccomp_available() {
        return;
    }
    let output = {
        let _spawning = SPAWN
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        child_command("kill").output().expect("child output")
    };
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.code() == Some(EXIT_SKIP) {
        skip_or_fail(&format!("the child could not install the filter: {stderr}"));
        return;
    }
    assert_eq!(
        output.status.signal(),
        Some(libc::SIGSYS),
        "status {:?}\n{stderr}",
        output.status
    );
}
