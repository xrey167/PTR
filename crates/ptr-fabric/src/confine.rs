//! Descriptor confinement for a Pod worker (doc 42 stage S1, scope guard).
//!
//! Rule 3b releases a shared buffer only when every isolation domain that could hold a
//! reference to it has been reaped. That is sound only if a worker cannot get a descriptor out
//! of its domain except through the supervisor. S1 therefore asks four things of a worker, and
//! this module has the parts that do not need a supervisor:
//!
//! 1. [`ConfinementProbe::detect`]: which of the four this host can provide, found without
//!    changing the calling process;
//! 2. [`Confinement`]: what was actually applied to a worker (`Full` only when all four);
//! 3. [`carrier_permitted`]: which carriers a worker with a given confinement may use;
//! 4. [`install_worker_seccomp`]: the seccomp-BPF allowlist, the one mechanism that is applied
//!    by the worker itself. Applying a separate user id, a cgroup-v2 domain and a network
//!    namespace belongs to the supervisor that spawns the worker and is not in this module.
//!
//! # Mechanisms
//!
//! * **seccomp**: the kernel supports `SECCOMP_MODE_FILTER` and the architecture is x86_64 or
//!   aarch64 (little endian), the two the filter is written for.
//! * **own user id**: the process holds `CAP_SETUID` and `CAP_SETGID` (root, or root with those
//!   two capabilities kept), so it can give the worker a user id of its own. No other process
//!   can then open that worker's `/proc/<pid>/fd` entries.
//! * **cgroup v2**: a `cgroup2` mount (found in `/proc/self/mountinfo`, so it may be
//!   `/sys/fs/cgroup` or the hybrid `/sys/fs/cgroup/unified`) whose directory for this
//!   process's own group, and its `cgroup.procs`, are writable: the supervisor can create a
//!   child group and move the worker into it.
//! * **network namespace**: `unshare(CLONE_NEWNET)` works, as root or through an unprivileged
//!   user namespace. Probed in a short-lived forked child that only calls `unshare` and
//!   `_exit`; the calling process is not changed.
//!
//! # The allowlist
//!
//! The filter is a default-deny **allowlist**: a denylist is not enough because `io_uring` can
//! create and connect sockets and send descriptors without any of the filtered system calls.
//! Everything not listed is denied, so the system calls doc 42 names (`io_uring_setup`,
//! `io_uring_enter`, `io_uring_register`, `socket`, `socketpair`, `connect`, `bind`, `listen`,
//! `accept`, `accept4`, `pidfd_getfd`, `pidfd_open`, `ptrace`, `process_vm_readv`,
//! `process_vm_writev`, `fork`, `vfork`, `execve`, `execveat`) are denied by omission, and
//! `clone3` is answered `ENOSYS` so that glibc falls back to `clone`, whose flags are checked.
//! [`worker_allowlist`] returns the exact list; the tests assert that no forbidden call is on
//! it.
//!
//! The architecture is checked first (a wrong one, for example the 32-bit compat ABI on an
//! x86_64 kernel, kills the process), then `PR_SET_NO_NEW_PRIVS` is set, then the filter is
//! installed on every thread of the process (`SECCOMP_FILTER_FLAG_TSYNC`).
//!
//! `clone` is allowed only for threads: its flags must include `CLONE_THREAD` and none of the
//! `CLONE_NEW*` bits. `prlimit64` only reads the calling process's limits (`pid` 0, no new
//! limit). `ioctl` only `FIONBIO`, `FIOCLEX` and `FIONCLEX`. `fcntl` only `F_GETFD`,
//! `F_SETFD`, `F_GETFL`, `F_SETFL`, `F_DUPFD_CLOEXEC`, `F_ADD_SEALS` and `F_GET_SEALS`
//! (`F_SETOWN`, `F_SETOWN_EX` and `F_SETSIG` are denied). `tgkill` only for the thread group
//! of the installing process, read with `getpid` at install time, so the filter must be
//! installed in the worker process after its last `fork` or `exec`; the worker cannot signal
//! any other process.
//!
//! Not restricted, and therefore a known limit: `mmap` and `mprotect` accept any protection
//! (no W^X), and `newfstatat` and `statx` accept any path, so a worker can read file metadata
//! (never open a file, because `open` and `openat` are not allowed).
//!
//! The default deny action is `SECCOMP_RET_ERRNO(EPERM)`: a worker that probes a forbidden
//! call gets a refusal it can report and the tests can observe, and no allowlist entry is
//! reachable through it. [`DenyAction::KillProcess`] is available for a supervisor that prefers
//! the worker to die on the first forbidden call.

use std::ffi::{c_int, c_long, c_ulong, CString};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use ptr_protocol::Carrier;

/// One of the four things S1 asks of a worker.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Mechanism {
    /// A seccomp-BPF allowlist.
    Seccomp,
    /// A user id of the worker's own.
    OwnUid,
    /// A cgroup-v2 isolation domain.
    CgroupV2,
    /// A network namespace of its own.
    NetNamespace,
}

impl Mechanism {
    /// Every mechanism, in the order of the doc.
    pub const ALL: [Mechanism; 4] = [
        Mechanism::Seccomp,
        Mechanism::OwnUid,
        Mechanism::CgroupV2,
        Mechanism::NetNamespace,
    ];
}

/// Whether this host can provide a mechanism, and if not, why.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Availability {
    Available,
    Unavailable(String),
}

impl Availability {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

/// Which mechanisms are applied to a worker (or, from a probe, could be).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AppliedMechanisms {
    pub seccomp: bool,
    pub own_uid: bool,
    pub cgroup_v2: bool,
    pub net_namespace: bool,
}

impl AppliedMechanisms {
    /// All four.
    pub const ALL: Self = Self {
        seccomp: true,
        own_uid: true,
        cgroup_v2: true,
        net_namespace: true,
    };

    fn has(&self, mechanism: Mechanism) -> bool {
        match mechanism {
            Mechanism::Seccomp => self.seccomp,
            Mechanism::OwnUid => self.own_uid,
            Mechanism::CgroupV2 => self.cgroup_v2,
            Mechanism::NetNamespace => self.net_namespace,
        }
    }
}

/// What this host can provide, per mechanism.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfinementProbe {
    pub seccomp: Availability,
    pub own_uid: Availability,
    pub cgroup_v2: Availability,
    pub net_namespace: Availability,
}

impl ConfinementProbe {
    /// Probe this host. The calling process is not changed: seccomp is asked with a null
    /// filter (the kernel answers `EFAULT` where filters are supported and does not install
    /// anything), capabilities and the cgroup are read from `/proc` and checked for write
    /// access, and the network namespace is tried in a forked child that exits at once.
    pub fn detect() -> Self {
        Self {
            seccomp: probe_seccomp(),
            own_uid: probe_own_uid(),
            cgroup_v2: probe_cgroup_v2(),
            net_namespace: probe_net_namespace(),
        }
    }

    /// The mechanisms the host can provide.
    pub fn available(&self) -> AppliedMechanisms {
        AppliedMechanisms {
            seccomp: self.seccomp.is_available(),
            own_uid: self.own_uid.is_available(),
            cgroup_v2: self.cgroup_v2.is_available(),
            net_namespace: self.net_namespace.is_available(),
        }
    }

    /// The best [`Confinement`] a supervisor on this host could give a worker.
    pub fn best_confinement(&self) -> Confinement {
        Confinement::from_applied(self.available())
    }
}

/// How well a worker is confined.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Confinement {
    /// All four mechanisms are applied.
    Full,
    /// Some but not all; `missing` names the rest.
    Partial { missing: Vec<Mechanism> },
    /// None is applied.
    None,
}

impl Confinement {
    /// `Full` only when every mechanism is applied.
    pub fn from_applied(applied: AppliedMechanisms) -> Self {
        let missing: Vec<Mechanism> = Mechanism::ALL
            .into_iter()
            .filter(|mechanism| !applied.has(*mechanism))
            .collect();
        match missing.len() {
            0 => Self::Full,
            count if count == Mechanism::ALL.len() => Self::None,
            _ => Self::Partial { missing },
        }
    }

    /// The mechanisms that are not applied.
    pub fn missing(&self) -> Vec<Mechanism> {
        match self {
            Self::Full => Vec::new(),
            Self::Partial { missing } => missing.clone(),
            Self::None => Mechanism::ALL.to_vec(),
        }
    }
}

/// May a worker with this confinement exchange tensors by this carrier with another process?
///
/// * `Bytes`: always. A copy shares nothing.
/// * `HostShm`, `CudaIpc`, `DevicePeer`, `PinnedHost`: only under [`Confinement::Full`]. They
///   are the shared carriers of rule 3b, released only on verified teardown of the isolation
///   domain, which is sound only if a descriptor cannot leave the domain.
/// * `InProcess`: never across processes; it has no meaning outside one address space.
/// * `Remote`: always. The frame is fetched from another host by the supervisor's own
///   transport and reaches the worker as `Bytes`; no descriptor or memory of this host is
///   shared with the worker, so its confinement does not matter to this carrier.
/// * `NvmeFile`: never for a worker. Reading a file chunk needs `open` or a path, which the
///   confined worker's allowlist does not have and an unconfined worker must not be given; the
///   owning process reads the chunk and hands the worker `Bytes`.
pub fn carrier_permitted(carrier: Carrier, confinement: &Confinement) -> bool {
    match carrier {
        Carrier::Bytes | Carrier::Remote => true,
        Carrier::HostShm | Carrier::CudaIpc | Carrier::DevicePeer | Carrier::PinnedHost => {
            matches!(confinement, Confinement::Full)
        }
        Carrier::InProcess | Carrier::NvmeFile => false,
    }
}

// --- probes -------------------------------------------------------------------------------

fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

fn probe_seccomp() -> Availability {
    if !filter::supported_arch() {
        return Availability::Unavailable(format!(
            "no seccomp allowlist for architecture {}",
            std::env::consts::ARCH
        ));
    }
    // SAFETY: `prctl` with `PR_SET_SECCOMP` and a null filter pointer. The kernel copies the
    // filter before it changes anything, so a null pointer fails with `EFAULT` and installs
    // nothing; the arguments are plain integers.
    let result = unsafe {
        libc::prctl(
            libc::PR_SET_SECCOMP,
            libc::SECCOMP_MODE_FILTER as c_ulong,
            0 as c_ulong,
            0 as c_ulong,
            0 as c_ulong,
        )
    };
    let errno = last_errno();
    if result == -1 && errno == libc::EFAULT {
        Availability::Available
    } else if result == -1 && errno == libc::EINVAL {
        Availability::Unavailable("the kernel has no seccomp filter mode".to_owned())
    } else {
        Availability::Unavailable(format!("seccomp probe: result {result}, errno {errno}"))
    }
}

/// Bit positions in the capability sets.
const CAP_SETGID: u32 = 6;
const CAP_SETUID: u32 = 7;

/// The effective capability mask from the text of `/proc/self/status`.
fn effective_capabilities(status: &str) -> Option<u64> {
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix("CapEff:"))?;
    u64::from_str_radix(line.trim(), 16).ok()
}

fn probe_own_uid() -> Availability {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return Availability::Unavailable("cannot read /proc/self/status".to_owned());
    };
    let Some(caps) = effective_capabilities(&status) else {
        return Availability::Unavailable("no CapEff line in /proc/self/status".to_owned());
    };
    let needed = (1u64 << CAP_SETUID) | (1u64 << CAP_SETGID);
    if caps & needed == needed {
        Availability::Available
    } else {
        Availability::Unavailable("the process lacks CAP_SETUID and CAP_SETGID".to_owned())
    }
}

/// Undo the octal escapes (`\040` for a space) of a `/proc/self/mountinfo` path.
fn unescape_mount_path(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let octal = bytes.get(index + 1..index + 4).filter(|digits| {
            bytes[index] == b'\\' && digits.iter().all(|digit| (b'0'..=b'7').contains(digit))
        });
        if let Some(digits) = octal {
            let value = digits
                .iter()
                .fold(0u32, |sum, digit| sum * 8 + u32::from(digit - b'0'));
            out.push(u8::try_from(value & 0xff).unwrap_or(b'?'));
            index += 4;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The first `cgroup2` mount in the text of `/proc/self/mountinfo`: its root inside the
/// filesystem and its mount point.
fn cgroup2_mount(mountinfo: &str) -> Option<(String, String)> {
    mountinfo.lines().find_map(|line| {
        let (head, tail) = line.split_once(" - ")?;
        if tail.split_whitespace().next()? != "cgroup2" {
            return None;
        }
        let mut fields = head.split_whitespace();
        let root = unescape_mount_path(fields.nth(3)?);
        let mount_point = unescape_mount_path(fields.next()?);
        Some((root, mount_point))
    })
}

/// This process's group in the unified hierarchy, from the text of `/proc/self/cgroup`.
fn own_cgroup_path(proc_cgroup: &str) -> Option<String> {
    proc_cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(|path| path.trim().to_owned())
}

/// The directory of the process's own group under the mount.
fn cgroup_directory(root: &str, mount_point: &str, own_path: &str) -> PathBuf {
    let relative = own_path
        .strip_prefix(root.trim_end_matches('/'))
        .unwrap_or(own_path)
        .trim_start_matches('/');
    Path::new(mount_point).join(relative)
}

fn writable(path: &Path) -> bool {
    let Ok(path) = CString::new(path.as_os_str().as_encoded_bytes()) else {
        return false;
    };
    // SAFETY: `path` is a valid NUL-terminated string that outlives the call, and the flags are
    // plain integers; `AT_EACCESS` checks with the effective ids.
    unsafe { libc::faccessat(libc::AT_FDCWD, path.as_ptr(), libc::W_OK, libc::AT_EACCESS) == 0 }
}

fn probe_cgroup_v2() -> Availability {
    let Ok(mountinfo) = fs::read_to_string("/proc/self/mountinfo") else {
        return Availability::Unavailable("cannot read /proc/self/mountinfo".to_owned());
    };
    let Some((root, mount_point)) = cgroup2_mount(&mountinfo) else {
        return Availability::Unavailable("no cgroup2 mount".to_owned());
    };
    let Some(own) = fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|text| own_cgroup_path(&text))
    else {
        return Availability::Unavailable("no unified cgroup for this process".to_owned());
    };
    let directory = cgroup_directory(&root, &mount_point, &own);
    if writable(&directory) && writable(&directory.join("cgroup.procs")) {
        Availability::Available
    } else {
        Availability::Unavailable(format!(
            "{} is not writable: no delegated cgroup v2",
            directory.display()
        ))
    }
}

fn probe_net_namespace() -> Availability {
    // SAFETY: `fork` in a possibly multithreaded process; the child below calls only
    // `unshare` and `_exit` (both async-signal-safe) and never allocates or returns.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Availability::Unavailable(format!("fork failed, errno {}", last_errno()));
    }
    if pid == 0 {
        // SAFETY: as above; `unshare` takes plain flags and `_exit` ends the child.
        unsafe {
            let mut result = libc::unshare(libc::CLONE_NEWNET);
            if result != 0 {
                result = libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNET);
            }
            let code = if result == 0 {
                0
            } else {
                (last_errno() & 0x7f).max(1)
            };
            libc::_exit(code);
        }
    }
    let mut status: c_int = 0;
    loop {
        // SAFETY: `pid` is the child just forked and `status` is a valid out pointer.
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        if waited == pid {
            break;
        }
        if waited < 0 && last_errno() != libc::EINTR {
            return Availability::Unavailable(format!("waitpid failed, errno {}", last_errno()));
        }
    }
    if libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0 {
        Availability::Available
    } else if libc::WIFEXITED(status) {
        Availability::Unavailable(format!(
            "unshare(CLONE_NEWNET) failed, errno {}",
            libc::WEXITSTATUS(status)
        ))
    } else {
        Availability::Unavailable("the probe child did not exit normally".to_owned())
    }
}

/// The directory of this process's own cgroup-v2 group, where a supervisor creates the child
/// group of a worker. `None` when there is no `cgroup2` mount or no unified group.
#[cfg(feature = "worker-supervisor")]
pub(crate) fn own_cgroup_directory() -> Option<PathBuf> {
    let mountinfo = fs::read_to_string("/proc/self/mountinfo").ok()?;
    let (root, mount_point) = cgroup2_mount(&mountinfo)?;
    let own = own_cgroup_path(&fs::read_to_string("/proc/self/cgroup").ok()?)?;
    Some(cgroup_directory(&root, &mount_point, &own))
}

/// One call a confined worker must not be able to make, and what happened when it tried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForbiddenCall {
    pub name: &'static str,
    /// The `errno` of the refused call, or 0 when the call succeeded (the worker is not
    /// confined).
    pub errno: i32,
}

/// Try the calls the allowlist must refuse and report each outcome: `socket`, `socketpair`,
/// `fork`, `open`, `io_uring_setup`, `process_vm_readv` and `pidfd_open`. A worker runs this after
/// [`install_worker_seccomp`] to show that the filter is in force (every `errno` is `EPERM`).
/// In an unconfined process the calls succeed or fail for ordinary reasons and are undone at
/// once (a socket is closed, a forked child exits immediately and is reaped), so calling it
/// there is harmless but shows nothing.
pub fn probe_forbidden_calls() -> Vec<ForbiddenCall> {
    let mut report = Vec::new();
    let errno_of = |result: c_long| if result >= 0 { 0 } else { last_errno() };
    // SAFETY (all blocks below): each call takes plain integers or a valid NUL-terminated
    // path, and every descriptor or child it creates is closed or reaped before the next call.
    let socket = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
    let socket_errno = errno_of(c_long::from(socket));
    if socket >= 0 {
        unsafe { libc::close(socket) };
    }
    report.push(ForbiddenCall {
        name: "socket",
        errno: socket_errno,
    });
    let mut pair = [0 as c_int; 2];
    let result =
        unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, pair.as_mut_ptr()) };
    let pair_errno = errno_of(c_long::from(result));
    if result == 0 {
        unsafe {
            libc::close(pair[0]);
            libc::close(pair[1]);
        }
    }
    report.push(ForbiddenCall {
        name: "socketpair",
        errno: pair_errno,
    });
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        // The child of a successful fork: leave at once.
        unsafe { libc::_exit(0) };
    }
    let fork_errno = errno_of(c_long::from(pid));
    if pid > 0 {
        let mut status: c_int = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };
    }
    report.push(ForbiddenCall {
        name: "fork",
        errno: fork_errno,
    });
    let path = c"/proc/self/status";
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    let open_errno = errno_of(c_long::from(fd));
    if fd >= 0 {
        unsafe { libc::close(fd) };
    }
    report.push(ForbiddenCall {
        name: "open",
        errno: open_errno,
    });
    let ring = unsafe { libc::syscall(libc::SYS_io_uring_setup, 1 as c_ulong, 0 as c_ulong) };
    let ring_errno = errno_of(ring);
    if ring >= 0 {
        unsafe { libc::close(ring as c_int) };
    }
    report.push(ForbiddenCall {
        name: "io_uring_setup",
        errno: ring_errno,
    });
    // A read of zero bytes from this very process: harmless where the call is allowed.
    let empty = libc::iovec {
        iov_base: std::ptr::null_mut(),
        iov_len: 0,
    };
    let vm = unsafe {
        libc::syscall(
            libc::SYS_process_vm_readv,
            libc::getpid() as c_ulong,
            &empty as *const libc::iovec,
            1 as c_ulong,
            &empty as *const libc::iovec,
            1 as c_ulong,
            0 as c_ulong,
        )
    };
    report.push(ForbiddenCall {
        name: "process_vm_readv",
        errno: errno_of(vm),
    });
    let pidfd = unsafe { libc::syscall(libc::SYS_pidfd_open, 1 as c_ulong, 0 as c_ulong) };
    let pidfd_errno = errno_of(pidfd);
    if pidfd >= 0 {
        unsafe { libc::close(pidfd as c_int) };
    }
    report.push(ForbiddenCall {
        name: "pidfd_open",
        errno: pidfd_errno,
    });
    report
}

// --- the seccomp allowlist ---------------------------------------------------------------

/// What a denied system call does.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DenyAction {
    /// Fail with `EPERM` and let the process go on (the default: observable, and the worker
    /// can report it).
    #[default]
    Errno,
    /// Kill the whole process (`SECCOMP_RET_KILL_PROCESS`).
    KillProcess,
}

/// Why the filter was not installed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfineError {
    /// This architecture or kernel cannot run the filter; the caller falls back to `Bytes`.
    Unsupported(String),
    /// An operating-system call failed; the text is the OS error.
    Io(String),
}

impl fmt::Display for ConfineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(why) => write!(formatter, "confinement unsupported: {why}"),
            Self::Io(message) => write!(formatter, "os error: {message}"),
        }
    }
}

impl std::error::Error for ConfineError {}

/// One entry of the allowlist.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllowedSyscall {
    pub name: &'static str,
    pub number: c_long,
    /// The argument condition, for a call that is allowed only in one form.
    pub restriction: Option<&'static str>,
}

/// The exact allowlist of the worker filter, for documentation and tests.
pub fn worker_allowlist() -> Vec<AllowedSyscall> {
    filter::allowlist()
}

/// Install the worker allowlist with the default deny action (`EPERM`): `PR_SET_NO_NEW_PRIVS`,
/// then the filter on every thread of the process. Irreversible. Call it in the worker after
/// everything that needs other system calls (opening the control socket, mapping the first
/// regions) and before it reads any request.
pub fn install_worker_seccomp() -> Result<(), ConfineError> {
    install_worker_seccomp_with(DenyAction::Errno)
}

/// Like [`install_worker_seccomp`] with a chosen deny action.
pub fn install_worker_seccomp_with(deny: DenyAction) -> Result<(), ConfineError> {
    filter::install(deny)
}

/// The filter. The numbers of each supported architecture are written out here (they come from
/// the kernel's tables, and a test compares the host's against `libc`), so the program for
/// both architectures is built and tested on any host; only the installing is host-specific.
mod filter {
    use super::{AllowedSyscall, ConfineError, DenyAction};
    use std::ffi::{c_long, c_ulong};

    /// One architecture: its `AUDIT_ARCH` value and its system call numbers.
    pub(super) struct Abi {
        pub(super) audit_arch: u32,
        /// Allowed in every form.
        pub(super) plain: &'static [(&'static str, u32)],
        pub(super) clone: u32,
        pub(super) clone3: u32,
        pub(super) ioctl: u32,
        pub(super) prlimit64: u32,
        pub(super) tgkill: u32,
        pub(super) fcntl: u32,
    }

    pub(super) const X86_64: Abi = Abi {
        audit_arch: 0xC000_003E,
        plain: &[
            ("read", 0),
            ("write", 1),
            ("readv", 19),
            ("writev", 20),
            ("close", 3),
            ("recvmsg", 47),
            ("sendmsg", 46),
            ("recvfrom", 45),
            ("sendto", 44),
            ("getsockopt", 55),
            ("ppoll", 271),
            ("epoll_create1", 291),
            ("epoll_ctl", 233),
            ("epoll_pwait", 281),
            ("fstat", 5),
            ("newfstatat", 262),
            ("statx", 332),
            ("ftruncate", 77),
            ("memfd_create", 319),
            ("mmap", 9),
            ("munmap", 11),
            ("mremap", 25),
            ("mprotect", 10),
            ("madvise", 28),
            ("brk", 12),
            ("futex", 202),
            ("set_robust_list", 273),
            ("rseq", 334),
            ("set_tid_address", 218),
            ("sched_yield", 24),
            ("sched_getaffinity", 204),
            ("clock_gettime", 228),
            ("clock_getres", 229),
            ("clock_nanosleep", 230),
            ("nanosleep", 35),
            ("gettimeofday", 96),
            ("rt_sigaction", 13),
            ("rt_sigprocmask", 14),
            ("rt_sigreturn", 15),
            ("sigaltstack", 131),
            ("restart_syscall", 219),
            ("getrandom", 318),
            ("getpid", 39),
            ("gettid", 186),
            ("exit", 60),
            ("exit_group", 231),
            // The legacy calls aarch64 does not have.
            ("poll", 7),
            ("epoll_wait", 232),
        ],
        clone: 56,
        clone3: 435,
        ioctl: 16,
        prlimit64: 302,
        tgkill: 234,
        fcntl: 72,
    };

    pub(super) const AARCH64: Abi = Abi {
        audit_arch: 0xC000_00B7,
        plain: &[
            ("read", 63),
            ("write", 64),
            ("readv", 65),
            ("writev", 66),
            ("close", 57),
            ("recvmsg", 212),
            ("sendmsg", 211),
            ("recvfrom", 207),
            ("sendto", 206),
            ("getsockopt", 209),
            ("ppoll", 73),
            ("epoll_create1", 20),
            ("epoll_ctl", 21),
            ("epoll_pwait", 22),
            ("fstat", 80),
            ("newfstatat", 79),
            ("statx", 291),
            ("ftruncate", 46),
            ("memfd_create", 279),
            ("mmap", 222),
            ("munmap", 215),
            ("mremap", 216),
            ("mprotect", 226),
            ("madvise", 233),
            ("brk", 214),
            ("futex", 98),
            ("set_robust_list", 99),
            ("rseq", 293),
            ("set_tid_address", 96),
            ("sched_yield", 124),
            ("sched_getaffinity", 123),
            ("clock_gettime", 113),
            ("clock_getres", 114),
            ("clock_nanosleep", 115),
            ("nanosleep", 101),
            ("gettimeofday", 169),
            ("rt_sigaction", 134),
            ("rt_sigprocmask", 135),
            ("rt_sigreturn", 139),
            ("sigaltstack", 132),
            ("restart_syscall", 128),
            ("getrandom", 278),
            ("getpid", 172),
            ("gettid", 178),
            ("exit", 93),
            ("exit_group", 94),
        ],
        clone: 220,
        clone3: 435,
        ioctl: 29,
        prlimit64: 261,
        tgkill: 131,
        fcntl: 25,
    };

    /// The ABI of this host, where one is supported (little endian only).
    pub(super) fn host_abi() -> Option<&'static Abi> {
        if cfg!(target_endian = "big") {
            None
        } else if cfg!(target_arch = "x86_64") {
            Some(&X86_64)
        } else if cfg!(target_arch = "aarch64") {
            Some(&AARCH64)
        } else {
            None
        }
    }

    pub(super) fn supported_arch() -> bool {
        host_abi().is_some()
    }

    const BPF_LD_W_ABS: u16 = 0x20;
    const BPF_ALU_AND_K: u16 = 0x54;
    const BPF_JEQ_K: u16 = 0x15;
    const BPF_RET_K: u16 = 0x06;

    /// `struct seccomp_data`: `nr` at 0, `arch` at 4, `args[6]` (u64) from 16.
    const OFFSET_NR: u32 = 0;
    const OFFSET_ARCH: u32 = 4;
    const fn offset_arg_low(index: u32) -> u32 {
        16 + 8 * index
    }
    const fn offset_arg_high(index: u32) -> u32 {
        16 + 8 * index + 4
    }

    pub(super) const RET_ALLOW: u32 = libc::SECCOMP_RET_ALLOW;
    pub(super) const RET_KILL_PROCESS: u32 = libc::SECCOMP_RET_KILL_PROCESS;
    pub(super) const fn ret_errno(errno: i32) -> u32 {
        libc::SECCOMP_RET_ERRNO | (errno as u32 & libc::SECCOMP_RET_DATA)
    }

    /// The clone flags that create a namespace; a thread never needs one.
    const CLONE_NEW_MASK: u32 = (libc::CLONE_NEWNS
        | libc::CLONE_NEWCGROUP
        | libc::CLONE_NEWUTS
        | libc::CLONE_NEWIPC
        | libc::CLONE_NEWUSER
        | libc::CLONE_NEWPID
        | libc::CLONE_NEWNET) as u32;
    const CLONE_THREAD: u32 = libc::CLONE_THREAD as u32;

    const FIONBIO: u32 = libc::FIONBIO as u32;
    const FIOCLEX: u32 = libc::FIOCLEX as u32;
    const FIONCLEX: u32 = libc::FIONCLEX as u32;

    /// The `fcntl` commands a worker needs: the descriptor and status flags, duplicating with
    /// close-on-exec, and the memfd seals. `F_SETOWN`, `F_SETOWN_EX`, `F_SETSIG` and the rest
    /// are denied.
    pub(super) const FCNTL_COMMANDS: [u32; 7] = [
        libc::F_GETFD as u32,
        libc::F_SETFD as u32,
        libc::F_GETFL as u32,
        libc::F_SETFL as u32,
        libc::F_DUPFD_CLOEXEC as u32,
        libc::F_ADD_SEALS as u32,
        libc::F_GET_SEALS as u32,
    ];

    pub(super) fn allowlist() -> Vec<AllowedSyscall> {
        let Some(abi) = host_abi() else {
            return Vec::new();
        };
        let mut list: Vec<AllowedSyscall> = abi
            .plain
            .iter()
            .map(|&(name, number)| AllowedSyscall {
                name,
                number: c_long::from(number),
                restriction: None,
            })
            .collect();
        let guarded = [
            (
                "clone",
                abi.clone,
                "flags include CLONE_THREAD and no CLONE_NEW* bit",
            ),
            (
                "ioctl",
                abi.ioctl,
                "request is FIONBIO, FIOCLEX or FIONCLEX",
            ),
            (
                "prlimit64",
                abi.prlimit64,
                "pid 0 and no new limit (read only)",
            ),
            (
                "tgkill",
                abi.tgkill,
                "thread group is the installing process",
            ),
            (
                "fcntl",
                abi.fcntl,
                "command is F_GETFD, F_SETFD, F_GETFL, F_SETFL, F_DUPFD_CLOEXEC, F_ADD_SEALS or F_GET_SEALS",
            ),
        ];
        list.extend(guarded.map(|(name, number, restriction)| AllowedSyscall {
            name,
            number: c_long::from(number),
            restriction: Some(restriction),
        }));
        list
    }

    #[derive(Clone, Copy)]
    enum Label {
        /// The next instruction.
        Next,
        /// The block's `RET ALLOW`.
        Allow,
        /// The block's `RET deny`.
        Deny,
        /// The first instruction after the block.
        End,
    }

    #[derive(Clone, Copy)]
    enum Op {
        Load(u32),
        And(u32),
        Jeq { k: u32, yes: Label, no: Label },
    }

    /// The finished program, built block by block. Every block either returns on every path
    /// or is skipped, so the accumulator still holds the syscall number at the next block.
    struct Program {
        code: Vec<libc::sock_filter>,
        deny: u32,
    }

    fn insn(code: u16, jt: u8, jf: u8, k: u32) -> libc::sock_filter {
        libc::sock_filter { code, jt, jf, k }
    }

    impl Program {
        fn new(deny: u32) -> Self {
            Self {
                code: Vec::new(),
                deny,
            }
        }

        /// A block: `ops`, then `RET ALLOW`, then `RET deny` (or `terminal` instead of the
        /// pair when it is `Some`). The syscall test is the first op.
        fn block(&mut self, ops: &[Op], terminal: Option<u32>) {
            let allow_at = ops.len();
            let (tail, deny_at) = match terminal {
                Some(value) => (vec![value], allow_at),
                None => (vec![RET_ALLOW, self.deny], allow_at + 1),
            };
            let end_at = ops.len() + tail.len();
            let target = |label: Label, here: usize| -> u8 {
                let to = match label {
                    Label::Next => here + 1,
                    Label::Allow => allow_at,
                    Label::Deny => deny_at,
                    Label::End => end_at,
                };
                u8::try_from(to - (here + 1)).expect("a block's jumps are short")
            };
            for (here, op) in ops.iter().enumerate() {
                self.code.push(match *op {
                    Op::Load(offset) => insn(BPF_LD_W_ABS, 0, 0, offset),
                    Op::And(mask) => insn(BPF_ALU_AND_K, 0, 0, mask),
                    Op::Jeq { k, yes, no } => {
                        insn(BPF_JEQ_K, target(yes, here), target(no, here), k)
                    }
                });
            }
            for value in tail {
                self.code.push(insn(BPF_RET_K, 0, 0, value));
            }
        }

        fn allow(&mut self, number: u32) {
            self.block(
                &[Op::Jeq {
                    k: number,
                    yes: Label::Next,
                    no: Label::End,
                }],
                None,
            );
        }

        /// `number` is allowed only when the low word of argument `arg` is one of `values`.
        fn allow_arg_in(&mut self, number: u32, arg: u32, values: &[u32]) {
            let mut ops = vec![
                Op::Jeq {
                    k: number,
                    yes: Label::Next,
                    no: Label::End,
                },
                Op::Load(offset_arg_low(arg)),
            ];
            for (index, value) in values.iter().enumerate() {
                let last = index + 1 == values.len();
                ops.push(Op::Jeq {
                    k: *value,
                    yes: Label::Allow,
                    no: if last { Label::Deny } else { Label::Next },
                });
            }
            self.block(&ops, None);
        }

        fn allow_clone_threads(&mut self, number: u32) {
            self.block(
                &[
                    Op::Jeq {
                        k: number,
                        yes: Label::Next,
                        no: Label::End,
                    },
                    Op::Load(offset_arg_low(0)),
                    Op::And(CLONE_NEW_MASK),
                    Op::Jeq {
                        k: 0,
                        yes: Label::Next,
                        no: Label::Deny,
                    },
                    Op::Load(offset_arg_low(0)),
                    Op::And(CLONE_THREAD),
                    Op::Jeq {
                        k: CLONE_THREAD,
                        yes: Label::Allow,
                        no: Label::Deny,
                    },
                ],
                None,
            );
        }

        /// `prlimit64(pid, resource, new_limit, old_limit)`: pid 0 and a null `new_limit`.
        fn allow_prlimit_read(&mut self, number: u32) {
            self.block(
                &[
                    Op::Jeq {
                        k: number,
                        yes: Label::Next,
                        no: Label::End,
                    },
                    Op::Load(offset_arg_low(0)),
                    Op::Jeq {
                        k: 0,
                        yes: Label::Next,
                        no: Label::Deny,
                    },
                    Op::Load(offset_arg_low(2)),
                    Op::Jeq {
                        k: 0,
                        yes: Label::Next,
                        no: Label::Deny,
                    },
                    Op::Load(offset_arg_high(2)),
                    Op::Jeq {
                        k: 0,
                        yes: Label::Allow,
                        no: Label::Deny,
                    },
                ],
                None,
            );
        }

        fn answer(&mut self, number: u32, value: u32) {
            self.block(
                &[Op::Jeq {
                    k: number,
                    yes: Label::Next,
                    no: Label::End,
                }],
                Some(value),
            );
        }
    }

    /// The program for `abi`. `tgid` is the thread group the filter is installed in: the only
    /// one `tgkill` may signal.
    pub(super) fn build(abi: &Abi, deny: DenyAction, tgid: u32) -> Vec<libc::sock_filter> {
        let deny_value = match deny {
            DenyAction::Errno => ret_errno(libc::EPERM),
            DenyAction::KillProcess => RET_KILL_PROCESS,
        };
        let mut program = Program::new(deny_value);
        // The architecture first: a call through another ABI (the 32-bit compat ABI of an
        // x86_64 kernel) has other numbers and must not be matched against these.
        program.code.push(insn(BPF_LD_W_ABS, 0, 0, OFFSET_ARCH));
        program.code.push(insn(BPF_JEQ_K, 1, 0, abi.audit_arch));
        program.code.push(insn(BPF_RET_K, 0, 0, RET_KILL_PROCESS));
        program.code.push(insn(BPF_LD_W_ABS, 0, 0, OFFSET_NR));
        for (_, number) in abi.plain {
            program.allow(*number);
        }
        program.allow_clone_threads(abi.clone);
        program.allow_arg_in(abi.ioctl, 1, &[FIONBIO, FIOCLEX, FIONCLEX]);
        program.allow_arg_in(abi.fcntl, 1, &FCNTL_COMMANDS);
        program.allow_arg_in(abi.tgkill, 0, &[tgid]);
        program.allow_prlimit_read(abi.prlimit64);
        // `clone3` cannot be argument-filtered (its flags are behind a pointer), so it is
        // answered `ENOSYS`: glibc then falls back to `clone`, which is.
        program.answer(abi.clone3, ret_errno(libc::ENOSYS));
        program.code.push(insn(BPF_RET_K, 0, 0, deny_value));
        program.code
    }

    pub(super) fn install(deny: DenyAction) -> Result<(), ConfineError> {
        let Some(abi) = host_abi() else {
            return Err(ConfineError::Unsupported(format!(
                "no seccomp allowlist for architecture {}",
                std::env::consts::ARCH
            )));
        };
        // SAFETY: `getpid` takes no arguments. The filter is installed by the worker in its
        // own, already spawned process, so this is the thread group `tgkill` may signal.
        let tgid = unsafe { libc::getpid() } as u32;
        let mut code = build(abi, deny, tgid);
        let program = libc::sock_fprog {
            len: u16::try_from(code.len())
                .map_err(|_| ConfineError::Io("seccomp program too long".to_owned()))?,
            filter: code.as_mut_ptr(),
        };
        // SAFETY: `prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)` takes plain integers and only
        // restricts the calling thread.
        let result = unsafe {
            libc::prctl(
                libc::PR_SET_NO_NEW_PRIVS,
                1 as c_ulong,
                0 as c_ulong,
                0 as c_ulong,
                0 as c_ulong,
            )
        };
        if result != 0 {
            return Err(errno_error("PR_SET_NO_NEW_PRIVS"));
        }
        // SAFETY: `seccomp(SECCOMP_SET_MODE_FILTER, TSYNC, prog)` reads `program` and the
        // `code` array it points at, both alive until after the call; the kernel copies them.
        let result = unsafe {
            libc::syscall(
                libc::SYS_seccomp,
                libc::SECCOMP_SET_MODE_FILTER as c_ulong,
                libc::SECCOMP_FILTER_FLAG_TSYNC,
                &program as *const libc::sock_fprog,
            )
        };
        match result {
            0 => Ok(()),
            -1 => Err(errno_error("seccomp")),
            thread => Err(ConfineError::Io(format!(
                "seccomp could not synchronise thread {thread}"
            ))),
        }
    }

    fn errno_error(call: &str) -> ConfineError {
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(code) if code == libc::ENOSYS || code == libc::EINVAL => {
                ConfineError::Unsupported(format!("{call}: {error}"))
            }
            _ => ConfineError::Io(format!("{call}: {error}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_only_when_all_four_are_applied() {
        assert_eq!(
            Confinement::from_applied(AppliedMechanisms::ALL),
            Confinement::Full
        );
        assert_eq!(
            Confinement::from_applied(AppliedMechanisms::default()),
            Confinement::None
        );
        for mechanism in Mechanism::ALL {
            let mut applied = AppliedMechanisms::ALL;
            match mechanism {
                Mechanism::Seccomp => applied.seccomp = false,
                Mechanism::OwnUid => applied.own_uid = false,
                Mechanism::CgroupV2 => applied.cgroup_v2 = false,
                Mechanism::NetNamespace => applied.net_namespace = false,
            }
            assert_eq!(
                Confinement::from_applied(applied),
                Confinement::Partial {
                    missing: vec![mechanism]
                }
            );
        }
    }

    #[test]
    fn carrier_policy_is_total_and_conservative() {
        let partial = Confinement::Partial {
            missing: vec![Mechanism::CgroupV2],
        };
        let all = [
            Carrier::InProcess,
            Carrier::Bytes,
            Carrier::HostShm,
            Carrier::CudaIpc,
            Carrier::DevicePeer,
            Carrier::PinnedHost,
            Carrier::NvmeFile,
            Carrier::Remote,
        ];
        for carrier in all {
            let under = |confinement: &Confinement| carrier_permitted(carrier, confinement);
            match carrier {
                Carrier::Bytes | Carrier::Remote => {
                    assert!(under(&Confinement::None) && under(&Confinement::Full));
                }
                Carrier::InProcess | Carrier::NvmeFile => {
                    assert!(!under(&Confinement::Full), "{carrier:?}");
                }
                Carrier::HostShm | Carrier::CudaIpc | Carrier::DevicePeer | Carrier::PinnedHost => {
                    assert!(under(&Confinement::Full), "{carrier:?}");
                    assert!(
                        !under(&partial) && !under(&Confinement::None),
                        "{carrier:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_cgroup_mount_is_found_in_either_layout() {
        let unified = "29 22 0:26 / /sys/fs/cgroup rw,nosuid - cgroup2 cgroup2 rw\n";
        assert_eq!(
            cgroup2_mount(unified),
            Some(("/".to_owned(), "/sys/fs/cgroup".to_owned()))
        );
        let hybrid = "24 23 0:21 / /sys/fs/cgroup ro - tmpfs tmpfs ro\n\
                      25 24 0:22 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n\
                      26 24 0:23 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n";
        assert_eq!(
            cgroup2_mount(hybrid),
            Some(("/".to_owned(), "/sys/fs/cgroup/unified".to_owned()))
        );
        assert_eq!(cgroup2_mount("1 0 0:1 / / rw - ext4 /dev/sda rw\n"), None);
        let spaced = "30 22 0:26 /a /sys/fs/cgroup\\040x rw - cgroup2 cgroup2 rw\n";
        assert_eq!(
            cgroup2_mount(spaced),
            Some(("/a".to_owned(), "/sys/fs/cgroup x".to_owned()))
        );
    }

    #[test]
    fn the_own_group_directory_is_resolved_under_the_mount() {
        assert_eq!(
            own_cgroup_path("12:memory:/x\n0::/user.slice/a.scope\n"),
            Some("/user.slice/a.scope".to_owned())
        );
        assert_eq!(own_cgroup_path("1:name=systemd:/\n"), None);
        assert_eq!(
            cgroup_directory("/", "/sys/fs/cgroup", "/user.slice/a.scope"),
            PathBuf::from("/sys/fs/cgroup/user.slice/a.scope")
        );
        assert_eq!(
            cgroup_directory("/", "/sys/fs/cgroup", "/"),
            PathBuf::from("/sys/fs/cgroup/")
        );
        // A mount whose root is a subgroup (a cgroup namespace or a bind mount).
        assert_eq!(
            cgroup_directory("/kube", "/sys/fs/cgroup", "/kube/pod1"),
            PathBuf::from("/sys/fs/cgroup/pod1")
        );
    }

    #[test]
    fn capabilities_are_read_from_the_status_text() {
        let status = "Name:\tx\nCapEff:\t000001ffffffffff\nCapBnd:\t0\n";
        assert_eq!(effective_capabilities(status), Some(0x0000_01ff_ffff_ffff));
        assert_eq!(effective_capabilities("Name:\tx\n"), None);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn the_allowlist_holds_no_forbidden_call_and_the_program_fits() {
        let forbidden: &[(&str, c_long)] = &[
            ("io_uring_setup", libc::SYS_io_uring_setup),
            ("io_uring_enter", libc::SYS_io_uring_enter),
            ("io_uring_register", libc::SYS_io_uring_register),
            ("socket", libc::SYS_socket),
            ("socketpair", libc::SYS_socketpair),
            ("connect", libc::SYS_connect),
            ("bind", libc::SYS_bind),
            ("listen", libc::SYS_listen),
            ("accept", libc::SYS_accept),
            ("accept4", libc::SYS_accept4),
            ("pidfd_getfd", libc::SYS_pidfd_getfd),
            ("pidfd_open", libc::SYS_pidfd_open),
            ("ptrace", libc::SYS_ptrace),
            ("process_vm_readv", libc::SYS_process_vm_readv),
            ("process_vm_writev", libc::SYS_process_vm_writev),
            ("clone3", libc::SYS_clone3),
            ("execve", libc::SYS_execve),
            ("execveat", libc::SYS_execveat),
            ("openat", libc::SYS_openat),
            ("kill", libc::SYS_kill),
            ("unshare", libc::SYS_unshare),
            ("setns", libc::SYS_setns),
        ];
        let allowed = worker_allowlist();
        for (name, number) in forbidden {
            assert!(
                allowed.iter().all(|entry| entry.number != *number),
                "{name} must not be allowed"
            );
        }
        let mut numbers: Vec<c_long> = allowed.iter().map(|entry| entry.number).collect();
        numbers.sort_unstable();
        numbers.dedup();
        assert_eq!(numbers.len(), allowed.len(), "no duplicate entries");
    }

    /// The numbers written out for the host's architecture are the ones `libc` has.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn the_host_tables_match_libc() {
        macro_rules! check {
            ($abi:expr, $($name:ident => $libc:expr),* $(,)?) => {
                $(
                    let found = $abi.plain.iter().find(|(n, _)| *n == stringify!($name));
                    assert_eq!(
                        found.map(|(_, number)| c_long::from(*number)),
                        Some($libc),
                        "{}",
                        stringify!($name)
                    );
                )*
            };
        }
        let abi = filter::host_abi().expect("supported host");
        check!(abi,
            read => libc::SYS_read, write => libc::SYS_write, readv => libc::SYS_readv,
            writev => libc::SYS_writev, close => libc::SYS_close,
            recvmsg => libc::SYS_recvmsg, sendmsg => libc::SYS_sendmsg,
            recvfrom => libc::SYS_recvfrom, sendto => libc::SYS_sendto,
            getsockopt => libc::SYS_getsockopt, ppoll => libc::SYS_ppoll,
            epoll_create1 => libc::SYS_epoll_create1, epoll_ctl => libc::SYS_epoll_ctl,
            epoll_pwait => libc::SYS_epoll_pwait, fstat => libc::SYS_fstat,
            newfstatat => libc::SYS_newfstatat, statx => libc::SYS_statx,
            ftruncate => libc::SYS_ftruncate, memfd_create => libc::SYS_memfd_create,
            mmap => libc::SYS_mmap, munmap => libc::SYS_munmap, mremap => libc::SYS_mremap,
            mprotect => libc::SYS_mprotect, madvise => libc::SYS_madvise, brk => libc::SYS_brk,
            futex => libc::SYS_futex, set_robust_list => libc::SYS_set_robust_list,
            rseq => libc::SYS_rseq, set_tid_address => libc::SYS_set_tid_address,
            sched_yield => libc::SYS_sched_yield,
            sched_getaffinity => libc::SYS_sched_getaffinity,
            clock_gettime => libc::SYS_clock_gettime, clock_getres => libc::SYS_clock_getres,
            clock_nanosleep => libc::SYS_clock_nanosleep, nanosleep => libc::SYS_nanosleep,
            gettimeofday => libc::SYS_gettimeofday, rt_sigaction => libc::SYS_rt_sigaction,
            rt_sigprocmask => libc::SYS_rt_sigprocmask, rt_sigreturn => libc::SYS_rt_sigreturn,
            sigaltstack => libc::SYS_sigaltstack, restart_syscall => libc::SYS_restart_syscall,
            getrandom => libc::SYS_getrandom, getpid => libc::SYS_getpid,
            gettid => libc::SYS_gettid, exit => libc::SYS_exit,
            exit_group => libc::SYS_exit_group,
        );
        assert_eq!(c_long::from(abi.clone), libc::SYS_clone);
        assert_eq!(c_long::from(abi.clone3), libc::SYS_clone3);
        assert_eq!(c_long::from(abi.ioctl), libc::SYS_ioctl);
        assert_eq!(c_long::from(abi.prlimit64), libc::SYS_prlimit64);
        assert_eq!(c_long::from(abi.tgkill), libc::SYS_tgkill);
        assert_eq!(c_long::from(abi.fcntl), libc::SYS_fcntl);
        #[cfg(target_arch = "x86_64")]
        {
            check!(abi, poll => libc::SYS_poll, epoll_wait => libc::SYS_epoll_wait);
        }
        assert_eq!(
            abi.audit_arch,
            if cfg!(target_arch = "x86_64") {
                0xC000_003E
            } else {
                0xC000_00B7
            }
        );
    }

    // --- an interpreter for the opcodes the program uses ---------------------------------

    const TGID: u32 = 4242;

    /// Run `program` over a `seccomp_data` and return the action word.
    fn run(program: &[libc::sock_filter], arch: u32, nr: u32, args: [u64; 6]) -> u32 {
        let mut data = [0u8; 64];
        data[0..4].copy_from_slice(&nr.to_le_bytes());
        data[4..8].copy_from_slice(&arch.to_le_bytes());
        for (index, arg) in args.iter().enumerate() {
            data[16 + 8 * index..24 + 8 * index].copy_from_slice(&arg.to_le_bytes());
        }
        let mut accumulator = 0u32;
        let mut pc = 0usize;
        loop {
            let insn = program[pc];
            pc += 1;
            match insn.code {
                0x20 => {
                    let at = insn.k as usize;
                    accumulator =
                        u32::from_le_bytes(data[at..at + 4].try_into().expect("in range"));
                }
                0x54 => accumulator &= insn.k,
                0x15 => {
                    pc += usize::from(if accumulator == insn.k {
                        insn.jt
                    } else {
                        insn.jf
                    });
                }
                0x06 => return insn.k,
                other => panic!("the program uses an opcode the interpreter lacks: {other:#x}"),
            }
        }
    }

    const ALLOW: u32 = filter::RET_ALLOW;
    const EPERM: u32 = libc::SECCOMP_RET_ERRNO | libc::EPERM as u32;
    const ENOSYS: u32 = libc::SECCOMP_RET_ERRNO | libc::ENOSYS as u32;
    const KILL: u32 = filter::RET_KILL_PROCESS;

    fn number(abi: &filter::Abi, name: &str) -> u32 {
        abi.plain
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, number)| *number)
            .expect("on the list")
    }

    #[test]
    fn the_program_for_each_architecture_does_what_the_module_docs_say() {
        let thread = (libc::CLONE_VM
            | libc::CLONE_FS
            | libc::CLONE_FILES
            | libc::CLONE_SIGHAND
            | libc::CLONE_THREAD) as u64;
        for abi in [&filter::X86_64, &filter::AARCH64] {
            let program = filter::build(abi, DenyAction::Errno, TGID);
            assert!(program.len() < 4096);
            let call = |nr: u32, args: [u64; 6]| run(&program, abi.audit_arch, nr, args);
            let none = [0u64; 6];

            // Every plain entry is allowed, and the same number through the x32 ABI is not.
            for (name, nr) in abi.plain {
                assert_eq!(call(*nr, none), ALLOW, "{name}");
                assert_eq!(call(*nr | 0x4000_0000, none), EPERM, "{name} as x32");
            }
            // Off the list: an unknown number, and the numbers (on either ABI) of calls that
            // are not on this one's list.
            let guarded = [
                abi.clone,
                abi.ioctl,
                abi.fcntl,
                abi.tgkill,
                abi.prlimit64,
                abi.clone3,
            ];
            for nr in [9999, 41, 42, 49, 50, 59, 200, 203, 425, 426, 427, 438] {
                let listed = abi.plain.iter().any(|(_, n)| *n == nr) || guarded.contains(&nr);
                if !listed {
                    assert_eq!(call(nr, none), EPERM, "nr {nr}");
                }
            }
            // A call through another ABI kills the process.
            let other = if abi.audit_arch == filter::X86_64.audit_arch {
                filter::AARCH64.audit_arch
            } else {
                filter::X86_64.audit_arch
            };
            assert_eq!(run(&program, other, number(abi, "read"), none), KILL);
            assert_eq!(run(&program, 0x4000_0003, number(abi, "read"), none), KILL);

            // clone: threads only, no namespace bit.
            let clone = |flags: u64| call(abi.clone, [flags, 0, 0, 0, 0, 0]);
            assert_eq!(clone(thread), ALLOW);
            assert_eq!(clone(thread | libc::CLONE_SETTLS as u64 | 17), ALLOW);
            assert_eq!(clone(libc::SIGCHLD as u64), EPERM);
            assert_eq!(clone(libc::CLONE_VM as u64), EPERM);
            assert_eq!(clone(thread | libc::CLONE_NEWNET as u64), EPERM);
            assert_eq!(clone(thread | libc::CLONE_NEWUSER as u64), EPERM);
            assert_eq!(clone(thread | libc::CLONE_NEWNS as u64), EPERM);
            assert_eq!(call(abi.clone3, none), ENOSYS);

            // ioctl: three requests.
            let ioctl = |request: u64| call(abi.ioctl, [0, request, 0, 0, 0, 0]);
            for request in [
                u64::from(libc::FIONBIO as u32),
                u64::from(libc::FIOCLEX as u32),
                u64::from(libc::FIONCLEX as u32),
            ] {
                assert_eq!(ioctl(request), ALLOW);
            }
            assert_eq!(ioctl(0x5412), EPERM, "TIOCSTI");

            // prlimit64: read the own limits only.
            let prlimit = |pid: u64, new: u64| call(abi.prlimit64, [pid, 7, new, 0, 0, 0]);
            assert_eq!(prlimit(0, 0), ALLOW);
            assert_eq!(prlimit(1, 0), EPERM);
            assert_eq!(prlimit(0, 8), EPERM);
            assert_eq!(prlimit(0, 1 << 32), EPERM, "high word of the new limit");

            // tgkill: the installing thread group only.
            let tgkill = |tgid: u64| call(abi.tgkill, [tgid, 1, 0, 0, 0, 0]);
            assert_eq!(tgkill(u64::from(TGID)), ALLOW);
            assert_eq!(tgkill(1), EPERM);
            assert_eq!(tgkill(u64::from(TGID) + 1), EPERM);

            // fcntl: seven commands; ownership and signal commands denied.
            let fcntl = |command: u64| call(abi.fcntl, [3, command, 0, 0, 0, 0]);
            for command in filter::FCNTL_COMMANDS {
                assert_eq!(fcntl(u64::from(command)), ALLOW, "fcntl {command}");
            }
            for command in [
                libc::F_SETOWN,
                libc::F_GETOWN,
                10, // F_SETSIG
                11, // F_GETSIG
                15, // F_SETOWN_EX
                libc::F_DUPFD,
                libc::F_SETLK,
                libc::F_SETPIPE_SZ,
            ] {
                assert_eq!(fcntl(command as u64), EPERM, "fcntl {command}");
            }

            // The kill action applies to a denied call; the architecture kill is the same.
            let killing = filter::build(abi, DenyAction::KillProcess, TGID);
            assert_eq!(run(&killing, abi.audit_arch, 9999, none), KILL);
            assert_eq!(
                run(&killing, abi.audit_arch, number(abi, "read"), none),
                ALLOW
            );
            assert_eq!(run(&killing, abi.audit_arch, abi.clone3, none), ENOSYS);
        }
    }
}
