//! The CUDA driver: whether it is there, what it can do, and a primary-context guard.
//!
//! With feature `cuda-ipc`, `cudarc` is built with `dynamic-loading`: `libcuda` is opened on the
//! first driver call, and when it cannot be opened that call **panics**. The only door to a
//! driver call in this crate is [`CudaDriver::open`], which asks whether the library can be
//! opened first (`is_culib_present` with `cuda-ipc`, a `dlopen` of `libcuda.so.1` with
//! `cuda-ipc-core` alone, where the linking mode is whatever the rest of the build chose) and
//! returns a typed [`CudaError::NoDriver`] when it cannot. Under dynamic linking a missing
//! `libcuda` fails the process load before this code runs. Every GPU operation takes a
//! `&CudaDriver`, so no call can be made without that proof.

use std::any::Any;
use std::ffi::{c_int, CStr};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

use cudarc::driver::sys;

use super::CudaError;

/// `cuInit` ran once, or why the driver is unusable. Cached: probing again changes nothing.
static INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Whether the driver library can be opened by name.
#[cfg(feature = "cuda-ipc")]
fn driver_library_present() -> bool {
    // SAFETY: `is_culib_present` only tries to open the driver library by its candidate names
    // and reports whether one opened; it calls nothing in it.
    unsafe { sys::is_culib_present() }
}

/// Whether the driver library can be opened by name. Without `cuda-ipc` the linking mode of
/// `cudarc` is not ours to know (`is_culib_present` exists only under dynamic loading), so ask
/// the dynamic loader directly; under dynamic linking the library is loaded already.
#[cfg(not(feature = "cuda-ipc"))]
fn driver_library_present() -> bool {
    ["libcuda.so.1", "libcuda.so"].iter().any(|name| {
        let Ok(name) = std::ffi::CString::new(*name) else {
            return false;
        };
        // SAFETY: `name` is a valid NUL-terminated string; the handle is closed right away.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return false;
        }
        // SAFETY: `handle` came from a successful `dlopen`.
        unsafe { libc::dlclose(handle) };
        true
    })
}

fn initialise() -> &'static Result<(), String> {
    INIT.get_or_init(|| {
        if !driver_library_present() {
            return Err("the CUDA driver library (libcuda) cannot be loaded".to_owned());
        }
        // A driver that lacks `cuInit` makes the loader panic; that is "no usable driver", not a
        // crash of the host.
        // SAFETY: the library is present; `cuInit(0)` takes no pointer and may be called more
        // than once.
        // `error_name` calls into the driver too, so it sits inside the guarded section.
        let outcome = guarded(|| {
            let result = unsafe { sys::cuInit(0) };
            if result == sys::CUresult::CUDA_SUCCESS {
                return Ok(());
            }
            Err(CudaError::NoDriver {
                reason: format!(
                    "cuInit failed with CUresult {} ({})",
                    result as u32,
                    error_name(result)
                ),
            })
        });
        match outcome {
            Ok(()) => Ok(()),
            Err(CudaError::NoDriver { reason }) => Err(reason),
            Err(CudaError::MissingEntryPoint) => {
                Err("the CUDA driver library has no cuInit".to_owned())
            }
            Err(other) => Err(other.to_string()),
        }
    })
}

/// The driver's name for `result`, or a placeholder. Only called after a driver call answered,
/// so the library is loaded.
fn error_name(result: sys::CUresult) -> String {
    let mut text: *const std::ffi::c_char = std::ptr::null();
    // SAFETY: `text` is a valid out pointer; on success the driver stores a pointer to a static
    // NUL-terminated string that is never freed.
    let status = unsafe { sys::cuGetErrorName(result, &mut text) };
    if status != sys::CUresult::CUDA_SUCCESS || text.is_null() {
        return "unknown".to_owned();
    }
    // SAFETY: `text` is non-null and points to a static NUL-terminated string (see above).
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

/// Run the FFI section of a public GPU entry point so that a driver entry point missing from
/// the installed `libcuda` (cudarc's lazy loader panics on it) becomes
/// [`CudaError::MissingEntryPoint`] instead of unwinding into the caller.
///
/// `AssertUnwindSafe` reasoning: the closure only touches values that are dropped or left
/// unused after a panic (the result is replaced by the error), the RAII guards inside it
/// (`VmmMapping`, `ContextGuard`, `DeviceContext`) release driver resources while unwinding,
/// and no shared state is left half-updated (the statics are cudarc's `OnceLock`s and `INIT`,
/// which a panic leaves unset). With `panic = "abort"` there is no unwinding and the catch
/// does not exist: the process aborts.
pub(crate) fn guarded<T>(section: impl FnOnce() -> Result<T, CudaError>) -> Result<T, CudaError> {
    match catch_unwind(AssertUnwindSafe(section)) {
        Ok(result) => result,
        Err(payload) if is_loader_panic(payload.as_ref()) => Err(CudaError::MissingEntryPoint),
        // Any other panic is a bug of ours or of a caller's closure: it is not a driver
        // problem and is not hidden.
        Err(payload) => resume_unwind(payload),
    }
}

/// Whether a panic payload is one of cudarc's loader panics: `Missing symbol {name}: ..` (an
/// entry point absent from `libcuda`) or `Unable to dynamically load ..` (no library).
fn is_loader_panic(payload: &(dyn Any + Send)) -> bool {
    let text = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied());
    text.is_some_and(|text| {
        text.starts_with("Missing symbol") || text.starts_with("Unable to dynamically load")
    })
}

/// The same for a destructor: a panic is swallowed (a destructor cannot report it), and the
/// resource it tried to release stays allocated.
pub(crate) fn guarded_drop(section: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(section));
}

/// Turn a driver result into a typed error naming the call.
pub(crate) fn check(call: &'static str, result: sys::CUresult) -> Result<(), CudaError> {
    if result == sys::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    Err(CudaError::Driver {
        call,
        code: result as u32,
        name: error_name(result),
    })
}

/// Proof that the CUDA driver is loaded and initialised. Zero-sized and `Copy`: holding one is
/// the only way to reach a driver call in this crate.
#[derive(Clone, Copy, Debug)]
pub struct CudaDriver {
    _private: (),
}

impl CudaDriver {
    /// The driver, or [`CudaError::NoDriver`] on a host without one (or whose `cuInit` fails).
    /// Never panics.
    pub fn open() -> Result<Self, CudaError> {
        match initialise() {
            Ok(()) => Ok(Self { _private: () }),
            Err(reason) => Err(CudaError::NoDriver {
                reason: reason.clone(),
            }),
        }
    }

    /// Ask the driver what this host can do. [`CudaProbe::Absent`] when there is no driver;
    /// otherwise one [`DeviceInfo`] per device and the ordered peer matrix, all measured, none
    /// assumed (peer access in particular is whatever `cuDeviceCanAccessPeer` says).
    pub fn probe() -> CudaProbe {
        let driver = match Self::open() {
            Ok(driver) => driver,
            Err(error) => {
                return CudaProbe::Absent {
                    reason: error.to_string(),
                }
            }
        };
        // A driver older than the entry points used here makes the loader panic on the missing
        // symbol; the probe reports that as absent instead of unwinding into the caller.
        match guarded(|| driver.probe_present()) {
            Ok(probe) => probe,
            Err(error) => CudaProbe::Absent {
                reason: error.to_string(),
            },
        }
    }

    fn probe_present(&self) -> Result<CudaProbe, CudaError> {
        let count = self.device_count()?;
        let mut devices = Vec::new();
        for ordinal in 0..count {
            devices.push(self.describe(ordinal)?);
        }
        let mut peers = Vec::new();
        for from in 0..count {
            for to in 0..count {
                if from == to {
                    continue;
                }
                peers.push(PeerLink {
                    from,
                    to,
                    can_access: self.can_access_peer(from, to)?,
                });
            }
        }
        Ok(CudaProbe::Present { devices, peers })
    }

    /// The number of CUDA devices.
    pub fn device_count(&self) -> Result<u32, CudaError> {
        guarded(|| {
            let mut count: c_int = 0;
            // SAFETY: `count` is a valid out pointer; the driver is initialised (`self` exists).
            check("cuDeviceGetCount", unsafe {
                sys::cuDeviceGetCount(&mut count)
            })?;
            u32::try_from(count).map_err(|_| CudaError::Overflow)
        })
    }

    // `CUDA_VISIBLE_DEVICES` remaps the ordinals the driver shows; `cuDeviceGet`, the probe and
    // every `CUmemLocation.id` of this module use the same remapped numbering, so they agree.
    pub(crate) fn device(&self, ordinal: u32) -> Result<sys::CUdevice, CudaError> {
        if ordinal >= self.device_count()? {
            return Err(CudaError::NoSuchDevice { ordinal });
        }
        let ordinal_int = c_int::try_from(ordinal).map_err(|_| CudaError::Overflow)?;
        let mut device: sys::CUdevice = 0;
        // SAFETY: `device` is a valid out pointer and `ordinal` was checked against the count.
        check("cuDeviceGet", unsafe {
            sys::cuDeviceGet(&mut device, ordinal_int)
        })?;
        Ok(device)
    }

    pub(crate) fn attribute(
        &self,
        device: sys::CUdevice,
        attribute: sys::CUdevice_attribute,
    ) -> Result<i32, CudaError> {
        let mut value: c_int = 0;
        // SAFETY: `value` is a valid out pointer and `device` came from `cuDeviceGet`.
        check("cuDeviceGetAttribute", unsafe {
            sys::cuDeviceGetAttribute(&mut value, attribute, device)
        })?;
        Ok(value)
    }

    fn describe(&self, ordinal: u32) -> Result<DeviceInfo, CudaError> {
        let device = self.device(ordinal)?;
        let mut name = [0 as std::ffi::c_char; 256];
        // SAFETY: the buffer is 256 bytes and that length is passed; the driver writes a
        // NUL-terminated name into it.
        check("cuDeviceGetName", unsafe {
            sys::cuDeviceGetName(name.as_mut_ptr(), 256, device)
        })?;
        // SAFETY: the driver NUL-terminated the buffer (see above); its length bounds the read.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut total: usize = 0;
        // SAFETY: `total` is a valid out pointer and `device` came from `cuDeviceGet`.
        check("cuDeviceTotalMem_v2", unsafe {
            sys::cuDeviceTotalMem_v2(&mut total, device)
        })?;
        let vmm_supported = self.attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        )? != 0;
        let posix_fd_supported = self.attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        )? != 0;
        let allocation_granularity = if vmm_supported && posix_fd_supported {
            super::vmm::allocation_granularity(self, ordinal).ok()
        } else {
            None
        };
        Ok(DeviceInfo {
            ordinal,
            name,
            total_memory: total as u64,
            vmm_supported,
            posix_fd_supported,
            allocation_granularity,
        })
    }

    pub(crate) fn can_access_peer(&self, from: u32, to: u32) -> Result<bool, CudaError> {
        let from = self.device(from)?;
        let to = self.device(to)?;
        let mut can: c_int = 0;
        // SAFETY: `can` is a valid out pointer; both devices came from `cuDeviceGet`.
        check("cuDeviceCanAccessPeer", unsafe {
            sys::cuDeviceCanAccessPeer(&mut can, from, to)
        })?;
        Ok(can != 0)
    }
}

/// What one device offers the carriers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub ordinal: u32,
    pub name: String,
    pub total_memory: u64,
    /// `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`.
    pub vmm_supported: bool,
    /// `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED`: a VMM allocation can
    /// be exported as a POSIX descriptor.
    pub posix_fd_supported: bool,
    /// `cuMemGetAllocationGranularity` (minimum) for a POSIX-descriptor-exportable allocation
    /// on this device; `None` when VMM export is not offered or the query failed.
    pub allocation_granularity: Option<u64>,
}

/// One ordered pair of the peer matrix: whether `from` can access memory of `to`
/// (`cuDeviceCanAccessPeer`). The relation is probed per direction, never assumed symmetric.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeerLink {
    pub from: u32,
    pub to: u32,
    pub can_access: bool,
}

/// The result of [`CudaDriver::probe`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaProbe {
    /// No usable driver: the CUDA carriers are unavailable on this host.
    Absent { reason: String },
    /// A driver and its devices.
    Present {
        devices: Vec<DeviceInfo>,
        /// Every ordered pair of distinct devices, in `(from, to)` order.
        peers: Vec<PeerLink>,
    },
}

impl CudaProbe {
    /// Whether a driver answered.
    pub fn is_present(&self) -> bool {
        matches!(self, Self::Present { .. })
    }

    /// Whether `from` can access `to`; `None` when the pair is not in the matrix (a device
    /// with itself, an unknown ordinal, or an absent driver).
    pub fn can_access(&self, from: u32, to: u32) -> Option<bool> {
        let Self::Present { peers, .. } = self else {
            return None;
        };
        peers
            .iter()
            .find(|link| link.from == from && link.to == to)
            .map(|link| link.can_access)
    }

    /// Whether the report is internally consistent: devices numbered 0..n in order, the matrix
    /// holds exactly the ordered pairs of distinct devices in order, and a granularity is only
    /// reported for a device that offers VMM export. An `Absent` probe is consistent.
    pub fn is_consistent(&self) -> bool {
        let Self::Present { devices, peers } = self else {
            return true;
        };
        let count = devices.len() as u32;
        let ordinals_ok = devices
            .iter()
            .enumerate()
            .all(|(index, device)| device.ordinal as usize == index);
        let granularity_ok = devices
            .iter()
            .all(|device| match device.allocation_granularity {
                Some(granularity) => {
                    granularity > 0 && device.vmm_supported && device.posix_fd_supported
                }
                None => true,
            });
        let expected = (0..count)
            .flat_map(|from| (0..count).map(move |to| (from, to)))
            .filter(|(from, to)| from != to);
        let matrix_ok = peers.len() == devices.len() * devices.len().saturating_sub(1)
            && peers
                .iter()
                .zip(expected)
                .all(|(link, (from, to))| link.from == from && link.to == to);
        ordinals_ok && granularity_ok && matrix_ok
    }
}

/// The primary context of one device, retained for as long as the value lives. A GPU object
/// keeps one, so its drop can make the context current before it frees driver resources.
pub(crate) struct DeviceContext {
    device: sys::CUdevice,
    context: sys::CUcontext,
}

// SAFETY: a `CUcontext` is a process-wide token for a driver object; the driver API is
// thread-safe and a context is made current on a thread by `cuCtxPushCurrent`, which `enter`
// does for the span of one operation. The value holds no thread-affine state.
unsafe impl Send for DeviceContext {}

impl DeviceContext {
    pub(crate) fn retain(driver: &CudaDriver, ordinal: u32) -> Result<Self, CudaError> {
        let device = driver.device(ordinal)?;
        let mut context: sys::CUcontext = std::ptr::null_mut();
        // SAFETY: `context` is a valid out pointer and `device` came from `cuDeviceGet`.
        check("cuDevicePrimaryCtxRetain", unsafe {
            sys::cuDevicePrimaryCtxRetain(&mut context, device)
        })?;
        Ok(Self { device, context })
    }

    /// Make the context current on this thread until the guard drops.
    pub(crate) fn enter(&self) -> Result<ContextGuard, CudaError> {
        // SAFETY: `context` is a live primary context retained by `self`.
        check("cuCtxPushCurrent_v2", unsafe {
            sys::cuCtxPushCurrent_v2(self.context)
        })?;
        Ok(ContextGuard {
            _not_send: std::marker::PhantomData,
        })
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        // SAFETY: balances the `cuDevicePrimaryCtxRetain` of `retain`; a failure cannot be
        // reported from a destructor and only leaves the context retained.
        guarded_drop(|| {
            let _ = unsafe { sys::cuDevicePrimaryCtxRelease_v2(self.device) };
        });
    }
}

/// Pops the context it pushed. Tied to the thread that pushed it (`!Send`); it does not borrow
/// the context, so the owner can still be used mutably while a guard is live.
pub(crate) struct ContextGuard {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        let mut popped: sys::CUcontext = std::ptr::null_mut();
        // SAFETY: balances the `cuCtxPushCurrent_v2` of `enter` on this thread; `popped` is a
        // valid out pointer. A failure cannot be reported from a destructor.
        guarded_drop(|| {
            let _ = unsafe { sys::cuCtxPopCurrent_v2(&mut popped) };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(ordinal: u32, granularity: Option<u64>) -> DeviceInfo {
        DeviceInfo {
            ordinal,
            name: format!("device {ordinal}"),
            total_memory: 1 << 30,
            vmm_supported: true,
            posix_fd_supported: true,
            allocation_granularity: granularity,
        }
    }

    fn link(from: u32, to: u32, can_access: bool) -> PeerLink {
        PeerLink {
            from,
            to,
            can_access,
        }
    }

    #[test]
    fn an_absent_probe_is_consistent_and_has_no_matrix() {
        let probe = CudaProbe::Absent {
            reason: "none".to_owned(),
        };
        assert!(!probe.is_present() && probe.is_consistent());
        assert_eq!(probe.can_access(0, 1), None);
    }

    #[test]
    fn a_two_device_matrix_is_ordered_and_directional() {
        let probe = CudaProbe::Present {
            devices: vec![device(0, Some(2 << 20)), device(1, None)],
            peers: vec![link(0, 1, true), link(1, 0, false)],
        };
        assert!(probe.is_consistent());
        assert_eq!(probe.can_access(0, 1), Some(true));
        assert_eq!(probe.can_access(1, 0), Some(false));
        assert_eq!(probe.can_access(0, 0), None);
        assert_eq!(probe.can_access(0, 2), None);
    }

    #[test]
    fn a_broken_report_is_not_consistent() {
        let missing_pair = CudaProbe::Present {
            devices: vec![device(0, None), device(1, None)],
            peers: vec![link(0, 1, true)],
        };
        assert!(!missing_pair.is_consistent());
        let wrong_order = CudaProbe::Present {
            devices: vec![device(0, None), device(1, None)],
            peers: vec![link(1, 0, true), link(0, 1, true)],
        };
        assert!(!wrong_order.is_consistent());
        let wrong_ordinal = CudaProbe::Present {
            devices: vec![device(1, None)],
            peers: vec![],
        };
        assert!(!wrong_ordinal.is_consistent());
        let zero_granularity = CudaProbe::Present {
            devices: vec![device(0, Some(0))],
            peers: vec![],
        };
        assert!(!zero_granularity.is_consistent());
        let mut no_vmm = device(0, Some(2 << 20));
        no_vmm.vmm_supported = false;
        let granularity_without_vmm = CudaProbe::Present {
            devices: vec![no_vmm],
            peers: vec![],
        };
        assert!(!granularity_without_vmm.is_consistent());
    }

    #[test]
    fn a_panic_in_the_ffi_section_becomes_a_typed_error() {
        let caught: Result<(), CudaError> =
            guarded(|| panic!("Missing symbol cuFoo: undefined symbol"));
        assert_eq!(caught, Err(CudaError::MissingEntryPoint));
        let no_library: Result<(), CudaError> =
            guarded(|| panic!("Unable to dynamically load the \"cuda\" shared library"));
        assert_eq!(no_library, Err(CudaError::MissingEntryPoint));
        // Any other panic is re-raised, not reported as a driver problem.
        let other = std::panic::catch_unwind(|| {
            let _: Result<(), CudaError> = guarded(|| panic!("index out of bounds"));
        });
        assert!(other.is_err());
        assert_eq!(guarded(|| Ok(5)), Ok(5));
        let mut ran = false;
        guarded_drop(|| {
            ran = true;
            panic!("Missing symbol cuBar: undefined symbol");
        });
        assert!(ran);
    }

    #[test]
    fn a_single_device_has_an_empty_matrix() {
        let probe = CudaProbe::Present {
            devices: vec![device(0, Some(2 << 20))],
            peers: vec![],
        };
        assert!(probe.is_consistent());
    }

    /// On a host without a driver the probe must say so and not panic; on a host with one it
    /// must be consistent. Either way no cudarc call runs unless `is_culib_present` said yes.
    #[test]
    fn the_probe_never_panics_and_is_consistent() {
        let probe = CudaDriver::probe();
        assert!(probe.is_consistent(), "{probe:?}");
        match (&probe, CudaDriver::open()) {
            (CudaProbe::Absent { .. }, Err(CudaError::NoDriver { .. })) => {}
            (CudaProbe::Present { .. }, Ok(_)) => {}
            // A driver that opened but whose probe failed part-way reports Absent while open
            // succeeds; the reason then names the failing call.
            (CudaProbe::Absent { reason }, Ok(_)) => assert!(!reason.is_empty()),
            (probe, open) => panic!("probe {probe:?} disagrees with open {open:?}"),
        }
    }
}
