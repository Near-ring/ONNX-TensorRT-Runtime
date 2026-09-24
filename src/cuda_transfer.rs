//! Batched pinned-memory copies between ORT-owned buffers, bypassing ort's copy cache.
//! Own one nonblocking stream and primary-context reference per native GPU session.
//! Raw CUDA pointers never escape this module.
use crate::Result;
use anyhow::{Context, ensure};
use libloading::Library;
use ort::{
    memory::AllocationDevice,
    value::{DynTensor, TensorElementType},
};
use std::{
    ffi::{CStr, c_char, c_int, c_void},
    ptr,
    sync::OnceLock,
};

type CuContext = *mut c_void;
type PointerAttribute = unsafe extern "system" fn(*mut c_void, c_int, u64) -> c_int;
type ContextPush = unsafe extern "system" fn(CuContext) -> c_int;
type ContextPop = unsafe extern "system" fn(*mut CuContext) -> c_int;
type ContextDevice = unsafe extern "system" fn(*mut c_int) -> c_int;
type Init = unsafe extern "system" fn(u32) -> c_int;
type DeviceGet = unsafe extern "system" fn(*mut c_int, c_int) -> c_int;
type ContextRetain = unsafe extern "system" fn(*mut CuContext, c_int) -> c_int;
type ContextRelease = unsafe extern "system" fn(c_int) -> c_int;
type StreamCreate = unsafe extern "system" fn(*mut *mut c_void, u32) -> c_int;
type StreamDestroy = unsafe extern "system" fn(*mut c_void) -> c_int;
type Upload = unsafe extern "system" fn(u64, *const c_void, usize, *mut c_void) -> c_int;
type Download = unsafe extern "system" fn(*mut c_void, u64, usize, *mut c_void) -> c_int;
type StreamSync = unsafe extern "system" fn(*mut c_void) -> c_int;
type ErrorName = unsafe extern "system" fn(c_int, *mut *const c_char) -> c_int;

struct Driver {
    pointer_attribute: PointerAttribute,
    push: ContextPush,
    pop: ContextPop,
    device: ContextDevice,
    init: Init,
    device_get: DeviceGet,
    retain: ContextRetain,
    release: ContextRelease,
    stream_create: StreamCreate,
    stream_destroy: StreamDestroy,
    upload: Upload,
    download: Download,
    stream_sync: StreamSync,
    error_name: ErrorName,
    _library: Library,
}

impl Driver {
    fn load() -> Result<Self> {
        ensure!(
            cfg!(target_pointer_width = "64"),
            "CUDA transfers require a 64-bit target"
        );
        let name = if cfg!(target_os = "windows") {
            "nvcuda.dll"
        } else {
            "libcuda.so.1"
        };
        // SAFETY: Load only the system CUDA driver and its documented C ABI symbols.
        // The Library is kept alive with the function pointers for the process lifetime.
        unsafe {
            let library = Library::new(name).with_context(|| format!(
                "Cannot load CUDA driver {name}. Install the NVIDIA driver required by your CUDA runtime \
                 (R580 or newer for CUDA 13.x), expose the GPU to this process, and restart it. \
                 Alternatively use OnnxOptions::cpu()."
            ))?;
            Ok(Self {
                pointer_attribute: *library.get(b"cuPointerGetAttribute\0")?,
                push: *library.get(b"cuCtxPushCurrent_v2\0")?,
                pop: *library.get(b"cuCtxPopCurrent_v2\0")?,
                device: *library.get(b"cuCtxGetDevice\0")?,
                init: *library.get(b"cuInit\0")?,
                device_get: *library.get(b"cuDeviceGet\0")?,
                retain: *library.get(b"cuDevicePrimaryCtxRetain\0")?,
                release: *library.get(b"cuDevicePrimaryCtxRelease_v2\0")?,
                stream_create: *library.get(b"cuStreamCreate\0")?,
                stream_destroy: *library.get(b"cuStreamDestroy_v2\0")?,
                upload: *library.get(b"cuMemcpyHtoDAsync_v2\0")?,
                download: *library.get(b"cuMemcpyDtoHAsync_v2\0")?,
                stream_sync: *library.get(b"cuStreamSynchronize\0")?,
                error_name: *library.get(b"cuGetErrorName\0")?,
                _library: library,
            })
        }
    }

    fn check(&self, status: c_int, operation: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let mut name = ptr::null();
        // SAFETY: CUDA writes a driver-owned, nul-terminated constant string on success.
        let name = unsafe {
            if (self.error_name)(status, &mut name) == 0 && !name.is_null() {
                CStr::from_ptr(name).to_string_lossy()
            } else {
                "unknown CUDA error".into()
            }
        };
        anyhow::bail!("{operation}: {name} ({status})")
    }
}

// This caches only library symbols, never tensors, GPU allocations, or CUDA contexts.
static DRIVER: OnceLock<Result<Driver>> = OnceLock::new();

struct ContextScope<'a> {
    driver: &'a Driver,
    active: bool,
    stream: *mut c_void,
}

impl ContextScope<'_> {
    fn restore(mut self) -> Result<()> {
        self.active = false;
        let mut popped = ptr::null_mut();
        // SAFETY: One successful push precedes this pop, on this same synchronous thread.
        self.driver
            .check(unsafe { (self.driver.pop)(&mut popped) }, "cuCtxPopCurrent")
    }
}

impl Drop for ContextScope<'_> {
    fn drop(&mut self) {
        if self.active {
            let mut popped = ptr::null_mut();
            // SAFETY: Unwinding cleanup balances our successful push; never destroys the context.
            // Normal returns use restore() to report errors instead of discarding them here.
            unsafe {
                if !self.stream.is_null() {
                    (self.driver.stream_sync)(self.stream);
                }
                (self.driver.pop)(&mut popped);
            }
        }
    }
}

pub(crate) fn checked_bytes(shape: &[i64], dtype: &TensorElementType) -> Result<usize> {
    let width = dtype
        .byte_size(1)
        .filter(|&n| n > 0)
        .context("CUDA copy requires a byte-addressable tensor type")?;
    ensure!(
        shape.iter().all(|&d| d >= 0),
        "Negative CUDA copy dimension"
    );
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(width, |bytes, &dimension| {
        bytes
            .checked_mul(usize::try_from(dimension)?)
            .context("CUDA copy byte count overflow")
    })
}

/// Retains the CUDA primary context used by ORT's runtime API. This type is
/// intentionally thread-bound. Native sessions and buffers must drop before it.
/// Raw handles make this type, and the enclosing OnnxSession, !Send and !Sync.
pub(crate) struct CudaStream {
    driver: &'static Driver,
    context: CuContext,
    device_id: i32,
    device: c_int,
    stream: *mut c_void,
}

impl CudaStream {
    pub fn new(device_id: i32) -> Result<Self> {
        ensure!(device_id >= 0, "Negative CUDA device id");
        let driver = DRIVER
            .get_or_init(Driver::load)
            .as_ref()
            .map_err(|e| anyhow::anyhow!("{e:#}"))?;
        let mut device = 0;
        let mut context = ptr::null_mut();
        // SAFETY: Documented driver ABI; outputs point to correctly sized storage.
        unsafe {
            driver.check((driver.init)(0), "cuInit")?;
            driver.check((driver.device_get)(&mut device, device_id), "cuDeviceGet")?;
            driver.check(
                (driver.retain)(&mut context, device),
                "cuDevicePrimaryCtxRetain",
            )?;
        }
        let mut owned = Self {
            driver,
            context,
            device_id,
            device,
            stream: ptr::null_mut(),
        };
        let scope = owned.enter()?;
        // SAFETY: The retained primary context is current. Flag 1 is
        // CU_STREAM_NON_BLOCKING, avoiding legacy default-stream dependencies.
        driver.check(
            unsafe { (driver.stream_create)(&mut owned.stream, 1) },
            "cuStreamCreate",
        )?;
        scope.restore()?;
        Ok(owned)
    }

    fn enter(&self) -> Result<ContextScope<'static>> {
        // SAFETY: Our primary-context reference keeps the context alive.
        self.driver.check(
            unsafe { (self.driver.push)(self.context) },
            "cuCtxPushCurrent",
        )?;
        Ok(ContextScope {
            driver: self.driver,
            active: true,
            stream: self.stream,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        let scope = self.enter()?;
        // SAFETY: Stream belongs to this retained, current context.
        let result = self.driver.check(
            unsafe { (self.driver.stream_sync)(self.stream) },
            "cuStreamSynchronize session",
        );
        result.and(scope.restore())
    }

    // These providers must remain private to a native session that retains this
    // stream until AFTER the provider, session, bindings and allocators are gone.
    pub fn cuda_provider(&self, provider: ort::ep::CUDA) -> ort::ep::CUDA {
        // SAFETY: NativeSession's field order, and load/compile local declaration
        // order, keep this stream alive throughout all ORT use including teardown.
        unsafe { provider.with_compute_stream(self.stream.cast()) }
    }

    pub fn tensorrt_provider(&self, provider: ort::ep::TensorRT) -> ort::ep::TensorRT {
        // SAFETY: Same lifetime invariant as cuda_provider.
        unsafe { provider.with_compute_stream(self.stream.cast()) }
    }

    pub fn upload<'a>(
        &self,
        pairs: impl IntoIterator<Item = (&'a DynTensor, &'a mut DynTensor)>,
    ) -> Result<()> {
        copy_batch(pairs, self, true)
    }

    pub fn download<'a>(
        &self,
        pairs: impl IntoIterator<Item = (&'a DynTensor, &'a mut DynTensor)>,
    ) -> Result<()> {
        copy_batch(pairs, self, false)
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null()
            && let Ok(scope) = self.enter()
        {
            // SAFETY: The owning session has already been destroyed under the
            // capture coordinator. Drain any outstanding work before destruction.
            unsafe {
                (self.driver.stream_sync)(self.stream);
                (self.driver.stream_destroy)(self.stream);
            }
            let _ = scope.restore();
        }
        // SAFETY: Balances our successful primary-context retain, never resets it.
        unsafe {
            (self.driver.release)(self.device);
        }
    }
}

fn validate(source: &DynTensor, target: &DynTensor, device_id: i32, upload: bool) -> Result<usize> {
    ensure!(
        source.data_type() == target.data_type() && source.shape() == target.shape(),
        "CUDA copy shape/type mismatch"
    );
    let bytes = checked_bytes(source.shape(), source.data_type())?;
    let (host, device) = if upload {
        (source.memory_info(), target.memory_info())
    } else {
        (target.memory_info(), source.memory_info())
    };
    ensure!(
        host.is_cpu_accessible() && host.allocation_device() == AllocationDevice::CUDA_PINNED,
        "Asynchronous CUDA copy requires ORT pinned host memory"
    );
    ensure!(
        device.allocation_device() == AllocationDevice::CUDA && device.device_id() == device_id,
        "CUDA copy device mismatch"
    );
    ensure!(
        bytes == 0 || (!source.data_ptr().is_null() && !target.data_ptr().is_null()),
        "Null CUDA copy buffer"
    );
    Ok(bytes)
}

fn pointer_context(driver: &Driver, pointer: u64) -> Result<CuContext> {
    let mut context: CuContext = ptr::null_mut();
    // SAFETY: Callers pass a validated, live ORT CUDA allocation. Attribute 1 is
    // CU_POINTER_ATTRIBUTE_CONTEXT, whose output is exactly one CUcontext pointer.
    driver.check(
        unsafe { (driver.pointer_attribute)((&mut context as *mut CuContext).cast(), 1, pointer) },
        "cuPointerGetAttribute(CONTEXT)",
    )?;
    ensure!(!context.is_null(), "CUDA tensor has no owning context");
    Ok(context)
}

fn copy_batch<'a>(
    pairs: impl IntoIterator<Item = (&'a DynTensor, &'a mut DynTensor)>,
    stream: &CudaStream,
    upload: bool,
) -> Result<()> {
    let device_id = stream.device_id;
    // Keep every source and exclusive destination borrow alive until the final
    // synchronization, including if a later copy in the batch fails.
    let mut pairs = pairs.into_iter().collect::<Vec<_>>();
    let sizes = pairs
        .iter()
        .map(|(s, t)| validate(s, t, device_id, upload))
        .collect::<Result<Vec<_>>>()?;
    if sizes.iter().all(|&size| size == 0) {
        return Ok(());
    }
    let driver = stream.driver;
    let context = stream.context;
    let scope = stream.enter()?;
    let result = (|| {
        let mut actual_device = -1;
        // SAFETY: Our pushed context is current; actual_device is writable storage.
        driver.check(
            unsafe { (driver.device)(&mut actual_device) },
            "cuCtxGetDevice",
        )?;
        ensure!(
            actual_device == device_id,
            "CUDA allocation context device mismatch"
        );
        for ((source, target), bytes) in pairs.iter_mut().zip(sizes) {
            if bytes == 0 {
                continue;
            }
            let source_ptr = source.data_ptr();
            let target_ptr = target.data_ptr_mut();
            let device_ptr = if upload {
                target_ptr as u64
            } else {
                source_ptr as u64
            };
            ensure!(
                pointer_context(driver, device_ptr)? == context,
                "Mixed CUDA contexts in transfer batch"
            );
            // SAFETY: Validated equal shape/type/size, pinned host memory, live
            // ORT-owned device memory in this context, and exclusive destination.
            // The session stream is valid in the current context. All borrows
            // remain alive until the stream is synchronized below.
            let status = unsafe {
                if upload {
                    (driver.upload)(device_ptr, source_ptr, bytes, stream.stream)
                } else {
                    (driver.download)(target_ptr, device_ptr, bytes, stream.stream)
                }
            };
            driver.check(
                status,
                if upload {
                    "cuMemcpyHtoDAsync"
                } else {
                    "cuMemcpyDtoHAsync"
                },
            )?;
        }
        Ok(())
    })();
    // SAFETY: Complete queued copies even on a partial-batch error before any
    // borrowed tensor can be released or inspected by the caller. Once per batch.
    let completed = driver.check(
        unsafe { (driver.stream_sync)(stream.stream) },
        "cuStreamSynchronize copies",
    );
    let restored = scope.restore();
    result.and(completed).and(restored)
}
