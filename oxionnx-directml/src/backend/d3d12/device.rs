//! D3D12 device, COMPUTE queue, command list, fence, event, descriptor heap.
//!
//! Everything here is thin FFI glue over `d3d12.dll` / `dxgi.dll` / `kernel32.dll`.
//! No shape logic, no sizing, no dispatch math — those live in [`crate::plan`] and
//! [`crate::layout`], which are compiled and unit-tested on every platform.
//!
//! # Threading — why this type is deliberately *not* `Send`/`Sync`
//!
//! `ID3D12Device` and `ID3D12CommandQueue` are free-threaded: the D3D12 spec allows
//! concurrent calls on them from any thread.  `ID3D12CommandAllocator` and
//! `ID3D12GraphicsCommandList` are **not** — recording into one list from two threads,
//! or resetting an allocator while another thread records from it, is undefined
//! behaviour.  [`D3d12Core`] owns one of each, plus a [`core::cell::Cell`] fence
//! counter, so it is `!Send + !Sync` by construction and stays that way.
//!
//! That is the whole justification for `context.rs`'s `Mutex<Backend>`: the mutex is
//! **load-bearing**, not decorative.  It is what makes the
//! `begin → record → Close → ExecuteCommandLists → Signal → wait` sequence atomic with
//! respect to other threads, and it is why `context.rs` (owner B8, not this file) may
//! write `unsafe impl Send + Sync for DirectMLContext`.  Nothing in this module may
//! hand a COM pointer or a `Cell` out past that lock.
//!
//! # Hazards this module owns
//!
//! * **`DXGI_ADAPTER_DESC1::Flags` is a bare `u32`; `DXGI_ADAPTER_FLAG_SOFTWARE` is a
//!   `DXGI_ADAPTER_FLAG(i32)` newtype.**  `desc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE`
//!   does not compile, and the "obvious" fix — comparing against
//!   `DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32` with `==` — compiles and is *wrong*, because
//!   `Flags` is a bitfield.  It must be masked.  Get this wrong and WARP is selected as
//!   a "hardware" adapter, which makes inference slower than the CPU path it replaced.
//! * **The event must be auto-reset** (`bManualReset = false`).  A manual-reset event
//!   that is never reset makes every *subsequent* wait return immediately, silently
//!   reading a buffer the GPU has not finished writing.
//! * **`next_fence_value` starts at 1.**  A fence is born at 0, so signalling 0 would
//!   make every wait a no-op.
//! * **`WaitForSingleObject != WAIT_OBJECT_0` must become an `Err`.**  Ignoring it means
//!   reading a half-written buffer after a device removal.
//! * **`ID3D12CommandAllocator::Reset` is UB while the GPU still owns the list.**  It is
//!   not merely "discouraged"; nothing in D3D12, in the debug layer, or in Rust will
//!   catch it.  [`D3d12Core::begin`] may therefore only run after a
//!   [`D3d12Core::submit_and_wait`] that *returned `Ok`* — which is enforced here by
//!   the `lost` flag rather than left to a comment.
//! * **The descriptor increment is vendor-specific** (32 on most NVIDIA/AMD parts, 64 on
//!   some Intel and on WARP).  Hard-coding it corrupts the heap on exactly one vendor's
//!   hardware.  It is queried from the device once and cached.
//!
//! [`EventHandle`] here, and `MapGuard` / `BarrierRef` in [`super::buffer`], are this
//! crate's paired-FFI RAII guards — the same pattern as `oxionnx-coreml`'s
//! `PixelBufferLockGuard`.  `CreateEventW` and `CloseHandle` can never be separated by an
//! early return.

use core::cell::Cell;

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_0,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue,
    ID3D12DescriptorHeap, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
    D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_FENCE_FLAG_NONE, D3D12_GPU_DESCRIPTOR_HANDLE,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory4, DXGI_ADAPTER_DESC1,
    DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

use crate::error::{DirectMLError, HrExt, Result};

/// Opt-in to the software (WARP) adapter.
///
/// Off by default, and deliberately so: WARP is a CPU rasteriser, so a "GPU" backend
/// running on it would be *slower* than `oxionnx-ops`' tuned CPU kernels — precisely the
/// regression this crate's `Ok(None)` fallback exists to avoid.  A silent WARP fallback
/// would look like a working GPU provider and behave like a performance bug.
///
/// It is honoured only when a human sets it, which is what makes
/// `DirectMLContext::self_check` runnable on a Windows VM that has no D3D12 hardware —
/// the one environment where correctness can be checked without a real GPU.
const ALLOW_SOFTWARE_ADAPTER_ENV: &str = "OXIONNX_DIRECTML_ALLOW_WARP";

/// Feature levels probed, in order, on each candidate adapter.
///
/// 12_0 first: it is what every GPU that DirectML actually targets reports, and asking
/// for it up front keeps us off the ancient-hardware path.  11_0 is a *real* fallback,
/// not a cosmetic one — D3D12 compute (`cs_5_0`, which is what
/// [`super::shader`] compiles) and DirectML both have a documented minimum of feature
/// level 11.0, so a FL11 part is genuinely usable by both engines.  Anything below 11_0
/// cannot create a D3D12 device at all.
const FEATURE_LEVELS: [D3D_FEATURE_LEVEL; 2] = [D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_11_0];

/// The D3D12 foundation, shared verbatim by both engines.
///
/// See the module documentation for why this type is `!Send + !Sync` and why that is the
/// point rather than an inconvenience.
pub(crate) struct D3d12Core {
    /// Free-threaded per the D3D12 spec.
    pub(crate) device: ID3D12Device,
    /// Free-threaded per the D3D12 spec.
    pub(crate) queue: ID3D12CommandQueue,
    /// **Not** thread-safe, and not safe to `Reset` while the GPU owns a list built
    /// from it.  Reachable only behind `context.rs`'s mutex.
    pub(crate) allocator: ID3D12CommandAllocator,
    /// **Not** thread-safe.  Reachable only behind `context.rs`'s mutex.
    pub(crate) list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: EventHandle,
    /// Next value to `Signal`.  **Starts at 1**: a fence is born at 0, so signalling 0
    /// would make every wait return immediately.
    ///
    /// `Cell` because submission goes through `&self` — the whole `Backend` already sits
    /// behind `DirectMLContext`'s mutex, which is what makes this sound.
    next_fence_value: Cell<u64>,
    /// Set between `ExecuteCommandLists` and a *successful* fence wait.
    ///
    /// If a submission is issued and the wait then fails (device removal, a broken
    /// `Signal`), we no longer know when — or whether — the GPU stops referencing the
    /// command allocator.  Resetting it in that state is undefined behaviour that
    /// nothing would catch, so [`D3d12Core::begin`] refuses instead: a permanently
    /// dead context that reports honest errors beats a live one that corrupts memory.
    lost: Cell<bool>,
    /// Cached `GetDescriptorHandleIncrementSize(CBV_SRV_UAV)` — vendor-specific, so it
    /// is queried, never assumed.
    descriptor_increment: u32,
    /// `DXGI_ADAPTER_DESC1::Description`, NUL-trimmed.
    pub(crate) adapter_name: String,
}

impl D3d12Core {
    /// Enumerate DXGI adapters, create a D3D12 device on the first that works, then
    /// build a `D3D12_COMMAND_LIST_TYPE_COMPUTE` queue, an allocator, a **closed**
    /// command list, a fence and an **auto-reset** event.
    ///
    /// Software (WARP) adapters are skipped unless [`ALLOW_SOFTWARE_ADAPTER_ENV`] is set
    /// — see that constant for why a silent WARP fallback would be a performance bug
    /// wearing a GPU costume.
    ///
    /// Returns `None` — never `Err` — when nothing works, so `DirectMLContext::try_new`
    /// can degrade silently to the CPU.  A machine with no D3D12 adapter is not a
    /// failure; it is a machine that runs on the CPU.  **Never panics.**
    pub(crate) fn try_new() -> Option<Self> {
        Self::acquire(software_adapter_allowed()).ok()
    }

    /// The fallible half of [`Self::try_new`], kept separate so every `HRESULT` can be
    /// carried with its call site up to the one place that discards it.
    fn acquire(allow_software: bool) -> Result<Self> {
        // SAFETY: `CreateDXGIFactory2` is a plain `dxgi.dll` export.  Its only
        // preconditions are a valid flags value and a valid IID; the flags are a
        // literal 0 ("no debug layer") and the IID is supplied by the `windows`
        // binding from `IDXGIFactory4`'s type.  The returned interface is a fresh +1
        // reference that `IDXGIFactory4`'s `Drop` releases.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .ctx("CreateDXGIFactory2")?;

        // Keep the *last* real failure so a machine that has adapters but cannot make a
        // device reports why, rather than the generic "no adapter" message.
        let mut last_error: Option<DirectMLError> = None;
        let mut index: u32 = 0;

        loop {
            // SAFETY: `factory` is a live COM pointer; `index` crosses by value.
            // `EnumAdapters1` reports the end of the list with `DXGI_ERROR_NOT_FOUND`,
            // which is this loop's only exit — so the loop is bounded by the adapter
            // count, not by a guess.
            let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
                break;
            };
            index = index.saturating_add(1);

            // SAFETY: `adapter` is the live +1 reference just returned; `GetDesc1`
            // fills a caller-owned `DXGI_ADAPTER_DESC1` by value and borrows nothing.
            let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
                continue;
            };

            if is_software_adapter(&desc) && !allow_software {
                continue;
            }

            match Self::from_adapter(&adapter, &desc) {
                Ok(core) => return Ok(core),
                Err(e) => last_error = Some(e),
            }
        }

        Err(last_error.unwrap_or_else(|| {
            DirectMLError::DeviceInitFailed(format!(
                "no D3D12-capable DXGI adapter found among {index} enumerated \
                 (software adapters are skipped unless {ALLOW_SOFTWARE_ADAPTER_ENV} is set)"
            ))
        }))
    }

    /// Build the whole core on one specific adapter, or explain why it could not be.
    fn from_adapter(adapter: &IDXGIAdapter1, desc: &DXGI_ADAPTER_DESC1) -> Result<Self> {
        let device = create_device(adapter)?;

        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            // COMPUTE, not DIRECT: we never touch the graphics pipeline, and on most
            // parts an async-compute queue is scheduled independently of any graphics
            // work the process is doing elsewhere.
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        // SAFETY: `device` is live; `&queue_desc` coerces to a `*const` that is valid
        // for the duration of the call and is only read by it (D3D12 copies the
        // descriptor).  The IID comes from the requested interface type.
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }
            .ctx("ID3D12Device::CreateCommandQueue")?;

        // SAFETY: `device` is live and the list type matches the queue's, which is what
        // D3D12 requires of an allocator that will feed that queue.
        let allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE) }
                .ctx("ID3D12Device::CreateCommandAllocator")?;

        // SAFETY: `allocator` was just created on `device` with the same list type and
        // is not backing any other command list.  `None` for the initial pipeline state
        // is legal: a compute list with no PSO is valid until the first `Dispatch`, and
        // both engines call `SetPipelineState` before dispatching.
        let list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, None)
        }
        .ctx("ID3D12Device::CreateCommandList")?;

        // A freshly created command list is **open**.  `begin()` starts with
        // `list.Reset(..)`, which requires a *closed* list — so close it now, once, and
        // maintain the invariant "the list is closed whenever no dispatch is in flight"
        // for the rest of its life.
        //
        // SAFETY: the list was created open and nothing has been recorded into it.
        unsafe { list.Close() }.ctx("ID3D12GraphicsCommandList::Close (initial)")?;

        // Initial value 0; `next_fence_value` therefore starts at 1 (see the field).
        // SAFETY: `device` is live; `CreateFence` has no other precondition.
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .ctx("ID3D12Device::CreateFence")?;

        let event = EventHandle::new()?;

        // SAFETY: `device` is live.  `GetDescriptorHandleIncrementSize` is a pure
        // accessor — it cannot fail and returns the value by value.
        let descriptor_increment = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        Ok(Self {
            device,
            queue,
            allocator,
            list,
            fence,
            event,
            next_fence_value: Cell::new(1),
            lost: Cell::new(false),
            descriptor_increment,
            adapter_name: adapter_description(desc),
        })
    }

    /// `allocator.Reset()` then `list.Reset(&allocator, None)`, opening the command list
    /// for a fresh recording.
    ///
    /// # Safety contract (not an `unsafe fn`, but load-bearing)
    ///
    /// May only be called after the *previous* [`Self::submit_and_wait`] returned `Ok`.
    /// Resetting an allocator whose command list the GPU is still executing is undefined
    /// behaviour and nothing will catch it.  That precondition is not left to the
    /// caller's discipline: a submission that did not provably complete sets `lost`, and
    /// this function refuses to run while it is set.
    ///
    /// # Errors
    /// [`DirectMLError::DispatchFailed`] when a previous submission never completed, so
    /// the GPU may still own the allocator.
    /// [`DirectMLError::Win32`] on any failing `HRESULT`.
    pub(crate) fn begin(&self) -> Result<()> {
        if self.lost.get() {
            return Err(DirectMLError::DispatchFailed(
                "a previous D3D12 submission never completed, so the GPU may still own the \
                 command allocator; this device is unusable and every op will fall back to \
                 the CPU"
                    .into(),
            ));
        }

        // SAFETY: `lost` is clear, so the last `submit_and_wait` (if any) observed the
        // fence reach its signalled value before returning — i.e. the GPU has finished
        // every command list ever built from this allocator.  That is exactly
        // `ID3D12CommandAllocator::Reset`'s documented precondition.  Before the first
        // submission it holds vacuously: nothing has been executed.
        unsafe { self.allocator.Reset() }.ctx("ID3D12CommandAllocator::Reset")?;

        // SAFETY: the list is closed — it is closed at creation, and `submit_and_wait`
        // closes it on every submission — and `self.allocator` was just reset and backs
        // no other open list, which is what `ID3D12GraphicsCommandList::Reset` requires.
        // `None` leaves the pipeline state unset; both engines bind a PSO before
        // dispatching.
        unsafe { self.list.Reset(&self.allocator, None) }.ctx("ID3D12GraphicsCommandList::Reset")
    }

    /// Close the command list, execute it on the compute queue, and block until the GPU
    /// has finished it.
    ///
    /// `Close` → `ExecuteCommandLists` → `Signal(fence, v)` → bump → **if**
    /// `GetCompletedValue() < v`, `SetEventOnCompletion(v, event)` +
    /// `WaitForSingleObject(event, INFINITE)`.
    ///
    /// When this returns `Ok`, every write the command list performed is visible to a
    /// subsequent CPU `Map` of a READBACK buffer.  That guarantee is the *only* thing
    /// standing between [`super::buffer::GpuBuffer::read_f32`] and a half-written
    /// buffer, which is why a failed wait is an error and never a warning.
    ///
    /// # Errors
    /// [`DirectMLError::Win32`] on any failing `HRESULT`.
    /// [`DirectMLError::DispatchFailed`] when the wait returns anything other than
    /// `WAIT_OBJECT_0` — typically a device removal.  The core is marked unusable so no
    /// later `begin()` can reset an allocator the GPU may still be reading.
    pub(crate) fn submit_and_wait(&self) -> Result<()> {
        // SAFETY: the list was opened by `begin()`; every recording call since then took
        // it by shared reference on this thread (the whole backend is behind a mutex).
        // Closing an already-closed list returns a failing HRESULT rather than
        // misbehaving, and `.ctx` surfaces it instead of swallowing it.
        unsafe { self.list.Close() }.ctx("ID3D12GraphicsCommandList::Close")?;

        // `ID3D12CommandList` is a base interface of `ID3D12GraphicsCommandList`, so the
        // `From` impl is a transmute of the same interface pointer — not a
        // `QueryInterface`, and not a different object.  `clone()` AddRefs, the
        // temporary slice's `Drop` Releases: net zero refcount change across the call.
        let submitted: ID3D12CommandList = self.list.clone().into();

        // SAFETY: `submitted` is a live, *closed* compute command list allocated from
        // `self.allocator` on `self.device`, and `self.queue` is a COMPUTE queue on that
        // same device — D3D12 requires the list type and the queue type to match.
        // `ExecuteCommandLists` reads the slice for the duration of the call and takes
        // no ownership of it.
        unsafe { self.queue.ExecuteCommandLists(&[Some(submitted)]) };

        // From this instant the GPU owns the allocator.  If anything below fails we
        // never learn when it lets go, so the core becomes permanently unusable rather
        // than a use-after-free waiting for the next `begin()`.
        self.lost.set(true);

        let value = self.next_fence_value.get();

        // SAFETY: `self.fence` was created on `self.device` and `self.queue` belongs to
        // the same device.  `value >= 1 > 0`, the fence's initial value, and the counter
        // only ever increases — so this signal is strictly monotonic, which is what
        // makes `GetCompletedValue() >= value` mean "this submission is done" rather
        // than "some submission is done".
        unsafe { self.queue.Signal(&self.fence, value) }.ctx("ID3D12CommandQueue::Signal")?;
        self.next_fence_value.set(value.saturating_add(1));

        // SAFETY: a plain read of the fence's completed value; no preconditions.
        if unsafe { self.fence.GetCompletedValue() } < value {
            // There is no lost-wakeup race here: if the fence reaches `value` between
            // the read above and this registration, `SetEventOnCompletion` signals the
            // event immediately rather than never.
            //
            // SAFETY: `self.event` is a live auto-reset event owned by `self.event` and
            // outliving this call; the fence is live.  D3D12 signals the handle from a
            // driver thread, which is exactly what a Win32 event is for.
            unsafe { self.fence.SetEventOnCompletion(value, self.event.raw()) }
                .ctx("ID3D12Fence::SetEventOnCompletion")?;

            // SAFETY: the handle is live and was created by `CreateEventW`; `INFINITE`
            // is the documented "wait forever" timeout.  The event is auto-reset, so a
            // successful wait consumes the signal and leaves it non-signalled for the
            // next submission — a manual-reset event here would make every *subsequent*
            // wait return instantly and read half-written buffers.
            let wait = unsafe { WaitForSingleObject(self.event.raw(), INFINITE) };
            if wait != WAIT_OBJECT_0 {
                return Err(DirectMLError::DispatchFailed(format!(
                    "WaitForSingleObject on the D3D12 fence event returned {:#010x}, not \
                     WAIT_OBJECT_0; the GPU may have been removed and any readback would be \
                     half-written",
                    wait.0
                )));
            }
        }

        // The GPU has provably finished with the allocator.
        self.lost.set(false);
        Ok(())
    }

    /// Cached `GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)`.
    ///
    /// Vendor-specific: 32 bytes on most NVIDIA and AMD parts, 64 on some Intel parts and
    /// on WARP.  Hard-coding it corrupts the descriptor heap on exactly one vendor's
    /// hardware, which is the kind of bug that ships.
    pub(crate) fn descriptor_increment(&self) -> u32 {
        self.descriptor_increment
    }
}

/// Try each feature level in [`FEATURE_LEVELS`] on `adapter`, newest first.
fn create_device(adapter: &IDXGIAdapter1) -> Result<ID3D12Device> {
    let mut last: Option<DirectMLError> = None;
    for level in FEATURE_LEVELS {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: `adapter` is a live `IDXGIAdapter1`, which derives from `IUnknown` and
        // so satisfies the parameter's bound.  `device` is a caller-owned `Option` slot
        // that `D3D12CreateDevice` fills with a +1 reference on success and leaves
        // `None` on failure; on success `Option::take` moves that reference out, so it
        // is released exactly once, by `ID3D12Device`'s `Drop`.
        let hr = unsafe { D3D12CreateDevice(adapter, level, &mut device) };
        match hr.ctx("D3D12CreateDevice") {
            Ok(()) => {
                if let Some(device) = device {
                    return Ok(device);
                }
                // A successful HRESULT with a null out-pointer would be a driver bug, but
                // it is representable in this API, so it is reported rather than assumed
                // away.
                last = Some(DirectMLError::DeviceInitFailed(
                    "D3D12CreateDevice reported success but produced no device".into(),
                ));
            }
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| {
        DirectMLError::DeviceInitFailed("D3D12CreateDevice was never attempted".into())
    }))
}

/// Is this the software (WARP) adapter?
///
/// **HAZARD.**  `DXGI_ADAPTER_DESC1::Flags` is a bare `u32` bitfield, while
/// `DXGI_ADAPTER_FLAG_SOFTWARE` is a `DXGI_ADAPTER_FLAG(i32)` newtype.  `desc.Flags ==
/// DXGI_ADAPTER_FLAG_SOFTWARE` does not type-check, and the tempting repair —
/// `desc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32` — compiles, is wrong (other flag
/// bits may be set alongside it), and fails *open*: WARP gets treated as hardware.  The
/// only correct test is a mask.
fn is_software_adapter(desc: &DXGI_ADAPTER_DESC1) -> bool {
    // `DXGI_ADAPTER_FLAG_SOFTWARE.0` is the positive literal 2, so this is a
    // value-preserving widening, not a sign-losing reinterpretation.
    let software_bit = DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32;
    desc.Flags & software_bit != 0
}

/// Read [`ALLOW_SOFTWARE_ADAPTER_ENV`].  Absent, empty or falsey ⇒ hardware only.
fn software_adapter_allowed() -> bool {
    std::env::var(ALLOW_SOFTWARE_ADAPTER_ENV).is_ok_and(|v| {
        let v = v.trim();
        v.eq_ignore_ascii_case("1")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("yes")
    })
}

/// `DXGI_ADAPTER_DESC1::Description` is a fixed 128-`u16` buffer, NUL-padded — not a
/// slice.  Trim at the first NUL before decoding, or the name carries 100 trailing `\0`s
/// into every log line and `SelfCheckReport`.
fn adapter_description(desc: &DXGI_ADAPTER_DESC1) -> String {
    let end = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..end])
        .trim()
        .to_owned()
}

/// RAII for a Win32 event `HANDLE`.
///
/// `CreateEventW` returns a handle that must be `CloseHandle`d exactly once.  Wrapping it
/// means the two calls cannot be separated by an early return, a `?`, or a panic — the
/// same paired-FFI guard pattern as `oxionnx-coreml`'s `PixelBufferLockGuard`.
///
/// The handle is never copied out of the struct except as a by-value `HANDLE` passed
/// *into* a call that does not take ownership of it ([`WaitForSingleObject`],
/// `ID3D12Fence::SetEventOnCompletion`), so there is exactly one owner and exactly one
/// close.
pub(crate) struct EventHandle(HANDLE);

impl EventHandle {
    /// `CreateEventW(None, /* bManualReset */ false, /* bInitialState */ false, None)`.
    ///
    /// **Auto-reset (`bManualReset = false`) is mandatory.**  A manual-reset event that is
    /// never reset stays signalled after the first wait, so *every subsequent* wait
    /// returns immediately — and each of those returns is a readback of a buffer the GPU
    /// has not finished writing.  The failure is silent, data-dependent, and looks like a
    /// numerical bug.
    ///
    /// `bInitialState = false`: the event starts non-signalled, so the very first wait
    /// blocks until the fence actually fires.
    ///
    /// # Errors
    /// [`DirectMLError::Win32`] when `CreateEventW` fails.
    pub(crate) fn new() -> Result<Self> {
        // SAFETY: `None` for `lpEventAttributes` requests the default security
        // descriptor and a non-inheritable handle — which is why this crate needs the
        // `Win32_Security` feature at all, since that parameter is typed as
        // `Option<*const SECURITY_ATTRIBUTES>`.  `None` for `lpName` creates an unnamed
        // event, so there is no chance of colliding with, or inheriting, another
        // process's object.  The returned handle is owned by the `EventHandle` we build
        // from it and closed exactly once, in `Drop`.
        let handle = unsafe { CreateEventW(None, false, false, None) }.ctx("CreateEventW")?;
        Ok(Self(handle))
    }

    /// The raw handle, for `SetEventOnCompletion` / `WaitForSingleObject`.
    ///
    /// Neither call takes ownership, so handing out a `Copy` of the handle does not
    /// create a second owner and cannot lead to a double close.
    pub(crate) fn raw(&self) -> HANDLE {
        self.0
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from a successful `CreateEventW` in `EventHandle::new`
        // (the failure path returns `Err` *before* constructing `Self`, so a handle we
        // never received is never closed).  `EventHandle` is not `Clone` and hands the
        // raw handle only to calls that do not take ownership, so this is the sole owner
        // and `Drop` runs exactly once.  The result is discarded because a failing
        // `CloseHandle` on a handle we know to be valid can only mean the process is
        // already tearing down, and `Drop` has nowhere to report it.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// A shader-visible CBV/SRV/UAV descriptor heap, plus the handle arithmetic D3D12 makes
/// you do by hand.
///
/// **Used only by the DirectML backend.**  The HLSL backend binds root descriptors
/// (`SetComputeRootShaderResourceView` takes a raw GPU virtual address) and needs no heap
/// at all — which is exactly how it sidesteps this entire hazard class.
///
/// # Handle math
///
/// `GetCPUDescriptorHandleForHeapStart()` returns a *byte address*, not an index.  Slot
/// `i` lives at `start.ptr + i * increment`, where `increment` is
/// `GetDescriptorHandleIncrementSize(CBV_SRV_UAV)` **queried from this device**.  It is
/// vendor-specific, so [`Self::cpu_handle`] and [`Self::gpu_handle`] are the only
/// sanctioned way to compute a handle; both bounds-check against `capacity`, because
/// D3D12 will happily let a driver write past the end of the heap.
pub(crate) struct DescriptorHeap {
    heap: ID3D12DescriptorHeap,
    increment: u32,
    capacity: u32,
}

impl DescriptorHeap {
    /// A `SHADER_VISIBLE` CBV/SRV/UAV heap with `capacity` slots.
    ///
    /// `capacity` must be at least the **maximum** of the DirectML operator
    /// initializer's and the compiled operator's `RequiredDescriptorCount` — those two
    /// differ, and sizing from the wrong one overruns the heap.
    ///
    /// # Errors
    /// [`DirectMLError::DispatchFailed`] when `capacity` is 0.
    /// [`DirectMLError::Win32`] when `CreateDescriptorHeap` fails.
    pub(crate) fn new(core: &D3d12Core, capacity: u32) -> Result<Self> {
        if capacity == 0 {
            return Err(DirectMLError::DispatchFailed(
                "a shader-visible descriptor heap needs at least one slot".into(),
            ));
        }

        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: capacity,
            // SHADER_VISIBLE is not optional: DirectML writes its bindings into this
            // heap and the shader it dispatches reads them from the GPU side.  A
            // non-shader-visible heap here is a device removal, not a wrong answer.
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };

        // SAFETY: `core.device` is live and `&desc` coerces to a `*const` valid for the
        // call, which only reads it (D3D12 copies the descriptor).  The IID comes from
        // the requested interface type.
        let heap: ID3D12DescriptorHeap = unsafe { core.device.CreateDescriptorHeap(&desc) }
            .ctx("ID3D12Device::CreateDescriptorHeap")?;

        Ok(Self {
            heap,
            increment: core.descriptor_increment(),
            capacity,
        })
    }

    /// CPU handle of slot `index`.
    ///
    /// # Errors
    /// [`DirectMLError::DispatchFailed`] when `index >= capacity`, or if the handle
    /// arithmetic overflows.
    pub(crate) fn cpu_handle(&self, index: u32) -> Result<D3D12_CPU_DESCRIPTOR_HANDLE> {
        let offset = self.byte_offset(index)?;
        // SAFETY: `self.heap` is live.  `GetCPUDescriptorHandleForHeapStart` is a pure
        // accessor returning a by-value handle; it cannot fail and borrows nothing.
        let start = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
        let ptr = start.ptr.checked_add(offset).ok_or_else(|| {
            DirectMLError::DispatchFailed("descriptor-heap CPU handle arithmetic overflowed".into())
        })?;
        Ok(D3D12_CPU_DESCRIPTOR_HANDLE { ptr })
    }

    /// GPU handle of slot `index`; same bounds check, same device-queried increment.
    ///
    /// # Errors
    /// [`DirectMLError::DispatchFailed`] when `index >= capacity`, or if the handle
    /// arithmetic overflows.
    pub(crate) fn gpu_handle(&self, index: u32) -> Result<D3D12_GPU_DESCRIPTOR_HANDLE> {
        if index >= self.capacity {
            return Err(self.out_of_range(index));
        }
        // SAFETY: `self.heap` is live.  `GetGPUDescriptorHandleForHeapStart` is a pure
        // accessor returning a by-value handle; it cannot fail and borrows nothing.  It
        // is only meaningful on a SHADER_VISIBLE heap, which `new` guarantees.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        let offset = u64::from(index)
            .checked_mul(u64::from(self.increment))
            .and_then(|offset| start.ptr.checked_add(offset))
            .ok_or_else(|| {
                DirectMLError::DispatchFailed(
                    "descriptor-heap GPU handle arithmetic overflowed".into(),
                )
            })?;
        Ok(D3D12_GPU_DESCRIPTOR_HANDLE { ptr: offset })
    }

    /// For `SetDescriptorHeaps` and `DML_BINDING_TABLE_DESC`.
    pub(crate) fn raw(&self) -> &ID3D12DescriptorHeap {
        &self.heap
    }

    /// Slot count this heap was created with.
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Byte offset of slot `index` from the heap start, bounds-checked.
    fn byte_offset(&self, index: u32) -> Result<usize> {
        if index >= self.capacity {
            return Err(self.out_of_range(index));
        }
        // Both operands are `u32`; on every target this crate builds for, `usize` is at
        // least 32 bits, so these are value-preserving widenings.  The multiply is still
        // checked, because a corrupt `increment` should surface as an error rather than
        // as a wrapped offset into somebody else's memory.
        let index = index as usize;
        let increment = self.increment as usize;
        index.checked_mul(increment).ok_or_else(|| {
            DirectMLError::DispatchFailed("descriptor-heap offset arithmetic overflowed".into())
        })
    }

    fn out_of_range(&self, index: u32) -> DirectMLError {
        DirectMLError::DispatchFailed(format!(
            "descriptor slot {index} is out of range for a {}-slot heap",
            self.capacity
        ))
    }
}
