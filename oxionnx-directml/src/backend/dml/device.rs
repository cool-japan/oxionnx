//! `IDMLDevice` acquisition — via `LoadLibraryW` + `GetProcAddress`, never a static
//! import.
//!
//! # Why `GetProcAddress` and not the `windows` crate's `DMLCreateDevice`
//!
//! `windows_core::link!` expands to `#[link(kind = "raw-dylib")] extern "system" { … }`,
//! i.e. a normal IAT import that the **loader** resolves at *process start*.  If
//! `DirectML.dll` is absent — Server Core, an older LTSB build, Wine, a Windows 10 build
//! before 1903 — the host process would fail to **launch**.  It would never reach the
//! HLSL fallback that this crate's entire architecture is built around.  Resolving this
//! one symbol dynamically is what makes "DirectML runtime not installed → use HLSL
//! compute shaders" a *reachable code path* rather than a comment.
//!
//! Every `IDML*` **method** is a plain vtable call and imports nothing, so the rest of
//! the DirectML surface stays statically typed and zero-cost.  It is only the one
//! entry-point function that must be looked up.
//!
//! # The ABI, and where it came from
//!
//! `DirectML.h` (Windows SDK) declares:
//!
//! ```text
//! STDAPI DMLCreateDevice(
//!     ID3D12Device* d3d12Device,
//!     DML_CREATE_DEVICE_FLAGS flags,
//!     REFIID riid,
//!     _COM_Outptr_opt_ void** ppv);
//! ```
//!
//! * `STDAPI` expands to `EXTERN_C HRESULT STDAPICALLTYPE`, and `STDAPICALLTYPE` is
//!   `__stdcall`.  Rust's `extern "system"` *is* `__stdcall` on 32-bit x86 and collapses
//!   to the single platform convention on x64 / ARM64 — so `extern "system"` is the
//!   correct spelling on every Windows target, and `extern "C"` would be wrong on x86.
//! * `ID3D12Device*` crosses the ABI as a plain `*mut c_void`; it is an **[in]**
//!   parameter, so the callee does not take ownership and we must not `AddRef`.
//! * `REFIID` is `const IID&`, i.e. `*const GUID`.
//! * `void** ppv` is a **[out]** COM pointer: on `S_OK` the callee has written an
//!   already-`AddRef`'d interface pointer, and the caller owns that reference.
//!
//! This is cross-checked, field for field, against `windows-0.62.2`'s own generated
//! declaration in `Win32/AI/MachineLearning/DirectML/mod.rs`:
//!
//! ```text
//! link!("directml.dll" "system" fn DMLCreateDevice(
//!     d3d12device: *mut core::ffi::c_void,
//!     flags: DML_CREATE_DEVICE_FLAGS,
//!     riid: *const windows_core::GUID,
//!     ppv: *mut *mut core::ffi::c_void) -> windows_core::HRESULT);
//! ```
//!
//! [`DmlCreateDeviceFn`] below is that signature, transcribed.  **Nothing checks this at
//! compile time** — a `transmute`d function pointer is exactly as correct as the human
//! who wrote it.  If it is wrong, the failure mode is a corrupted stack at the first
//! call, on a machine we cannot test from here.
//!
//! # The module handle's lifetime
//!
//! The `HMODULE` from `LoadLibraryW` is deliberately **never** `FreeLibrary`'d, and it is
//! deliberately **not stored**.
//!
//! `LoadLibraryW` increments the DLL's reference count; `FreeLibrary` decrements it, and
//! at zero the loader unmaps the image.  Every `IDMLDevice`, `IDMLCompiledOperator` and
//! `IDMLBindingTable` we create has its vtable *inside that image*, and DirectML's
//! internal state (its own D3D12 objects, its shader caches) lives there too.  Those COM
//! objects outlive this function by design — [`crate::DirectMLContext`] holds them for
//! the whole process — and a `FreeLibrary` while any of them is alive unmaps the code
//! their vtables point at.  There is no ordering we could impose that makes an
//! unload-on-`Drop` safe, because `IDMLDevice` is `Clone`-able and can be held anywhere.
//!
//! So: exactly **one** reference on `DirectML.dll` is taken, once per process
//! ([`entry_point`]'s `OnceLock`), and it is never released.  That is a bounded,
//! intentional leak of one module reference for the process lifetime — the same bargain
//! every `LoadLibrary`-based plugin host makes.  The `OnceLock` is load-bearing: calling
//! `LoadLibraryW` once per [`DmlDevice::load`] would leak one *additional* reference per
//! call, and would also re-walk the loader's search path on every failed attempt.

use core::ffi::c_void;
use std::sync::OnceLock;

use windows::core::{s, w, Interface, GUID, HRESULT};
use windows::Win32::Foundation::{
    ERROR_FILE_NOT_FOUND, ERROR_MOD_NOT_FOUND, ERROR_PATH_NOT_FOUND, HMODULE,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::AI::MachineLearning::DirectML::{
    IDMLCommandRecorder, IDMLDevice, DML_CREATE_DEVICE_FLAGS, DML_CREATE_DEVICE_FLAG_NONE,
};

use crate::backend::d3d12::device::D3d12Core;
use crate::error::{DirectMLError, HrExt, Result};

/// `DMLCreateDevice`'s raw ABI, hand-declared and resolved at run time.
///
/// See this module's documentation for the header it was transcribed from and for why
/// the calling convention is `system` rather than `C`.
type DmlCreateDeviceFn = unsafe extern "system" fn(
    d3d12_device: *mut c_void,
    flags: DML_CREATE_DEVICE_FLAGS,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT;

/// The outcome of trying to resolve `DMLCreateDevice`, cached for the process lifetime.
///
/// `Copy` so it can live in a `OnceLock` and be read without cloning; a function pointer
/// is `Copy`, `Send` and `Sync`, and an `HRESULT` is a plain `i32`.
///
/// The three failure arms are kept apart because they mean genuinely different things and
/// [`DmlDevice::load`] answers them differently: [`Self::NotInstalled`] is the *expected*
/// state on a large fraction of Windows machines and is not an error, while the other two
/// mean a `DirectML.dll` **is** there and is broken.
#[derive(Clone, Copy)]
enum DmlEntryPoint {
    /// `DirectML.dll` is not on this machine (the loader could not find the file).
    ///
    /// **Not a failure.**  It is the documented signal to run the HLSL compute engine.
    NotInstalled,
    /// A `DirectML.dll` exists but the loader refused it — wrong architecture, corrupt
    /// image, a failing `DllMain`, a missing transitive dependency.  A real fault.
    LoadFailed(HRESULT),
    /// The DLL loaded but does not export `DMLCreateDevice`.  Whatever that file is, it
    /// is not the DirectML runtime.  A real fault.
    MissingExport,
    /// Resolved.
    Resolved(DmlCreateDeviceFn),
}

/// Resolve `DMLCreateDevice` exactly once per process.
///
/// See this module's documentation for why the resulting `HMODULE` is never released and
/// why the `OnceLock` is not merely an optimisation.
fn entry_point() -> DmlEntryPoint {
    static RESOLVED: OnceLock<DmlEntryPoint> = OnceLock::new();
    *RESOLVED.get_or_init(resolve_entry_point)
}

/// The uncached resolution.  Called at most once, through [`entry_point`].
fn resolve_entry_point() -> DmlEntryPoint {
    // SAFETY: `w!` expands to a `'static`, NUL-terminated UTF-16 literal, which is
    // precisely the `LPCWSTR` contract `LoadLibraryW` documents for `lpLibFileName` — it
    // reads forward until the terminator, and the terminator is in the literal.  The name
    // is a bare file name, so the loader applies its standard search order; we
    // deliberately do not pass an absolute path, because DirectML may legitimately be
    // either the inbox system copy or a redistributable next to the executable.  On
    // failure `LoadLibraryW` returns a null `HMODULE`, which the `windows` wrapper turns
    // into `Err(Error::from_thread())` rather than handing us the null — so there is no
    // invalid handle to mishandle here.
    let module: HMODULE = match unsafe { LoadLibraryW(w!("DirectML.dll")) } {
        Ok(module) => module,
        Err(err) => {
            let code = err.code();
            // "The file is not there" is the one outcome that is not a fault.  Everything
            // else — a bad image, a failed `DllMain`, an unresolvable dependency — means a
            // DirectML.dll IS present and is broken, and must be loud.
            return if code == HRESULT::from_win32(ERROR_MOD_NOT_FOUND.0)
                || code == HRESULT::from_win32(ERROR_FILE_NOT_FOUND.0)
                || code == HRESULT::from_win32(ERROR_PATH_NOT_FOUND.0)
            {
                DmlEntryPoint::NotInstalled
            } else {
                DmlEntryPoint::LoadFailed(code)
            };
        }
    };

    // SAFETY: `module` is a live module handle from a successful `LoadLibraryW` on the
    // line above, and it is never freed (see the module docs), so it cannot be stale.
    // `s!` expands to a `'static`, NUL-terminated ASCII literal, which is the `LPCSTR`
    // contract for `lpProcName`.  `GetProcAddress` returns `FARPROC`, which the `windows`
    // crate models as `Option<unsafe extern "system" fn() -> isize>`: `None` for the
    // null return, so a missing export cannot be mistaken for a valid address.
    let Some(symbol) = (unsafe { GetProcAddress(module, s!("DMLCreateDevice")) }) else {
        return DmlEntryPoint::MissingExport;
    };

    // SAFETY: `symbol` is a non-null code address that the DLL's export table maps to the
    // name `DMLCreateDevice`.  `FARPROC`'s `unsafe extern "system" fn() -> isize` is a
    // placeholder shape, not the real one — Win32 has no way to type an export — so the
    // *only* way to call it is to restate the true signature and transmute.  Both sides
    // of this transmute are bare function pointers: same size, same alignment, no
    // provenance to lose.  The correctness of the *call* rests entirely on
    // `DmlCreateDeviceFn` matching `DirectML.h`, which is argued in the module docs and
    // cross-checked against `windows-0.62.2`'s own generated `link!` declaration.  It is
    // checked by nothing at compile time.
    let create: DmlCreateDeviceFn = unsafe {
        core::mem::transmute::<unsafe extern "system" fn() -> isize, DmlCreateDeviceFn>(symbol)
    };
    DmlEntryPoint::Resolved(create)
}

/// Build a [`DirectMLError::Win32`] from an `HRESULT` we are already holding.
///
/// [`HrExt`] covers the common case of a `windows::core::Result`; this covers the one
/// place where the `HRESULT` was cached (in [`DmlEntryPoint::LoadFailed`]) and the
/// original `Error` is long gone.  The call site is still named, and the system's message
/// is still recovered, so the two paths produce identical diagnostics.
fn win32_error(context: &'static str, hresult: HRESULT) -> DirectMLError {
    DirectMLError::Win32 {
        context,
        // An HRESULT is a bit pattern, not a magnitude: 0x8007007E is *the* value, and
        // printing it as -2147024770 helps nobody.  Not a shape-derived quantity.
        hresult: hresult.0 as u32,
        message: windows::core::Error::from_hresult(hresult).message(),
    }
}

/// The DirectML device, plus the command recorder reused across every dispatch.
///
/// Both are ordinary COM smart pointers: dropping this struct releases them.  The
/// `DirectML.dll` module reference is *not* held here — see the module docs.
pub(crate) struct DmlDevice {
    /// The `IDMLDevice` created on top of B2's `ID3D12Device`.
    pub(crate) device: IDMLDevice,
    /// Created once and reused: `IDMLCommandRecorder` is stateless with respect to the
    /// command list it records into, so one per device is enough.
    pub(crate) recorder: IDMLCommandRecorder,
}

impl DmlDevice {
    /// Acquire DirectML, reporting the three outcomes honestly.
    ///
    /// * `Ok(Some(device))` — `DirectML.dll` resolved and an `IDMLDevice` was created on
    ///   top of `core.device`.
    /// * `Ok(None)` — **`DirectML.dll` is not installed on this machine.**  This is *not*
    ///   a failure: it is the documented signal to fall back to
    ///   [`crate::backend::d3d12::hlsl_backend::HlslEngine`], and it is the expected state
    ///   on every Windows build before 10/1903 and on SKUs that omit the runtime.
    ///   Reporting it as an `Err` would turn a supported configuration into a logged
    ///   fault.
    /// * `Err(_)` — a `DirectML.dll` **is** present and something went genuinely wrong:
    ///   the loader rejected the image, the file does not export `DMLCreateDevice`, or
    ///   `DMLCreateDevice` / `CreateCommandRecorder` returned a failing `HRESULT`.  That
    ///   is a real fault and must not be silently swallowed.
    ///
    /// # Errors
    /// [`DirectMLError::Win32`] carrying the failing call's name and `HRESULT`, or
    /// [`DirectMLError::DeviceInitFailed`] when DirectML violates its own contract
    /// (a missing export, or `S_OK` with a null out-pointer).
    pub(crate) fn load(core: &D3d12Core) -> Result<Option<Self>> {
        let create = match entry_point() {
            DmlEntryPoint::NotInstalled => return Ok(None),
            DmlEntryPoint::LoadFailed(hresult) => {
                return Err(win32_error("LoadLibraryW(\"DirectML.dll\")", hresult))
            }
            DmlEntryPoint::MissingExport => {
                return Err(DirectMLError::DeviceInitFailed(
                    "DirectML.dll loaded but does not export DMLCreateDevice; the file on \
                     this machine is not the DirectML runtime"
                        .into(),
                ))
            }
            DmlEntryPoint::Resolved(create) => create,
        };

        let mut raw_device: *mut c_void = core::ptr::null_mut();

        // SAFETY: four obligations, one per argument, plus the signature itself.
        //
        // 1. `create` is the address the DLL's export table gives for `DMLCreateDevice`,
        //    and `DmlCreateDeviceFn` restates that function's declaration from
        //    `DirectML.h` (module docs).  Calling through it is sound exactly insofar as
        //    that transcription is right; nothing here can verify it.
        // 2. `core.device.as_raw()` yields the `ID3D12Device`'s interface pointer without
        //    transferring ownership.  `d3d12Device` is an [in] parameter — DirectML
        //    `AddRef`s it itself if it wants to keep it — so we neither `AddRef` nor
        //    `Release`, and `core` outlives this call, so the pointer is live.
        // 3. `&<IDMLDevice as Interface>::IID` is a reference to a `'static` GUID const,
        //    which coerces to the `*const GUID` the `REFIID` parameter expects.
        // 4. `&mut raw_device` coerces to `*mut *mut c_void`, points at a live local, and
        //    is the [out] slot.  DirectML writes an already-`AddRef`'d pointer there on
        //    success and leaves it null on failure.
        let hresult = unsafe {
            create(
                core.device.as_raw(),
                DML_CREATE_DEVICE_FLAG_NONE,
                &<IDMLDevice as Interface>::IID,
                &mut raw_device,
            )
        };

        // Never swallowed: a failing HRESULT becomes a `DirectMLError::Win32` naming the
        // call, which propagates out of `load` and is a real, loud error.
        hresult.ok().ctx("DMLCreateDevice")?;

        if raw_device.is_null() {
            // S_OK with a null out-pointer would violate `_COM_Outptr_`'s contract.  It
            // must never happen; if it does, `IDMLDevice::from_raw(null)` below would
            // manufacture a null COM pointer that crashes on its first vtable call, far
            // from here.  Refuse instead.
            return Err(DirectMLError::DeviceInitFailed(
                "DMLCreateDevice returned S_OK but wrote a null IDMLDevice pointer".into(),
            ));
        }

        // SAFETY: `raw_device` is non-null (checked above) and, because `DMLCreateDevice`
        // returned S_OK, is a valid `IDMLDevice` interface pointer with **one reference
        // already taken on our behalf** — that is what `_COM_Outptr_` means.
        // `Interface::from_raw` *takes ownership* of exactly that reference, so we must
        // not `AddRef` it here, and the eventual `Drop` of `device` releases it exactly
        // once.  The GUID we passed was `IDMLDevice`'s own, so the vtable matches the
        // type we are constructing.
        let device: IDMLDevice = unsafe { IDMLDevice::from_raw(raw_device) };

        // SAFETY: `device` is a live `IDMLDevice` constructed on the line above.
        // `CreateCommandRecorder` is a plain vtable call with no preconditions beyond
        // that; the `windows` wrapper allocates the out-pointer and hands back an owned
        // `IDMLCommandRecorder`, so there is no manual refcounting to get wrong.
        let recorder: IDMLCommandRecorder =
            unsafe { device.CreateCommandRecorder() }.ctx("IDMLDevice::CreateCommandRecorder")?;

        Ok(Some(Self { device, recorder }))
    }

    /// The degrading adapter [`crate::backend::dml::dml_backend::DmlEngine::new`] calls.
    ///
    /// `None` means "run the HLSL engine instead", for either of the two reasons
    /// [`Self::load`] distinguishes.  **Never panics.**
    ///
    /// The distinction is not thrown away: a *missing* `DirectML.dll` degrades in
    /// silence, because that is a supported configuration and not a fault, whereas a
    /// `DirectML.dll` that is **present and broken** is reported on stderr before we
    /// degrade.  This crate has no logging facility in its dependency set (it depends on
    /// `oxionnx-core` and `thiserror`, and nothing else), so stderr is the only loud
    /// channel available; the alternative — returning `None` in both cases — would make a
    /// broken DirectML installation indistinguishable from an absent one, and users would
    /// silently run the slower engine forever with no way to find out why.
    ///
    /// Callers that want to *handle* the fault rather than merely hear about it should
    /// call [`Self::load`], which returns it.
    pub(crate) fn try_new(core: &D3d12Core) -> Option<Self> {
        match Self::load(core) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "oxionnx-directml: DirectML is installed but unusable; falling back to \
                     HLSL compute shaders. Cause: {err}"
                );
                None
            }
        }
    }
}
