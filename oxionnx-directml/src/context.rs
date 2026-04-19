//! DirectML execution context.
//!
//! `DirectMLContext` is an opaque handle to a DirectML device.  On non-Windows
//! targets `try_new()` always returns `None`.  On Windows targets the initial
//! D3D12 device acquisition is stubbed (returns `None`) until the full HLSL
//! shader pipeline is wired in a later wave.

/// Opaque DirectML execution context.
///
/// On non-Windows platforms this struct cannot be meaningfully constructed;
/// `try_new()` always returns `None`.
pub struct DirectMLContext {
    /// Windows-only inner state.
    #[cfg(target_os = "windows")]
    #[allow(dead_code)]
    inner: windows_impl::WindowsContext,
    /// Non-Windows placeholder — prevents external construction.
    #[cfg(not(target_os = "windows"))]
    _private: (),
}

impl DirectMLContext {
    /// Attempt to initialize a DirectML context.
    ///
    /// Returns `None` when:
    /// - The current platform is not Windows.
    /// - D3D12 is unavailable (old hardware, non-desktop SKU, etc.).
    /// - The DirectML runtime is not installed.
    pub fn try_new() -> Option<Self> {
        #[cfg(target_os = "windows")]
        {
            windows_impl::WindowsContext::try_new().map(|inner| Self { inner })
        }
        #[cfg(not(target_os = "windows"))]
        {
            None
        }
    }

    /// Whether this context is backed by a live DirectML device.
    ///
    /// Always `false` on non-Windows platforms.
    pub fn is_active(&self) -> bool {
        #[cfg(target_os = "windows")]
        {
            self.inner.is_active()
        }
        #[cfg(not(target_os = "windows"))]
        {
            false
        }
    }
}

/// Windows-only D3D12 device state.
///
/// This module is compiled only on Windows targets.  The inner initialisation
/// is intentionally stubbed for this wave — the full D3D12 device/queue/fence
/// setup and HLSL pipeline will be wired in a subsequent phase once the HLSL
/// shader compilation path is in place.
#[cfg(target_os = "windows")]
mod windows_impl {
    // Bring in the D3D12 + DXGI symbols used by the eventual implementation.
    // Even in the stub we import them so that the compiler validates the feature
    // flags are correct and Wave 3 can activate the code without import surgery.
    #[allow(unused_imports)]
    use windows::Win32::{
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_0,
            Direct3D12::{
                D3D12CreateDevice, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
                D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_QUEUE_DESC, D3D12_FENCE_FLAG_NONE,
            },
            Dxgi::{CreateDXGIFactory2, IDXGIFactory4, DXGI_CREATE_FACTORY_DEBUG},
        },
        System::Threading::{CreateEventW, WaitForSingleObject, INFINITE},
    };

    /// Stub Windows context.  Fields will be populated once the HLSL pipeline
    /// lands; for now `try_new` always returns `None`.
    pub(super) struct WindowsContext {
        /// Placeholder — reserved for `ID3D12Device`.
        _reserved: (),
    }

    impl WindowsContext {
        /// Attempt D3D12 device acquisition.
        ///
        /// # Current status
        ///
        /// Returns `None` (stub).  Full DXGI factory + adapter enumeration +
        /// command-queue creation will be implemented in the Wave 3 wiring pass
        /// once the HLSL compute-shader compilation infrastructure is in place.
        pub(super) fn try_new() -> Option<Self> {
            // TODO(Wave3): enumerate DXGI adapters, create D3D12 device,
            // build compute command-queue, allocate fence + event handle.
            None
        }

        /// Returns `true` when backed by a live D3D12 device.
        pub(super) fn is_active(&self) -> bool {
            false
        }
    }
}
