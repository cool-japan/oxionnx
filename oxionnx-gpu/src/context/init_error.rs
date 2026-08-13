//! Why GPU initialization failed, and — on the one platform where the usual
//! cause is a missing *system package* rather than missing hardware — what to
//! install.
//!
//! # The failure this module exists for
//!
//! [`crate::GpuContext::try_new`] returns `Option`, and `None` has always meant
//! "run on the CPU". That is the right contract: a machine with no GPU must
//! still run the model. But it collapses two very different situations into one
//! silent answer:
//!
//! * this machine genuinely has no GPU, and
//! * this machine has a perfectly good GPU with its kernel driver loaded, and
//!   `wgpu` cannot reach it because one 128 KB system package is absent.
//!
//! The second is the normal state of a minimal Linux container. `wgpu` reaches
//! a Vulkan GPU through the **Vulkan loader** — `libvulkan.so.1`, shipped as
//! `libvulkan1` on Debian/Ubuntu, `vulkan-loader` on Fedora/RHEL/Alpine and
//! `vulkan-icd-loader` on Arch. The loader is what reads the installable client
//! driver (ICD) manifests in `/usr/share/vulkan/icd.d/` and dlopens the vendor
//! driver behind them. NVIDIA's own driver package installs the ICD manifest
//! and the driver library but **not** the loader — it is a separate package,
//! and nothing pulls it in on a headless image.
//!
//! Reproduced on this crate's reference box (RTX A4000, driver 550.144.03,
//! Ubuntu 22.04 container): with `/usr/share/vulkan/icd.d/nvidia_icd.json`
//! present and `nvidia-smi` fully working, `wgpu` enumerated **zero** Vulkan
//! adapters and `GpuContext::try_new()` returned a bare `None`. `apt-get
//! install libvulkan1` — one package, no reboot, no driver change — turned the
//! same call into two Vulkan adapters and a working context.
//!
//! # How the diagnostic knows
//!
//! It does not guess from file paths. On failure,
//! [`GpuInitDiagnostic::probe`] re-enumerates adapters across *every* backend
//! `wgpu` was built with. If a non-Vulkan backend (in practice OpenGL, which
//! NVIDIA's driver package does install a loader for) reports a real GPU while
//! Vulkan reports none, then the hardware and its kernel driver are demonstrably
//! present and the Vulkan path specifically is broken — which on Linux is the
//! missing loader, essentially always. That is evidence, not inference from a
//! guessed filename, and it costs nothing on the success path because it only
//! runs after `request_adapter` has already failed.

use std::fmt;

/// Why a [`crate::GpuContext`] could not be created.
///
/// Every variant is a *decline*, not an error in the engine sense — the caller
/// runs the CPU operators. It exists so that decline can be explained rather
/// than merely observed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuInitError {
    /// `request_adapter` found nothing on the requested backends.
    NoAdapter {
        /// What was asked for.
        backends: String,
        /// What the cross-backend probe found, if anything.
        diagnostic: GpuInitDiagnostic,
    },
    /// An adapter exists, but every `request_device` attempt was refused —
    /// including the fallback to `wgpu::Limits::default()` with no optional
    /// features.
    NoDevice {
        /// The adapter that refused, for the report.
        adapter: String,
    },
    /// A device was acquired but a compute pipeline could not be built.
    PipelineBuild,
    /// This target cannot acquire an adapter synchronously.
    ///
    /// wasm32 only: a browser thread may not block on `requestAdapter`'s
    /// promise. Use [`crate::GpuContext::try_new_async`].
    BlockingUnavailable,
}

/// What a cross-backend adapter probe saw after Vulkan came up empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuInitDiagnostic {
    /// Another backend reports a real GPU that Vulkan cannot see. On Linux this
    /// is the missing-Vulkan-loader signature: the kernel driver is loaded (the
    /// other backend is talking to it) and only the Vulkan userspace entry
    /// point is absent.
    GpuVisibleToAnotherBackend {
        /// The adapter that other backend reported, e.g.
        /// `"NVIDIA RTX A4000/PCIe/SSE2 (Gl)"`.
        adapter: String,
    },
    /// No backend reported anything but a software rasterizer.
    OnlySoftwareAdapters {
        /// The software adapter found, for the report.
        adapter: String,
    },
    /// No backend reported any adapter at all.
    NoAdaptersAnywhere,
    /// The probe was not run (not compiled for this target).
    NotProbed,
}

impl GpuInitDiagnostic {
    /// Re-enumerate adapters across every backend this build of `wgpu`
    /// supports, to distinguish "no GPU here" from "GPU here, wrong backend".
    ///
    /// Only ever called after an adapter request has already failed, so its
    /// cost — a second instance and a full enumeration — is off the hot path
    /// entirely, and **memoized** on top of that: on a machine with no GPU
    /// *every* `try_new` fails, and a caller that constructs a context per
    /// session would otherwise pay a full all-backend enumeration (which on
    /// Linux includes bringing up an EGL/GL context) every time. What backends
    /// exist does not change within a process, so the first answer is the only
    /// one needed.
    ///
    /// A race between two first callers is harmless: both probe, both compute
    /// the same answer, one `set` wins.
    ///
    /// `requested` is the backend set that already failed; it is **excluded**
    /// from the enumeration, so `GpuVisibleToAnotherBackend` means literally
    /// that and cannot be satisfied by the very backend whose absence is being
    /// explained.
    #[must_use]
    pub async fn probe(requested: wgpu::Backends) -> Self {
        static CACHED: std::sync::OnceLock<GpuInitDiagnostic> = std::sync::OnceLock::new();
        if let Some(cached) = CACHED.get() {
            return cached.clone();
        }
        let probed = Self::probe_uncached(requested).await;
        CACHED.get_or_init(|| probed).clone()
    }

    /// [`Self::probe`] without the memoization.
    async fn probe_uncached(requested: wgpu::Backends) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            // `enumerate_adapters` is not available on the web backend, and
            // there is exactly one backend there anyway.
            let _ = requested;
            Self::NotProbed
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let others = wgpu::Backends::all() - requested;
            if others.is_empty() {
                return Self::NoAdaptersAnywhere;
            }
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: others,
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapters = instance.enumerate_adapters(others).await;
            let mut software = None;
            for adapter in &adapters {
                let info = adapter.get_info();
                let described = format!("{} ({:?})", info.name, info.backend);
                if info.device_type == wgpu::DeviceType::Cpu {
                    software.get_or_insert(described);
                } else {
                    return Self::GpuVisibleToAnotherBackend { adapter: described };
                }
            }
            match software {
                Some(adapter) => Self::OnlySoftwareAdapters { adapter },
                None => Self::NoAdaptersAnywhere,
            }
        }
    }
}

/// The package that ships the Vulkan loader, per distribution family.
///
/// Named rather than inlined so the list is one thing to keep current, and so
/// the test that pins the message can point at it.
const VULKAN_LOADER_PACKAGES: &str =
    "`libvulkan1` (Debian/Ubuntu), `vulkan-loader` (Fedora/RHEL/Alpine), \
     `vulkan-icd-loader` (Arch)";

impl fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter {
                backends,
                diagnostic,
            } => {
                write!(f, "no GPU adapter on backends {backends}")?;
                match diagnostic {
                    GpuInitDiagnostic::GpuVisibleToAnotherBackend { adapter } => {
                        write!(
                            f,
                            ": a GPU *is* present and its driver *is* loaded — another \
                             backend reports {adapter} — so the requested backend's \
                             userspace entry point is what is missing"
                        )?;
                        if cfg!(target_os = "linux") {
                            write!(
                                f,
                                ". On Linux that is the Vulkan loader (`libvulkan.so.1`): \
                                 install {VULKAN_LOADER_PACKAGES}. The GPU vendor's driver \
                                 package installs the ICD manifest \
                                 (/usr/share/vulkan/icd.d/) and the driver library but not \
                                 the loader, so a minimal container image has a fully \
                                 working GPU that Vulkan cannot reach"
                            )?;
                        }
                        Ok(())
                    }
                    GpuInitDiagnostic::OnlySoftwareAdapters { adapter } => write!(
                        f,
                        ": only a software rasterizer was found ({adapter}); no hardware \
                         GPU is reachable from this process"
                    ),
                    GpuInitDiagnostic::NoAdaptersAnywhere => {
                        write!(f, ": no adapter on any backend")?;
                        if cfg!(target_os = "linux") {
                            write!(
                                f,
                                ". If this machine does have a GPU, check that the Vulkan \
                                 loader ({VULKAN_LOADER_PACKAGES}) and the vendor's ICD \
                                 manifest (/usr/share/vulkan/icd.d/) are both installed, \
                                 and that the container was started with the GPU devices \
                                 passed through"
                            )?;
                        }
                        Ok(())
                    }
                    GpuInitDiagnostic::NotProbed => Ok(()),
                }
            }
            Self::NoDevice { adapter } => write!(
                f,
                "adapter {adapter} refused every device request, including the \
                 conservative fallback (default limits, no optional features)"
            ),
            Self::PipelineBuild => {
                write!(f, "a compute pipeline could not be built on this device")
            }
            Self::BlockingUnavailable => write!(
                f,
                "a GPU context cannot be acquired synchronously on this target; \
                 use GpuContext::try_new_async"
            ),
        }
    }
}

impl std::error::Error for GpuInitError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the diagnostic: the operator must be told which
    /// package to install, not merely that there is no GPU.
    #[test]
    fn a_gpu_on_another_backend_names_the_loader_package() {
        let err = GpuInitError::NoAdapter {
            backends: "Backends(VULKAN)".into(),
            diagnostic: GpuInitDiagnostic::GpuVisibleToAnotherBackend {
                adapter: "NVIDIA RTX A4000/PCIe/SSE2 (Gl)".into(),
            },
        };
        let text = err.to_string();
        assert!(text.contains("NVIDIA RTX A4000"), "{text}");
        assert!(text.contains("driver *is* loaded"), "{text}");
        if cfg!(target_os = "linux") {
            assert!(text.contains("libvulkan1"), "{text}");
            assert!(text.contains("vulkan-loader"), "{text}");
            assert!(text.contains("icd.d"), "{text}");
        }
    }

    #[test]
    fn no_adapter_anywhere_still_points_at_the_two_things_to_check() {
        let err = GpuInitError::NoAdapter {
            backends: "Backends(VULKAN)".into(),
            diagnostic: GpuInitDiagnostic::NoAdaptersAnywhere,
        };
        let text = err.to_string();
        assert!(text.contains("no adapter on any backend"), "{text}");
        if cfg!(target_os = "linux") {
            assert!(text.contains("libvulkan1"), "{text}");
        }
    }

    /// A software rasterizer is a different problem with a different answer:
    /// nothing to install, the GPU simply is not reachable. It must not be
    /// reported as a missing loader.
    #[test]
    fn a_software_adapter_is_not_reported_as_a_missing_package() {
        let err = GpuInitError::NoAdapter {
            backends: "Backends(VULKAN)".into(),
            diagnostic: GpuInitDiagnostic::OnlySoftwareAdapters {
                adapter: "llvmpipe (Vulkan)".into(),
            },
        };
        let text = err.to_string();
        assert!(text.contains("software rasterizer"), "{text}");
        assert!(!text.contains("libvulkan1"), "{text}");
    }

    #[test]
    fn the_other_variants_say_what_failed() {
        assert!(GpuInitError::NoDevice {
            adapter: "some adapter".into()
        }
        .to_string()
        .contains("refused every device request"));
        assert!(GpuInitError::PipelineBuild
            .to_string()
            .contains("compute pipeline"));
        assert!(GpuInitError::BlockingUnavailable
            .to_string()
            .contains("try_new_async"));
    }
}
