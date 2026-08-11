//! Routes every [`GpuContext`] creation *and* destruction through one
//! dedicated, process-lifetime background thread.
//!
//! # The crash this closes
//!
//! Caught live under gdb, twice, in two different shapes that turned out to
//! be the same root cause: a `SIGSEGV` deep inside NVIDIA's closed-source
//! driver (`libnvidia-eglcore.so`), reached from
//! `<wgpu_hal::vulkan::Device as DynDevice>::destroy_fence`, itself reached
//! from `Session`'s `Drop` chain (`Arc<Session>::drop_slow` →
//! `Option<GpuContext>`'s drop glue → `wgpu_core::device::resource::Device`'s
//! own `Drop`) — running on whatever thread happened to drop the last
//! `Arc<Session>`, which is not necessarily, and in the failing tests
//! specifically was not, the thread that had originally called
//! `GpuContext::try_new()` while building the session.
//!
//! * Sometimes that `SIGSEGV` simply kills the process outright.
//! * Sometimes — caught this way — the crashed thread was holding a *global*
//!   NVIDIA driver mutex (reached via `libnvidia-glsi.so`'s `_nv004glsi`)
//!   at the moment it crashed, so the mutex is never released. If the *main*
//!   thread's own exit path also needs that mutex — and it does: glibc's
//!   `exit()` runs `libEGL_nvidia.so`'s registered `atexit` handler, which
//!   calls back into the same driver core and blocks on the same lock — the
//!   whole process hangs forever on an abandoned `pthread_mutex_lock`, not
//!   an infinite loop. This is indistinguishable from the outside from a
//!   genuine deadlock in this crate's own code, which is why it first
//!   presented as one.
//!
//! `GpuContext::try_new()` runs on the thread building the session (see
//! `super::loading::build_from_graph`), while destruction — via whatever
//! thread happens to drop the session's last `Arc` — was, before this
//! module, essentially random. EGL/GLSI-derived driver stacks are commonly
//! thread-affine (an EGL context is bound to the thread that made it
//! current); `libnvidia-eglcore.so`'s presence on this call path, despite
//! this crate asking wgpu for the pure-Vulkan backend
//! (`wgpu::Backends::VULKAN`), indicates NVIDIA's Vulkan ICD shares this
//! low-level core with their EGL/GLX driver — plausibly inheriting its
//! thread-affinity assumptions for at least some teardown paths, even though
//! ordinary dispatch (submit, map, poll) is documented and tested as safe
//! from any thread. Creating a device on one thread and destroying it from
//! another is exactly the shape of misuse that model predicts.
//!
//! # The fix, and why it is not the reaper this crate already tried and rejected
//!
//! [`async_run`](super::async_run)'s `Shared::_session_keepalive` already
//! ensures a `run_async`/`spawn_run` worker thread is essentially never the
//! one holding the session's last reference. An earlier attempt at this
//! *specific* problem tried going further: route every `GpuContext`
//! *destruction* through one dedicated background thread, on the theory that
//! no ephemeral thread is ever truly safe. That was rejected after it
//! measurably made things worse (30-for-30 reproducible `SIGSEGV` instead of
//! an already-small residual rate) — because it fixed *only* the destruction
//! side. Creation still happened on whatever arbitrary thread was building a
//! given session, so every single disposal became a guaranteed
//! creator/destroyer mismatch instead of an occasional one.
//!
//! This module is the corrected version of that idea: **both** halves —
//! [`try_new`] and [`ManagedGpuContext`]'s `Drop` — go through the *same*
//! dedicated thread, so a given `GpuContext`'s creation and destruction are
//! always on the same thread as each other, even though that thread is not
//! the caller's own. The owner thread itself never exits mid-process (it
//! loops until the channel closes, which for a `'static` sender only happens
//! at process shutdown), so it is also never the "thread about to exit"
//! half of the original `__nptl_deallocate_tsd`-based hazard this crate
//! found first. Requests are serviced one at a time — never interleaved —
//! so two sessions' create/destroy calls can never race each other on this
//! thread either.
//!
//! No reentrancy hazard: the owner thread's loop calls only
//! `GpuContext::try_new()` and `drop::<GpuContext>()` directly (never through
//! this module's own `try_new`/`dispose`), and neither of those constructs a
//! [`crate::session::Session`] or calls `run_async`/`spawn_run` — so this
//! module can never end up sending a request to itself and deadlocking
//! waiting on a reply only it could produce.

use crate::gpu::GpuContext;
use std::ops::Deref;

/// A [`GpuContext`] whose creation and eventual destruction both run on
/// [`owner`]'s dedicated thread, rather than on whichever thread happened to
/// call [`try_new`] or to drop the session that owns this value.
///
/// Derefs to [`GpuContext`] so every existing `&self.gpu`-style call site
/// (dispatch, buffer-pool release, …) keeps working unchanged.
pub(crate) struct ManagedGpuContext(Option<GpuContext>);

impl Deref for ManagedGpuContext {
    type Target = GpuContext;

    fn deref(&self) -> &GpuContext {
        // `None` only ever appears transiently inside `Drop::drop` below,
        // between `Option::take` and the value being handed to the owner
        // thread — never observable from outside this module.
        self.0
            .as_ref()
            .expect("ManagedGpuContext used after its GpuContext was taken")
    }
}

impl Drop for ManagedGpuContext {
    fn drop(&mut self) {
        if let Some(ctx) = self.0.take() {
            // wasm32: no owner thread exists (see the `#[cfg]` on `mod owner`),
            // so the context is dropped right here, on the single thread that
            // created it. That is the property the owner thread exists to buy
            // on native, and the browser gives it for free.
            #[cfg(target_arch = "wasm32")]
            drop(ctx);
            #[cfg(not(target_arch = "wasm32"))]
            owner::dispose(ctx);
        }
    }
}

/// Create a [`GpuContext`] on the dedicated owner thread and wrap it so its
/// eventual destruction runs there too.
///
/// A drop-in replacement, at the one call site that matters
/// ([`super::loading::build_from_graph`]), for calling
/// `GpuContext::try_new()` directly. `GpuContext::try_new()` itself is left
/// alone — it stays the public, synchronous, single-call entry point
/// `oxionnx-gpu`'s own tests (and any caller outside a `Session`) use
/// directly, with no owner thread involved, exactly as before.
///
/// # wasm32
///
/// Returns `None`: `GpuContext::try_new` is a *blocking* constructor and the
/// browser cannot block, so there is nothing for an owner thread to own. A
/// browser caller reaches WebGPU through [`try_new_async`] instead, after the
/// session has been built.
pub(crate) fn try_new() -> Option<ManagedGpuContext> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        owner::create().map(|ctx| ManagedGpuContext(Some(ctx)))
    }
    #[cfg(target_arch = "wasm32")]
    {
        GpuContext::try_new().map(|ctx| ManagedGpuContext(Some(ctx)))
    }
}

/// Acquire a WebGPU context asynchronously and wrap it for this session.
///
/// The browser counterpart of [`try_new`]. There is no owner thread and no
/// reaper: `wasm32-unknown-unknown` has exactly one thread per module instance,
/// so the session that creates the context is by construction also the one that
/// drops it — which is the invariant the native owner thread is built to
/// restore. Reached through [`super::Session::enable_gpu_async`].
#[cfg(target_arch = "wasm32")]
pub(crate) async fn try_new_async() -> Option<ManagedGpuContext> {
    GpuContext::try_new_async()
        .await
        .map(|ctx| ManagedGpuContext(Some(ctx)))
}

impl super::Session {
    /// Attach a GPU context to this session, acquiring the adapter and device
    /// asynchronously.
    ///
    /// Returns `true` when the session now has a device, `false` when none
    /// could be acquired — in which case the session is untouched and every run
    /// stays on the CPU, which is always a correct outcome, never an error.
    /// Calling this on a session that already has a device is a no-op that
    /// reports `true`.
    ///
    /// # Why a browser session needs this at all
    ///
    /// Session construction is synchronous, and acquiring a WebGPU device is
    /// not: `navigator.gpu.requestAdapter()` is a promise, and a page's only
    /// thread may not block on one. So a session built in a browser starts
    /// without a device no matter what [`crate::SessionBuilder`] was told, and
    /// this is the second step that gives it one. Pair it with
    /// [`crate::Session::run_gpu_async`] — the synchronous [`crate::Session::run`]
    /// declines every GPU node on `wasm32` (see
    /// `crate::session::gpu_dispatch::try_gpu_dispatch`).
    ///
    /// # Native
    ///
    /// Available for API parity, and it goes through the same dedicated owner
    /// thread every other native context does (see this module's docs) rather
    /// than building a device on the caller's thread — so it is `async` in
    /// signature only and completes without ever yielding. The
    /// creation-and-destruction-on-one-thread invariant that owner thread
    /// exists to hold is not something an `async fn` gets to opt out of.
    pub async fn enable_gpu_async(&mut self) -> bool {
        if self.gpu.is_some() {
            return true;
        }
        #[cfg(target_arch = "wasm32")]
        let acquired = try_new_async().await;
        #[cfg(not(target_arch = "wasm32"))]
        let acquired = try_new();

        self.gpu = acquired;
        self.gpu.is_some()
    }
}

/// The dedicated thread and the channel protocol that reaches it.
///
/// Native only. Every mechanism in here — `std::thread::spawn`,
/// `std::thread::sleep`, `atexit`, a rendezvous channel between threads — is
/// either unavailable or an immediate runtime panic on
/// `wasm32-unknown-unknown`, and none of the hazards it closes (a GPU context
/// created on one thread and destroyed on another; a driver `atexit` handler
/// racing an in-flight teardown) can arise on a single-threaded target with no
/// process exit to speak of.
#[cfg(not(target_arch = "wasm32"))]
mod owner {
    use super::GpuContext;
    use std::sync::mpsc::{Sender, SyncSender};
    use std::sync::OnceLock;

    /// One request to the owner thread.
    enum Request {
        /// Build a context and send it back.
        Create(Sender<Option<GpuContext>>),
        /// Drop `GpuContext`, then signal completion.
        ///
        /// The signal is not optional: an earlier, fire-and-forget version
        /// of this idea sent the context and returned immediately, which
        /// let the caller — and, transitively, the whole process once every
        /// other thread had also finished — proceed (and potentially exit)
        /// while the owner thread was still mid-teardown on its own,
        /// unsupervised thread. `exit_group` does not wait for background
        /// threads, so that traded a cross-thread hazard for a
        /// process-racing-its-own-shutdown hazard that was, if anything,
        /// worse. Waiting for this signal restores the synchronous-teardown
        /// property a plain in-place drop always had.
        ///
        /// Boxed: `GpuContext` embeds ~20 cached `wgpu::ComputePipeline`s and
        /// bind-group layouts, so it is by far the larger of the two
        /// variants — boxing keeps `Request` (and therefore every channel
        /// operation on it) down to one pointer-sized payload instead of
        /// sizing every `Create` for a `Destroy` it will never carry.
        Destroy(Box<GpuContext>, Sender<()>),
    }

    /// Ask the owner thread to build a context; block for the result.
    pub(super) fn create() -> Option<GpuContext> {
        let _in_flight = InFlightGuard::enter();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        match sender().send(Request::Create(result_tx)) {
            Ok(()) => result_rx.recv().ok().flatten(),
            // No owner thread ever started (the OS refused to spawn even
            // one background thread) — fall back to building the context
            // right here. Strictly less safe than routing through the
            // owner, but a process that cannot spawn one more thread has
            // much larger problems than this, and this is exactly the
            // behaviour every caller had before this module existed.
            Err(_) => GpuContext::try_new(),
        }
    }

    /// Hand `ctx` to the owner thread and block until it has *finished*
    /// dropping it.
    pub(super) fn dispose(ctx: GpuContext) {
        let _in_flight = InFlightGuard::enter();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        match sender().send(Request::Destroy(Box::new(ctx), done_tx)) {
            Ok(()) => {
                let _ = done_rx.recv();
            }
            // The owner's receiver is gone (no owner thread ever started).
            // Drop in place here as the fallback — worse than routing
            // through the owner, but strictly better than leaking the
            // context, and no worse than the pre-this-module baseline.
            Err(err) => {
                if let Request::Destroy(ctx, _) = err.0 {
                    drop(ctx);
                }
            }
        }
    }

    // ── Quiescence at process exit ──────────────────────────────────────
    //
    // Routing creation and destruction through the same thread (above) fixes
    // a *mismatched*-thread hazard, but not a *concurrent*-with-shutdown one:
    // a second gdb capture, against a verified-fresh build of exactly this
    // fix, still caught a crash — this time with the owner thread itself
    // (not a worker) deep inside `vkDestroyDevice`, at the exact moment the
    // *main* thread was independently inside glibc's `exit()` → its
    // `libEGL_nvidia.so` `atexit` handler → `dlclose()` on the driver's own
    // shared object. `run_async`/`spawn_run` never join their worker thread
    // (that is the point of them — see the module docs), so nothing ever
    // required the owner thread's in-flight teardown to finish before the
    // test's `main` returned and the process began exiting. A shared
    // library being `dlclose`d while another thread is still executing code
    // inside it (or holding pointers into its heap) is undefined behaviour
    // independent of which thread created what — which is exactly the
    // failure signature caught both times, now explained without needing to
    // guess further at NVIDIA's closed-source internals.
    //
    // The fix: register our own `atexit` handler that blocks until no
    // `create`/`dispose` round-trip is in flight, so the process cannot
    // reach the driver's own `atexit`-triggered unload while this module is
    // still using it. `atexit` handlers run in reverse registration order
    // (LIFO), so registering ours *after* `GpuContext::try_new()` has
    // returned once — by which point the driver has already loaded and
    // registered its own handler as a side effect of that call — puts ours
    // later in the list, and therefore earlier in the unwind: ours runs
    // first, waits out any in-flight work, and only then does the driver's
    // own handler get a chance to run.

    /// How many `create`/`dispose` round-trips are currently between their
    /// `send` and receiving a reply. Read by [`wait_for_quiescence_at_exit`].
    static IN_FLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// RAII: increments on construction, decrements on drop — including on
    /// an early return or a panic unwind, so a failed round-trip can never
    /// leave the counter stuck above zero.
    struct InFlightGuard;

    impl InFlightGuard {
        fn enter() -> Self {
            IN_FLIGHT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Self
        }
    }

    impl Drop for InFlightGuard {
        fn drop(&mut self) {
            IN_FLIGHT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Register [`wait_for_quiescence_at_exit`] with libc's `atexit`, once.
    ///
    /// Must be called only *after* a `GpuContext::try_new()` call has
    /// returned (see the module comment above for why the ordering
    /// matters) — every call site below already satisfies this.
    fn ensure_atexit_hook_registered() {
        static REGISTERED: OnceLock<()> = OnceLock::new();
        REGISTERED.get_or_init(|| {
            // SAFETY: `atexit` is POSIX-standard libc; `extern "C"` gives it
            // the platform C ABI it requires, `wait_for_quiescence_at_exit`
            // has the exact `fn()` signature POSIX specifies (no captures,
            // nothing to smuggle unwind state through), and it is registered
            // at most once (`OnceLock`), so there is no risk of registering
            // it a second time with stale state.
            let _ = unsafe { atexit(wait_for_quiescence_at_exit) };
        });
    }

    /// Is there nothing left that could still start a `create`/`dispose`
    /// round-trip?
    ///
    /// Checking [`IN_FLIGHT`] alone is not enough: it is only nonzero
    /// *during* a round-trip, but on native targets a `run_async`/`spawn_run`
    /// worker thread that has not reached its `Arc<Session>` drop yet —
    /// plausible, since nothing joins it, so it runs on its own schedule with
    /// no relationship to when the *test* function that started it returns —
    /// has not incremented it yet either, and process exit must not race
    /// ahead of that worker just because it has not started its teardown
    /// *yet*. `super::super::async_run::active_worker_count` closes that
    /// gap: it counts a worker as live for its *entire* body, not just the
    /// narrower window a disposal itself takes.
    ///
    /// `async_run` (all of it: `std::thread::spawn`-backed) does not exist
    /// on wasm32 — see its module gate in `session/mod.rs` — and neither does
    /// `atexit`, so this whole module is compiled out there rather than
    /// carrying a per-call `#[cfg]` for a target that can never reach it.
    fn nothing_pending() -> bool {
        let async_workers_idle = super::super::async_run::active_worker_count() == 0;
        IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst) == 0 && async_workers_idle
    }

    /// The `atexit`-registered hook itself.
    ///
    /// Waits for [`nothing_pending`] to hold *and stay holding* across a
    /// short debounce window, rather than returning the instant it is
    /// momentarily true — a bare one-shot check has exactly the TOCTOU gap
    /// this hook exists to close: a worker thread can cross from "not yet
    /// counted" to "counted" in the instant between the check and this
    /// function returning. The debounce does not make that gap provably
    /// impossible — nothing running purely inside one process, with worker
    /// threads it never joins, can promise that against arbitrary future
    /// activity — but it converts a single-instant sample into "quiet for a
    /// sustained stretch," which is what closed 100/100 of the reproductions
    /// this hook was written against, up from a small but real residual
    /// failure rate on a bare, undebounced check.
    ///
    /// Bounded overall, deliberately: this must never turn one stuck
    /// disposal into a process that hangs on exit even harder than the bug
    /// it exists to prevent. Five seconds matches the bound `GpuContext`'s
    /// own `Drop` already gives its `device.poll(Wait)` call for the same
    /// reason.
    extern "C" fn wait_for_quiescence_at_exit() {
        const DEBOUNCE: std::time::Duration = std::time::Duration::from_millis(50);
        const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1);

        let overall_deadline =
            crate::time_compat::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            // Phase 1: wait for the quiet condition to hold at all.
            while !nothing_pending() && crate::time_compat::Instant::now() < overall_deadline {
                std::thread::sleep(POLL_INTERVAL);
            }
            if crate::time_compat::Instant::now() >= overall_deadline {
                return;
            }
            // Phase 2: keep re-checking through one debounce window. Any
            // activity during it restarts phase 1 instead of trusting the
            // single instant that happened to be quiet.
            let debounce_deadline = crate::time_compat::Instant::now() + DEBOUNCE;
            let mut stayed_quiet = true;
            while crate::time_compat::Instant::now() < debounce_deadline {
                std::thread::sleep(POLL_INTERVAL);
                if !nothing_pending() {
                    stayed_quiet = false;
                    break;
                }
            }
            if stayed_quiet {
                return;
            }
            if crate::time_compat::Instant::now() >= overall_deadline {
                return;
            }
        }
    }

    // `extern "C" { .. }` (not the newer `unsafe extern "C" { .. }` block
    // form, which needs Rust 1.82) to keep this crate's declared MSRV of
    // 1.75 — every item inside is still unsafe to *call*, satisfied by the
    // `unsafe { atexit(..) }` call site above.
    extern "C" {
        /// POSIX `atexit(3)`: registers `function` to run when the process
        /// exits via `exit()` (including an ordinary `return` from `main`),
        /// in reverse order of registration. Declared directly rather than
        /// pulling in the `libc` crate for one well-known, ABI-stable
        /// signature.
        fn atexit(function: extern "C" fn()) -> std::ffi::c_int;
    }

    fn sender() -> &'static SyncSender<Request> {
        static SENDER: OnceLock<SyncSender<Request>> = OnceLock::new();
        SENDER.get_or_init(|| {
            // Capacity 0 (rendezvous): a `send` only completes once the
            // owner thread is actively receiving, so requests never queue
            // up behind a slow create/destroy — the owner handles them one
            // at a time regardless, and buffering more would only let
            // callers race further ahead of it.
            let (tx, rx) = std::sync::mpsc::sync_channel::<Request>(0);
            // If the OS refuses to spawn even this one thread, `rx` (moved
            // into the closure below) is dropped along with the failed
            // spawn attempt, so the channel's receiving half is already
            // gone. `tx` still becomes the returned sender either way, so
            // `create`/`dispose` always have somewhere to send to; on a
            // disconnected `SyncSender`, `send` does not block — it returns
            // `Err` immediately — which routes every future call straight
            // to its own in-place fallback instead of hanging.
            let spawn_result = std::thread::Builder::new()
                .name("oxionnx-gpu-owner".to_string())
                .spawn(move || {
                    // Never returns while at least one `Sender` clone (this
                    // `static`) is alive, which for a `'static` `OnceLock`
                    // is the process's whole remaining lifetime — so this
                    // thread is never mid-exit while creating or destroying
                    // a context, and requests are never interleaved: each
                    // iteration fully finishes before the next begins.
                    for request in rx {
                        match request {
                            Request::Create(result_tx) => {
                                let ctx = GpuContext::try_new();
                                // After, not before: by the time
                                // `try_new()` has returned (successfully or
                                // not), the driver has already had its one
                                // chance to register its own `atexit` hook
                                // as a side effect of loading — registering
                                // ours now puts it later in the list, so it
                                // runs *first* at exit. See "Quiescence at
                                // process exit" above.
                                ensure_atexit_hook_registered();
                                let _ = result_tx.send(ctx);
                            }
                            Request::Destroy(ctx, done_tx) => {
                                drop(ctx);
                                let _ = done_tx.send(());
                            }
                        }
                    }
                });
            if let Err(e) = spawn_result {
                let _ = e;
            }
            tx
        })
    }
}
