//! Cross-platform monotonic clock.
//!
//! `std::time::Instant::now()` compiles on `wasm32-unknown-unknown` but
//! **panics at runtime** — there is no OS clock to query without a JS shim,
//! and `cargo check`/`cargo build` cannot catch that, only running the code
//! can. This module re-exports [`std::time::Instant`] unchanged on every
//! other target, and swaps in `web_time::Instant` — a drop-in replacement
//! backed by [`Performance.now()`], with the exact same API surface
//! (`now()`, `elapsed()`, `duration_since()`, `Add<Duration>`, ordering,
//! …) — on `wasm32-unknown-unknown`, so load-path instrumentation, node
//! dispatch timing and the GPU-owner debounce loop stay meaningful in the
//! browser instead of crashing on their first call.
//!
//! [`Performance.now()`]: https://developer.mozilla.org/en-US/docs/Web/API/Performance/now
//!
//! Import `Instant` from here — `crate::time_compat::Instant` — wherever
//! this crate needs wall-clock timing; never `std::time::Instant` directly.
//! (`std::time::Duration` itself is a plain arithmetic type with no OS
//! dependency and needs no substitute.)

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::Instant;
