//! Cross-platform wall-clock time.
//!
//! `std::time::SystemTime::now()` compiles on `wasm32-unknown-unknown` but
//! **panics at runtime** — there is no OS clock to query without a JS shim,
//! and `cargo check`/`cargo build` cannot catch that, only running the code
//! can. This crate calls it exactly once, as a last-resort PRNG seed for
//! `Random*`/`Bernoulli` operators whose ONNX `seed` attribute is absent
//! (spec-legal non-determinism) — a case a browser-hosted model can hit as
//! easily as a native one. This module re-exports [`std::time::SystemTime`]
//! and [`std::time::UNIX_EPOCH`] unchanged on every other target, and swaps
//! in `web_time`'s drop-in replacements — backed by [`Date.now()`], same API
//! surface (`now()`, `duration_since()`, …) — on `wasm32-unknown-unknown`.
//!
//! [`Date.now()`]: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/now
//!
//! Import from here — `crate::time_compat::{SystemTime, UNIX_EPOCH}` —
//! wherever this crate needs wall-clock time; never `std::time::SystemTime`
//! directly.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::{SystemTime, UNIX_EPOCH};
