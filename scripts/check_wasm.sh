#!/bin/bash
# Verify the wasm32-unknown-unknown build: oxionnx (root crate, `wasm` feature)
# builds, and oxionnx-ops (the crate with the per-target oxifft split) checks.
# See CHANGELOG.md's [0.1.6] "Fixed" entry for why both halves matter --
# `cargo check` alone cannot prove the runtime-panic bugs this release fixed
# (std::time::Instant/SystemTime, std::thread::spawn) stay fixed; that needs
# an actual wasm-bindgen-test / browser run, not this script.
#
# Usage: ./scripts/check_wasm.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
    echo "Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

echo "==> cargo build -p oxionnx --target wasm32-unknown-unknown --features wasm"
cargo build -p oxionnx --target wasm32-unknown-unknown --features wasm

echo "==> cargo check -p oxionnx-ops --target wasm32-unknown-unknown"
cargo check -p oxionnx-ops --target wasm32-unknown-unknown

echo "OK: wasm32-unknown-unknown build is clean."
