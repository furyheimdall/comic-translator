#!/usr/bin/env bash
# Build the Koharu batch driver (engines/koharu/ct_batch.rs) against a pinned
# Koharu checkout. Idempotent; safe to re-run after editing ct_batch.rs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KOHARU_TAG="${KOHARU_TAG:-0.83.5}"
SRC="$ROOT/engines/koharu-src"
TARGET="$ROOT/engines/koharu-target"
SHIM="$ROOT/engines/sysshim"

if [[ ! -d "$SRC/.git" ]]; then
    git clone --depth 1 --branch "$KOHARU_TAG" https://github.com/koharu-rs/koharu.git "$SRC"
fi

# 1. DGX Spark ships driver 580 (CUDA 13.0). Koharu gates on 13.3 although its
#    CUDA 13.x runtime packages run under CUDA minor-version compatibility.
sed -i 's/const MIN_DRIVER_VERSION: c_int = 13030;/const MIN_DRIVER_VERSION: c_int = 13000;/' \
    "$SRC/crates/koharu-runtime/src/hardware/cuda.rs"

# Apply our pinned-source detection and typesetting fixes. Fail closed if the
# checkout matches neither the original nor the patched source.
QUALITY_PATCH="$ROOT/engines/koharu/quality.patch"
if git -C "$SRC" apply --check "$QUALITY_PATCH" 2>/dev/null; then
    git -C "$SRC" apply "$QUALITY_PATCH"
elif ! git -C "$SRC" apply --reverse --check "$QUALITY_PATCH" 2>/dev/null; then
    echo "Koharu quality patch does not match this checkout; refusing an unpatched build." >&2
    exit 1
fi

# 2. Batch driver binary and the extra dependencies it needs.
install -m 0644 "$ROOT/engines/koharu/ct_batch.rs" "$SRC/crates/koharu-pipeline/src/bin/ct_batch.rs"
MANIFEST="$SRC/crates/koharu-pipeline/Cargo.toml"
for dep in 'serde_json = { workspace = true }' 'url = { workspace = true }' 'koharu-secrets = { workspace = true }'; do
    name="${dep%% *}"
    grep -q "^$name = " "$MANIFEST" || sed -i "/^\[dependencies\]/a $dep" "$MANIFEST"
done
grep -q '^clap = { workspace = true, features = \["env"\] }' "$MANIFEST" \
    || sed -i 's/^clap = { workspace = true }/clap = { workspace = true, features = ["env"] }/' "$MANIFEST"

# 3. Build prerequisites without root: fontconfig pkg-config shim over the
#    runtime library, and GCC's stddef.h for bindgen (libclang-18 lacks its
#    resource headers on this host).
mkdir -p "$SHIM/lib/pkgconfig"
ln -sf "$(ldconfig -p | awk '/libfontconfig.so.1 /{print $NF; exit}')" "$SHIM/lib/libfontconfig.so"
cat > "$SHIM/lib/pkgconfig/fontconfig.pc" <<EOF
prefix=$SHIM
libdir=\${prefix}/lib
Name: Fontconfig
Description: Fontconfig runtime shim
Version: 2.15.0
Libs: -L\${libdir} -lfontconfig
Cflags:
EOF
GCC_INCLUDE="$(dirname "$(gcc -print-file-name=include/stddef.h)")"
LIBCLANG_DIR="${LIBCLANG_PATH:-$(ls -d /usr/lib/llvm-*/lib | sort -V | tail -1)}"

rustup toolchain install stable --profile minimal >/dev/null
PKG_CONFIG_PATH="$SHIM/lib/pkgconfig" \
LIBCLANG_PATH="$LIBCLANG_DIR" \
BINDGEN_EXTRA_CLANG_ARGS="-I$GCC_INCLUDE" \
CARGO_TARGET_DIR="$TARGET" \
    rustup run stable cargo build --release --manifest-path "$SRC/Cargo.toml" -p koharu-pipeline --bin ct_batch

echo "built: $TARGET/release/ct_batch"
