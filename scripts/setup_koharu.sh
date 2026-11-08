#!/usr/bin/env bash
# Build the Koharu batch driver (engines/koharu/ct_batch.rs) against a pinned
# Koharu checkout. Idempotent; safe to re-run after editing ct_batch.rs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KOHARU_TAG="${KOHARU_TAG:-0.83.5}"
SRC="$ROOT/engines/koharu-src"
TARGET="$ROOT/engines/koharu-target"
SHIM="$ROOT/engines/sysshim"

# Koharu's prebuilt Metal runtimes support native Apple Silicon, not Intel or
# an x86_64 Rust toolchain launched through Rosetta.
OS="$(uname -s)"
case "$OS" in
    Darwin)
        if [[ "$(uname -m)" != arm64 ]]; then
            echo "Koharu on macOS requires Apple Silicon and a native arm64 terminal/Rust toolchain; Intel and Rosetta builds are unsupported." >&2
            exit 1
        fi
        for command in xcrun pkg-config; do
            command -v "$command" >/dev/null || {
                echo "Missing $command: install Xcode Command Line Tools and native pkg-config (Homebrew: brew install pkgconf)." >&2
                exit 1
            }
        done
        export CC="${CC:-$(xcrun --find clang)}"
        export CXX="${CXX:-$(xcrun --find clang++)}"
        export SDKROOT="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
        if [[ -z "${LIBCLANG_PATH:-}" ]]; then
            LIBCLANG_PATH="$(dirname "$(xcrun --find clang)")/../lib"
        fi
        if [[ ! -f "$LIBCLANG_PATH/libclang.dylib" && ! -f "$LIBCLANG_PATH" ]]; then
            echo "libclang is required by Koharu bindgen. Install Xcode Command Line Tools or LLVM (brew install llvm), then set LIBCLANG_PATH to its lib directory." >&2
            exit 1
        fi
        export LIBCLANG_PATH
        ;;
    Linux) ;;
    *)
        echo "Unsupported OS: $OS. Koharu setup supports Linux and native Apple Silicon macOS only." >&2
        exit 1
        ;;
esac
command -v python3 >/dev/null || {
    echo "python3 is required to apply portable Koharu source modifications." >&2
    exit 1
}

if [[ ! -d "$SRC/.git" ]]; then
    git clone --depth 1 --branch "$KOHARU_TAG" https://github.com/koharu-rs/koharu.git "$SRC"
fi

# Apply our pinned-source detection and typesetting fixes. Fail closed if the
# checkout matches neither the original nor the patched source.
QUALITY_PATCH="$ROOT/engines/koharu/quality.patch"
if git -C "$SRC" apply --check "$QUALITY_PATCH" 2>/dev/null; then
    git -C "$SRC" apply "$QUALITY_PATCH"
elif ! git -C "$SRC" apply --reverse --check "$QUALITY_PATCH" 2>/dev/null; then
    echo "Koharu quality patch does not match this checkout; refusing an unpatched build." >&2
    exit 1
fi

# Inject the batch driver and its dependencies without GNU-specific sed -i.
install -d "$SRC/crates/koharu-pipeline/src/bin"
install -m 0644 "$ROOT/engines/koharu/ct_batch.rs" "$SRC/crates/koharu-pipeline/src/bin/ct_batch.rs"
python3 - "$SRC" "$OS" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
if sys.argv[2] == "Linux":
    # CUDA 13.x minor-version compatibility on the tested 580 driver.
    cuda = source / "crates/koharu-runtime/src/hardware/cuda.rs"
    text = cuda.read_text()
    original = "const MIN_DRIVER_VERSION: c_int = 13030;"
    patched = "const MIN_DRIVER_VERSION: c_int = 13000;"
    if original in text:
        cuda.write_text(text.replace(original, patched))
    elif patched not in text:
        sys.exit("Koharu CUDA driver patch does not match this checkout.")

manifest = source / "crates/koharu-pipeline/Cargo.toml"
text = manifest.read_text()
section = re.search(r"(?m)^\[dependencies\]\n(.*?)(?=^\[|\Z)", text, re.S)
if section is None:
    sys.exit("Koharu pipeline manifest has no dependencies section.")
dependencies = section.group(1)
for dependency in (
    "serde_json = { workspace = true }",
    "url = { workspace = true }",
    "koharu-secrets = { workspace = true }",
    "koharu-runtime = { workspace = true }",
    'sha2 = "0.10"',
    "tempfile = { workspace = true }",
):
    name = dependency.split(" ", 1)[0]
    if not re.search(rf"(?m)^{re.escape(name)}\s*=", dependencies):
        dependencies += dependency + "\n"
original = "clap = { workspace = true }"
patched = 'clap = { workspace = true, features = ["env"] }'
if original in dependencies:
    dependencies = dependencies.replace(original, patched)
elif patched not in dependencies:
    sys.exit("Koharu clap env-feature patch does not match this checkout.")
updated = text[:section.start(1)] + dependencies + text[section.end(1):]
if updated != text:
    manifest.write_text(updated)
PY

rustup toolchain install stable --profile minimal >/dev/null
if [[ "$OS" == Darwin ]]; then
    RUST_INFO="$(rustup run stable rustc -vV)"
    if [[ "$RUST_INFO" != *$'\nhost: aarch64-apple-darwin\n'* ]]; then
        echo "Koharu Metal requires native Rust: install rustup/stable for aarch64-apple-darwin, not x86_64/Rosetta." >&2
        exit 1
    fi
    # Upstream downloads its Metal Torch/llama/diffusion libraries at runtime.
    # CoreText supplies macOS fonts; use the native SDK/libclang/pkg-config,
    # never the Linux fontconfig or GCC-header workarounds below.
    CARGO_TARGET_DIR="$TARGET" \
        rustup run stable cargo build --release --manifest-path "$SRC/Cargo.toml" -p koharu-pipeline --bin ct_batch
else
    # Build prerequisites without root on the Linux host: fontconfig runtime
    # shim and GCC's stddef.h for libclang installations without resource headers.
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
    PKG_CONFIG_PATH="$SHIM/lib/pkgconfig" \
    LIBCLANG_PATH="$LIBCLANG_DIR" \
    BINDGEN_EXTRA_CLANG_ARGS="-I$GCC_INCLUDE" \
    CARGO_TARGET_DIR="$TARGET" \
        rustup run stable cargo build --release --manifest-path "$SRC/Cargo.toml" -p koharu-pipeline --bin ct_batch
fi

echo "built: $TARGET/release/ct_batch"
