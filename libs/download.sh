#!/usr/bin/env bash
#
# Fetch EVAnalyzer's prebuilt native dependencies and place them where the build
# and packaging steps expect them.
#
# Sources:
#   * libtorch  -> download.pytorch.org (the canonical CDN). It is fetched here,
#     NOT from our GitHub release, because CUDA libtorch is far larger than the
#     2 GiB-per-file limit on GitHub release assets - and pytorch.org has no such
#     limit and is the reproducible upstream source.
#   * everything else (JRE, Bio-Formats, DuckDB) -> our GitHub "app-deps" release.
#     These are all small (< 100 MB), so the size limit is a non-issue.
#
# Why a script and not build.rs:
#   * libtorch must already be unpacked before `torch-sys`'s build script runs
#     (it reads $LIBTORCH from .cargo/config.toml), and torch-sys builds *before*
#     this workspace's crates - so provisioning it from our own build.rs is too
#     late.
#   * Pulling multi-GB archives on every `cargo build` would be wrong; this runs
#     once, explicitly, before cargo.
#
# Usage:
#   libs/download.sh <target>
#     target: linux-cpu | linux-cuda | win-cpu | win-cuda | mac-cpu
#
# The "app-deps" release must contain these (small) assets:
#   libduckdb-windows-amd64.zip   libduckdb-osx-universal.zip
#   jre_linux.zip   jre_win.zip   jre_macos_arm.zip
#   bioformats.jar
#
# Overridable via env: EVA_DEPS_REPO, EVA_DEPS_TAG, LIBTORCH_VERSION, CUDA_VARIANT.

set -euo pipefail

REPO="${EVA_DEPS_REPO:-evanalyzer/evanalyzer}"
TAG="${EVA_DEPS_TAG:-app-deps}"
RELEASE_BASE="https://github.com/${REPO}/releases/download/${TAG}"

LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.11.0}"
CUDA_VARIANT="${CUDA_VARIANT:-cu128}"
TORCH_BASE="https://download.pytorch.org/libtorch"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIBS_DIR="${SCRIPT_DIR}"
JAVA_SRC_DIR="${ROOT_DIR}/crates/core/java/src"

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
    echo "usage: $0 <linux-cpu|linux-cuda|win-cpu|win-cuda|mac-cpu>" >&2
    exit 2
fi

# fetch <url> <dest-path>
# Idempotent: skips files that are already present (e.g. provided by a toolchain
# image or already committed), so it never clobbers existing files.
fetch() {
    local url="$1" dest="$2"
    if [[ -s "${dest}" ]]; then
        echo "✓ ${dest##*/} already present - skipping"
        return
    fi
    echo "↓ ${url}"
    mkdir -p "$(dirname "${dest}")"
    curl -fL --retry 3 --retry-delay 2 -o "${dest}" "${url}"
}

# fetch_libtorch <flavour> <filename>
#   flavour:  cpu | <cuda-variant>   (URL path segment on download.pytorch.org)
#   filename: the libtorch archive name (also kept locally; the CI unzip globs
#             libs/libtorch-*.zip, so any libtorch-*.zip name works)
fetch_libtorch() {
    local flavour="$1" filename="$2"
    # '+' must be percent-encoded as %2B in the pytorch.org URL.
    local url_name="${filename//+/%2B}"
    fetch "${TORCH_BASE}/${flavour}/${url_name}" "${LIBS_DIR}/${filename}"
}

# Bio-Formats reader: needed for every target (javac compiles against it).
fetch "${RELEASE_BASE}/bioformats.jar" "${JAVA_SRC_DIR}/bioformats.jar"

case "${TARGET}" in
    linux-cpu)
        fetch_libtorch cpu "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip"
        fetch "${RELEASE_BASE}/jre_linux.zip" "${LIBS_DIR}/jre_linux.zip"
        ;;
    linux-cuda)
        fetch_libtorch "${CUDA_VARIANT}" "libtorch-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_VARIANT}.zip"
        fetch "${RELEASE_BASE}/jre_linux.zip" "${LIBS_DIR}/jre_linux.zip"
        ;;
    win-cpu)
        fetch_libtorch cpu "libtorch-win-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip"
        fetch "${RELEASE_BASE}/libduckdb-windows-amd64.zip" "${LIBS_DIR}/libduckdb-windows-amd64.zip"
        fetch "${RELEASE_BASE}/jre_win.zip" "${LIBS_DIR}/jre_win.zip"
        ;;
    win-cuda)
        fetch_libtorch "${CUDA_VARIANT}" "libtorch-win-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_VARIANT}.zip"
        fetch "${RELEASE_BASE}/libduckdb-windows-amd64.zip" "${LIBS_DIR}/libduckdb-windows-amd64.zip"
        fetch "${RELEASE_BASE}/jre_win.zip" "${LIBS_DIR}/jre_win.zip"
        ;;
    mac-cpu)
        fetch_libtorch cpu "libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
        fetch "${RELEASE_BASE}/libduckdb-osx-universal.zip" "${LIBS_DIR}/libduckdb-osx-universal.zip"
        fetch "${RELEASE_BASE}/jre_macos_arm.zip" "${LIBS_DIR}/jre_macos_arm.zip"
        ;;
    *)
        echo "unknown target: ${TARGET}" >&2
        echo "usage: $0 <linux-cpu|linux-cuda|win-cpu|win-cuda|mac-cpu>" >&2
        exit 2
        ;;
esac

echo "✓ native dependencies for ${TARGET} are ready"
