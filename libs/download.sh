#!/usr/bin/env bash
#
# Fetch EVAnalyzer's prebuilt native dependencies (libtorch, DuckDB, the bundled
# Java runtime and the Bio-Formats reader) from the GitHub "app-deps" release and
# place them where the build and packaging steps expect them.
#
# Why a script and not build.rs:
#   * libtorch must already be unpacked before `torch-sys`'s build script runs
#     (it reads $LIBTORCH from .cargo/config.toml), and torch-sys builds *before*
#     this workspace's crates - so provisioning it from our own build.rs is too
#     late.
#   * These archives are huge (libtorch alone is ~2 GB). Downloading them on
#     every `cargo build` - including offline/sandboxed/docs builds - would be
#     wrong. This runs once, explicitly, before cargo.
#
# Usage:
#   libs/download.sh <target>
#     target: linux-cpu | linux-cuda | win-cpu | win-cuda | mac-cpu
#
# The "app-deps" release must contain assets with these exact names:
#   libtorch-linux-cpu.zip   libtorch-linux-cuda.zip
#   libtorch-win-cpu.zip     libtorch-win-cuda.zip   libtorch-macos-arm64.zip
#   libduckdb-windows-amd64.zip   libduckdb-osx-universal.zip
#   jre_linux.zip   jre_win.zip   jre_macos_arm.zip
#   bioformats.jar
#
# Repo/tag are overridable via EVA_DEPS_REPO / EVA_DEPS_TAG.

set -euo pipefail

REPO="${EVA_DEPS_REPO:-evanalyzer/evanalyzer}"
TAG="${EVA_DEPS_TAG:-app-deps}"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIBS_DIR="${SCRIPT_DIR}"
JAVA_SRC_DIR="${ROOT_DIR}/crates/core/java/src"

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
    echo "usage: $0 <linux-cpu|linux-cuda|win-cpu|win-cuda|mac-cpu>" >&2
    exit 2
fi

# fetch <asset-name> <dest-path>
# Idempotent: skips assets that are already present (e.g. provided by the
# toolchain image or already committed), so it never clobbers existing files.
fetch() {
    local asset="$1" dest="$2"
    if [[ -s "${dest}" ]]; then
        echo "✓ ${dest##*/} already present - skipping"
        return
    fi
    echo "↓ ${asset}"
    mkdir -p "$(dirname "${dest}")"
    curl -fL --retry 3 --retry-delay 2 -o "${dest}" "${BASE}/${asset}"
}

# Bio-Formats reader: needed for every target (javac compiles against it).
fetch bioformats.jar "${JAVA_SRC_DIR}/bioformats.jar"

case "${TARGET}" in
    linux-cpu)
        fetch libtorch-linux-cpu.zip "${LIBS_DIR}/libtorch-linux-cpu.zip"
        fetch jre_linux.zip          "${LIBS_DIR}/jre_linux.zip"
        ;;
    linux-cuda)
        fetch libtorch-linux-cuda.zip "${LIBS_DIR}/libtorch-linux-cuda.zip"
        fetch jre_linux.zip           "${LIBS_DIR}/jre_linux.zip"
        ;;
    win-cpu)
        fetch libtorch-win-cpu.zip        "${LIBS_DIR}/libtorch-win-cpu.zip"
        fetch libduckdb-windows-amd64.zip "${LIBS_DIR}/libduckdb-windows-amd64.zip"
        fetch jre_win.zip                 "${LIBS_DIR}/jre_win.zip"
        ;;
    win-cuda)
        fetch libtorch-win-cuda.zip       "${LIBS_DIR}/libtorch-win-cuda.zip"
        fetch libduckdb-windows-amd64.zip "${LIBS_DIR}/libduckdb-windows-amd64.zip"
        fetch jre_win.zip                 "${LIBS_DIR}/jre_win.zip"
        ;;
    mac-cpu)
        fetch libtorch-macos-arm64.zip    "${LIBS_DIR}/libtorch-macos-arm64.zip"
        fetch libduckdb-osx-universal.zip "${LIBS_DIR}/libduckdb-osx-universal.zip"
        fetch jre_macos_arm.zip           "${LIBS_DIR}/jre_macos_arm.zip"
        ;;
    *)
        echo "unknown target: ${TARGET}" >&2
        echo "usage: $0 <linux-cpu|linux-cuda|win-cpu|win-cuda|mac-cpu>" >&2
        exit 2
        ;;
esac

echo "✓ native dependencies for ${TARGET} are ready"
