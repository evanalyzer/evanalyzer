#!/usr/bin/env bash
#
# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
#
# Fetch EVAnalyzer's prebuilt native dependencies and place them where the build
# and packaging steps expect them: DuckDB's library for the platforms that don't
# compile it from source, from our GitHub "app-deps" release. (The AI models run
# on burn, built in - no libtorch since 2026-11-25.)
#
# Why a script and not build.rs: libduckdb-sys reads DUCKDB_LIB_DIR before this
# workspace's crates build, and pulling archives on every `cargo build` would be
# wrong; this runs once, explicitly, before cargo.
#
# Usage:
#   libs/download.sh <target>
#     target: win | win-arm64 | mac
#
# Linux has no target here: DuckDB is compiled from source there.
#
# The "app-deps" release must contain these (small) assets:
#   libduckdb-windows-amd64.zip   libduckdb-windows-arm64.zip
#   libduckdb-osx-universal.zip
#
# Overridable via env: EVA_DEPS_REPO, EVA_DEPS_TAG.

set -euo pipefail

REPO="${EVA_DEPS_REPO:-evanalyzer/evanalyzer}"
TAG="${EVA_DEPS_TAG:-app-deps}"
RELEASE_BASE="https://github.com/${REPO}/releases/download/${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBS_DIR="${SCRIPT_DIR}"

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
    echo "usage: $0 <win|win-arm64|mac>" >&2
    exit 2
fi

# The SHA-256 of every archive this script fetches, by file name - a download
# (or a file already present) that differs stops it, so a changed or tampered
# archive never reaches a build or release. The "app-deps" release's assets
# (matched copies from 2026-06). A new DuckDB needs its sums added here.
#
# A function, not an associative array: macOS ships bash 3.2, which has none.
pinned_sha256() {
    case "$1" in
        libduckdb-windows-amd64.zip) echo "74e73afd3b010c6f310e14a961fff679f876952bc196f82584b0e2e76d11a91f" ;;
        libduckdb-windows-arm64.zip) echo "8056a9fc58a7c0ced9ebd5eb804e1bd2caf0e1ffb6711c8bd0c3ac0558a7da7f" ;;
        libduckdb-osx-universal.zip) echo "3f3c52970ad1407ec5037062e1a5e575b24bd5b993c889f89fe5876eff47782c" ;;
        *) echo "" ;;
    esac
}

# sha256_check <sha> <path>: the file's SHA-256 is <sha> (sha256sum on Linux
# and Windows, shasum on macOS, which has no sha256sum).
sha256_check() {
    if command -v sha256sum >/dev/null 2>&1; then
        echo "$1  $2" | sha256sum -c --quiet -
    else
        echo "$1  $2" | shasum -a 256 -c --quiet -
    fi
}

# verify <path>: the file's SHA-256 is the one pinned for its name.
verify() {
    local path="$1" name="${1##*/}"
    local sha
    sha="$(pinned_sha256 "${name}")"
    if [[ -z "${sha}" ]]; then
        echo "✗ no SHA-256 pinned for ${name} - add it to pinned_sha256 in $0" >&2
        exit 1
    fi
    if ! sha256_check "${sha}" "${path}"; then
        echo "✗ ${name} is not the archive pinned in $0 - delete it to download it again" >&2
        exit 1
    fi
}

# fetch <url> <dest-path>
# Idempotent: skips files that are already present (e.g. provided by a toolchain
# image or already downloaded), so it never clobbers existing files.
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

# fetch_duckdb <filename>: a DuckDB library from the "app-deps" release.
fetch_duckdb() {
    fetch "${RELEASE_BASE}/$1" "${LIBS_DIR}/$1"
    verify "${LIBS_DIR}/$1"
}

case "${TARGET}" in
    win)
        fetch_duckdb "libduckdb-windows-amd64.zip"
        ;;
    win-arm64)
        fetch_duckdb "libduckdb-windows-arm64.zip"
        ;;
    mac)
        fetch_duckdb "libduckdb-osx-universal.zip"
        ;;
    *)
        echo "unknown target: ${TARGET}" >&2
        echo "usage: $0 <win|win-arm64|mac>" >&2
        exit 2
        ;;
esac

echo "✓ native dependencies for ${TARGET} are ready"
