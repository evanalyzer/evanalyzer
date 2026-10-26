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
#   * NVIDIA CUDA libraries the Linux CUDA libtorch needs but doesn't contain
#     -> PyPI (NVIDIA's own wheels, the ones the torch pip package installs).
#   * everything else (DuckDB) -> our GitHub "app-deps" release.
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
#     target: linux-cpu | linux-cuda | win-cpu | win-cuda | win-arm64-cpu | mac-cpu
#
# Linux ARM64 has no target here: pytorch.org publishes no libtorch for it
# (that build ships without the AI plugin), and DuckDB is compiled from
# source on Linux.
#
# The "app-deps" release must contain these (small) assets:
#   libduckdb-windows-amd64.zip   libduckdb-windows-arm64.zip
#   libduckdb-osx-universal.zip
#
# Overridable via env: EVA_DEPS_REPO, EVA_DEPS_TAG, LIBTORCH_VERSION, CUDA_VARIANT.

set -euo pipefail

REPO="${EVA_DEPS_REPO:-evanalyzer/evanalyzer}"
TAG="${EVA_DEPS_TAG:-app-deps}"
RELEASE_BASE="https://github.com/${REPO}/releases/download/${TAG}"

LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.11.0}"
# cu130: Turing (RTX 20xx / GTX 16xx) and newer GPUs, NVIDIA driver >= 580.
# Older GPUs use the CPU build.
CUDA_VARIANT="${CUDA_VARIANT:-cu130}"
TORCH_BASE="https://download.pytorch.org/libtorch"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBS_DIR="${SCRIPT_DIR}"

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
    echo "usage: $0 <linux-cpu|linux-cuda|win-cpu|win-cuda|win-arm64-cpu|mac-cpu>" >&2
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

# The CUDA libraries the Linux cu130 libtorch links (libtorch_cuda.so needs
# libcufft.so.12, libcurand.so.10, libcusparse.so.12 - and libcusparse needs
# libnvJitLink.so.13) but its zip leaves out: it expects them installed like
# the torch pip package does, from NVIDIA's wheels. The versions are the ones
# torch 2.11.0 pins (cuda-toolkit==13.0.2) - update them together with
# LIBTORCH_VERSION.
#
# Not cuSOLVER: it is only for libtorch_cuda_linalg.so (loaded on the first
# linear-algebra op on the GPU - none of the segmentation models has one),
# and it links cuBLAS by its plain name, which libtorch ships renamed. Both
# are left out of the package (.github/NOT_SHIP_LIST.txt).
NVIDIA_WHEELS=(
    "a8/2f/7b57e29836ea8714f81e9898409196f47d772d5ddedddf1592eadb8ab743/nvidia_cufft-12.0.0.61-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 6c44f692dce8fd5ffd3e3df134b6cdb9c2f72d99cf40b62c32dde45eea9ddad3"
    "a5/9f/be0a41ca4a4917abf5cb9ae0daff1a6060cc5de950aec0396de9f3b52bc5/nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl 1aee33a5da6e1db083fe2b90082def8915f30f3248d5896bcec36a579d941bfc"
    "fa/18/623c77619c31d62efd55302939756966f3ecc8d724a14dab2b75f1508850/nvidia_cusparse-12.6.3.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 2b3c89c88d01ee0e477cb7f82ef60a11a4bcd57b6b87c33f789350b59759360b"
    "56/7a/123e033aaff487c77107195fa5a2b8686795ca537935a24efae476c41f05/nvidia_nvjitlink-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 13a74f429e23b921c1109976abefacc69835f2f433ebd323d3946e11d804e47b"
)
NVIDIA_DIR="${LIBS_DIR}/nvidia-cu13"

# fetch_nvidia_libs
# Downloads the wheels above (checked against their SHA-256) and unpacks
# their shared libraries into libs/nvidia-cu13/lib.
fetch_nvidia_libs() {
    local entry path sha wheel
    for entry in "${NVIDIA_WHEELS[@]}"; do
        read -r path sha <<< "${entry}"
        wheel="${NVIDIA_DIR}/${path##*/}"
        fetch "https://files.pythonhosted.org/packages/${path}" "${wheel}"
        echo "${sha}  ${wheel}" | sha256sum -c --quiet -
        unzip -qoj "${wheel}" 'nvidia/cu13/lib/*.so*' -d "${NVIDIA_DIR}/lib"
    done
}

case "${TARGET}" in
    linux-cpu)
        fetch_libtorch cpu "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip"
        ;;
    linux-cuda)
        fetch_libtorch "${CUDA_VARIANT}" "libtorch-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_VARIANT}.zip"
        fetch_nvidia_libs
        ;;
    win-cpu)
        fetch_libtorch cpu "libtorch-win-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip"
        fetch "${RELEASE_BASE}/libduckdb-windows-amd64.zip" "${LIBS_DIR}/libduckdb-windows-amd64.zip"
        ;;
    win-cuda)
        fetch_libtorch "${CUDA_VARIANT}" "libtorch-win-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_VARIANT}.zip"
        fetch "${RELEASE_BASE}/libduckdb-windows-amd64.zip" "${LIBS_DIR}/libduckdb-windows-amd64.zip"
        ;;
    win-arm64-cpu)
        fetch_libtorch cpu "libtorch-win-arm64-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip"
        fetch "${RELEASE_BASE}/libduckdb-windows-arm64.zip" "${LIBS_DIR}/libduckdb-windows-arm64.zip"
        ;;
    mac-cpu)
        fetch_libtorch cpu "libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
        fetch "${RELEASE_BASE}/libduckdb-osx-universal.zip" "${LIBS_DIR}/libduckdb-osx-universal.zip"
        ;;
    *)
        echo "unknown target: ${TARGET}" >&2
        echo "usage: $0 <linux-cpu|linux-cuda|win-cpu|win-cuda|win-arm64-cpu|mac-cpu>" >&2
        exit 2
        ;;
esac

echo "✓ native dependencies for ${TARGET} are ready"
