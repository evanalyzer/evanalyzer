#!/usr/bin/env bash
#
# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
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

LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.13.0}"
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

# The Linux CUDA libtorch contains no NVIDIA library at all: it expects
# them installed like the torch pip package does, from NVIDIA's wheels. Each
# entry: the wheel (path under files.pythonhosted.org/packages), its SHA-256,
# and the libraries to take from it - those libtorch links (readelf -d
# libtorch/lib/*.so) and the ones they open by name (cuDNN's engines,
# NVRTC's builtins); the wheels' profiler/MPI/network extras stay out. The
# versions are the ones torch 2.13.0 pins (cuda-toolkit==13.0.3 and its own
# nvidia-* requirements on PyPI) - update them together with
# LIBTORCH_VERSION; the release's ldd check reports a missing one.
NVIDIA_WHEELS=(
    "2e/24/d1558f3b68b1d26e706813b1d10aa1d785e4698c425af8db8edc3dced472/nvidia_cuda_runtime-13.0.96-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 7f82250d7782aa23b6cfe765ecc7db554bd3c2870c43f3d1821f1d18aebf0548 nvidia/cu13/lib/libcudart.so.13"
    "3b/cd/154ca20c38269e05eff77c1464e6c1da89f50a6390b565e9d82e06bc11e1/nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl 37936a16db8fe4ac1f065c2139360608a543a09275cb1a1af612e08cfa065436 nvidia/cu13/lib/libcublas.so.13 nvidia/cu13/lib/libcublasLt.so.13"
    "6e/5e/edb9c0ae051602c3ccaffe424256463636d639e27d7f302dde9975ef9e7a/nvidia_cudnn_cu13-9.20.0.48-py3-none-manylinux_2_27_x86_64.whl 0c45dd8eeb50b603f07995b1b300c62ffe6a1980482b82b3bcf94a4ca9d49304 nvidia/cudnn/lib/libcudnn*.so.9"
    "a8/2f/7b57e29836ea8714f81e9898409196f47d772d5ddedddf1592eadb8ab743/nvidia_cufft-12.0.0.61-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 6c44f692dce8fd5ffd3e3df134b6cdb9c2f72d99cf40b62c32dde45eea9ddad3 nvidia/cu13/lib/libcufft.so.12"
    "a5/9f/be0a41ca4a4917abf5cb9ae0daff1a6060cc5de950aec0396de9f3b52bc5/nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl 1aee33a5da6e1db083fe2b90082def8915f30f3248d5896bcec36a579d941bfc nvidia/cu13/lib/libcurand.so.10"
    "fa/18/623c77619c31d62efd55302939756966f3ecc8d724a14dab2b75f1508850/nvidia_cusparse-12.6.3.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 2b3c89c88d01ee0e477cb7f82ef60a11a4bcd57b6b87c33f789350b59759360b nvidia/cu13/lib/libcusparse.so.12"
    "56/7a/123e033aaff487c77107195fa5a2b8686795ca537935a24efae476c41f05/nvidia_nvjitlink-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 13a74f429e23b921c1109976abefacc69835f2f433ebd323d3946e11d804e47b nvidia/cu13/lib/libnvJitLink.so.13"
    "5f/67/cba3777620cdacb99102da4042883709c41c709f4b6323c10781a9c3aa34/nvidia_cusolver-12.0.4.66-py3-none-manylinux_2_27_x86_64.whl 0a759da5dea5c0ea10fd307de75cdeb59e7ea4fcb8add0924859b944babf1112 nvidia/cu13/lib/libcusolver.so.12"
    "eb/4d/84149aecfc0cc3c448e09494ba1a71b1667f3dc4179c79834a3c708046aa/nvidia_cusparselt_cu13-0.8.1.post1-py3-none-manylinux2014_x86_64.whl 0dd8522d704f8181f5fa60b14e350495096bc53990749b7685e82262ac53b48c nvidia/cusparselt/lib/libcusparseLt.so.0"
    "67/f4/58e4e91b6919367c7aafb8e36fce9aad1a3047e536bf7e2fd560927d3a4c/nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_x86_64.whl edd81538446786ec3b73972543e53bb43bcaf0bfc8ef76cb679fcc390ffe136d nvidia/nccl/lib/libnccl.so.2"
    "3c/35/a9bf80a609e74e3b000fef598933235c908fcefcef9026042b8e6dfde2a9/nvidia_nvshmem_cu13-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 290f0a2ee94c9f3687a02502f3b9299a9f9fe826e6d0287ee18482e78d495b80 nvidia/nvshmem/lib/libnvshmem_host.so.3"
    "3f/70/4f193de89a48b71714e74602ee14d04e4019ad36a5a9f20c425776e72cd6/nvidia_cufile-1.15.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 08a3ecefae5a01c7f5117351c64f17c7c62efa5fffdbe24fc7d298da19cd0b44 nvidia/cu13/lib/libcufile.so.0"
    "33/6d/737d164b4837a9bbd202f5ae3078975f0525a55730fe871d8ed4e3b952b0/nvidia_cuda_cupti-13.0.85-py3-none-manylinux_2_25_x86_64.whl 4eb01c08e859bf924d222250d2e8f8b8ff6d3db4721288cf35d14252a4d933c8 nvidia/cu13/lib/libcupti.so.13"
    "c3/68/483a78f5e8f31b08fb1bb671559968c0ca3a065ac7acabfc7cee55214fd6/nvidia_cuda_nvrtc-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl ad9b6d2ead2435f11cbb6868809d2adeeee302e9bb94bcf0539c7a40d80e8575 nvidia/cu13/lib/libnvrtc.so.13 nvidia/cu13/lib/libnvrtc-builtins.so.13.0"
)
NVIDIA_DIR="${LIBS_DIR}/nvidia-cu13"

# fetch_nvidia_libs
# Downloads the wheels above (checked against their SHA-256) and unpacks
# their shared libraries into libs/nvidia-cu13/lib.
fetch_nvidia_libs() {
    local entry path sha libs wheel
    for entry in "${NVIDIA_WHEELS[@]}"; do
        read -r path sha libs <<< "${entry}"
        wheel="${NVIDIA_DIR}/${path##*/}"
        fetch "https://files.pythonhosted.org/packages/${path}" "${wheel}"
        echo "${sha}  ${wheel}" | sha256sum -c --quiet -
        # One argument per library; a pattern (libcudnn*) is unzip's, not the
        # shell's (set -f).
        # shellcheck disable=SC2086
        (set -f; unzip -qoj "${wheel}" ${libs} -d "${NVIDIA_DIR}/lib")
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
