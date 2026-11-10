# syntax=docker/dockerfile:1.7
#
# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
#
# EVAnalyzer without the desktop app: the command line (`cli analyze`, ...),
# the server and its workers - made from the release's command-line
# packages, nothing is compiled here:
#
#   evanalyzer-cli-linux-x86_64.tar.gz       -> linux/amd64 (AI on the CPU)
#   evanalyzer-cli-linux-arm64.tar.gz        -> linux/arm64 (no AI)
#   evanalyzer-cli-linux-x86_64-cuda.tar.xz  -> linux/amd64 (AI with CUDA)
#
# The packages come in as the build context `package`, unpacked into one
# folder per architecture (amd64/, arm64/):
#
#   mkdir -p pkg/amd64 pkg/arm64
#   tar -xzf evanalyzer-cli-linux-x86_64.tar.gz -C pkg/amd64
#   tar -xzf evanalyzer-cli-linux-arm64.tar.gz  -C pkg/arm64
#   docker buildx build -f docker/Dockerfile --build-context package=pkg \
#       --platform linux/amd64,linux/arm64 --build-arg VERSION=1.2.0 \
#       -t evanalyzer .
#
# The main build context is the repository root (only docker/server.toml is
# read from it, see .dockerignore). Unreleased code: docker/Dockerfile.source.
# Usage: see docker/README.md.

# Debian 13: glibc 2.41, newer than the 2.39 the packages need.
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian13:nonroot
# The same image with a shell (busybox) - only to create the folders below,
# none of it ends up in the image.
ARG HELPER_IMAGE=gcr.io/distroless/cc-debian13:debug-nonroot

# ---------------------------------------------------------------------------
# Replaced by `--build-context package=<folder>`; empty otherwise, so a build
# without it fails at the COPY below.
FROM scratch AS package

# ---------------------------------------------------------------------------
# Empty folders owned by the nonroot user (uid 65532), to be mounted over -
# a named volume takes over their owner. Folders are the same on every
# architecture: made on the build machine's own, no emulation.
FROM --platform=$BUILDPLATFORM ${HELPER_IMAGE} AS folders
SHELL ["/busybox/sh", "-c"]
RUN mkdir -p /home/nonroot/out/data /home/nonroot/out/var/lib/evanalyzer

# ---------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE}

ARG TARGETARCH
ARG VERSION=0.0.0-dev

LABEL org.opencontainers.image.title="EVAnalyzer" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/evanalyzer/evanalyzer" \
      org.opencontainers.image.description="EVAnalyzer command line and server (no desktop app)" \
      org.opencontainers.image.licenses="PolyForm-Noncommercial-1.0.0" \
      org.opencontainers.image.authors="Joachim Danmayr"

# The package as released: the program, ai/ (the plugin and its libtorch),
# templates/, the licences.
COPY --from=package /${TARGETARCH}/ /opt/evanalyzer/
# The server's container defaults: listen on every interface, state in a
# volume, users may work in /data. Override with your own --config file.
COPY docker/server.toml /opt/evanalyzer/server.toml
COPY --from=folders --chown=65532:65532 /home/nonroot/out/ /

USER 65532:65532
WORKDIR /data
VOLUME ["/data", "/var/lib/evanalyzer"]
# `evanalyzer server` (docker/server.toml).
EXPOSE 7400

ENTRYPOINT ["/opt/evanalyzer/evanalyzer"]
CMD ["--help"]
