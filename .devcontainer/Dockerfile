FROM rust:1.99-trixie AS live

# Slint debugging
RUN apt-get update && apt-get install -y \
    libxkbcommon0 \
    libgbm1 \
    libfontconfig1 \
    libinput10 \
    # X11 libraries
    libx11-6 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libxcursor1 \
    libxi6 \
    libxrandr2 \
    # Wayland libraries
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    && rm -rf /var/lib/apt/lists/*


# Tools
# Every tool pinned (version + --locked): CI and the release (which signs
# with these tools in the image) get exactly what was reviewed, not the
# newest upload of a crate.

RUN rustup component add rustfmt llvm-tools-preview
RUN cargo install --locked cargo-llvm-cov --version 0.9.1
RUN cargo install --locked cargo-audit --version 0.22.2
RUN cargo install --locked cargo-auditable --version 0.7.7
RUN cargo install --locked cargo-about --version 0.9.3 --features cli
RUN cargo install --locked cargo-cyclonedx --version 0.5.9
RUN cargo install --locked cargo-outdated --version 0.19.0
RUN cargo install --locked cargo-edit --version 0.13.14
RUN cargo install --locked ssign --version 0.1.8
RUN cargo install --locked tokei --version 15.0.0

# Packaging: zip/unzip (Windows zips, signing job), 7z (CUDA packages),
# xz (CLI CUDA package).
RUN apt-get update && apt-get install -y zip unzip p7zip-full xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Cross build
RUN cargo install --locked cargo-xwin --version 0.23.1
RUN apt-get update && apt-get install -y clang llvm gcc-aarch64-linux-gnu build-essential binutils lld mold \
    && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-pc-windows-msvc aarch64-pc-windows-msvc aarch64-unknown-linux-gnu
RUN ldconfig
