FROM rust:1.95-trixie AS live


# Install the Java Development Kit (JDK)
# openjdk-21-jdk is the current standard for Trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -LsSf https://github.com/axodotdev/cargo-dist/releases/latest/download/cargo-dist-installer.sh | sh

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

RUN rustup component add rustfmt
RUN cargo install cargo-llvm-cov
RUN rustup component add llvm-tools-preview

# Cross build
RUN cargo install cargo-xwin
RUN apt-get update && apt-get install -y clang llvm gcc-aarch64-linux-gnu zip build-essential binutils lld mold
RUN rustup target add x86_64-pc-windows-msvc
RUN rustup target add aarch64-unknown-linux-gnu
RUN ldconfig


