#!/usr/bin/env bash
# check-glibc.sh — fails if a Linux package needs a newer glibc than allowed.
#
# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
#
# A program needs the glibc of the system it was built on - the toolchain
# image's. Moving that image to a newer base would raise the oldest Linux
# EVAnalyzer runs on without anyone noticing; this check makes it fail the
# build instead. The floor is documented on the download page (glibc 2.39:
# Ubuntu 24.04+, Debian 13+, Fedora 40+, RHEL 10+).
#
# Checks every ELF file (program and libraries) under the given folders or
# files - ours and the bundled ones (libtorch, NVIDIA's).
#
# Usage: check-glibc.sh <max version, e.g. 2.39> <file or folder>...

set -euo pipefail

max="${1:?usage: check-glibc.sh <max version> <file or folder>...}"
shift
[ "$#" -gt 0 ] || { echo "usage: check-glibc.sh <max version> <file or folder>..." >&2; exit 2; }

failed=0
checked=0
while IFS= read -r -d '' file; do
    # ELF files only (scripts, licenses and READMEs are skipped).
    [ "$(head -c 4 "$file" | od -An -tx1 | tr -d ' \n')" = "7f454c46" ] || continue
    checked=$((checked + 1))
    needed=$(objdump -T "$file" 2>/dev/null | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1 || true)
    [ -n "$needed" ] || continue
    newest=$(printf '%s\n%s\n' "$needed" "$max" | sort -V | tail -1)
    if [ "$newest" != "$max" ]; then
        echo "::error::$file needs glibc $needed - newer than the $max EVAnalyzer supports (see .github/scripts/check-glibc.sh)"
        failed=1
    else
        echo "ok: $file needs glibc $needed"
    fi
done < <(find "$@" -type f -print0)

[ "$checked" -gt 0 ] || { echo "::error::no ELF files found in $*"; exit 1; }
exit "$failed"
