#!/usr/bin/env bash
#
# Regenerates docs/audit-badge.svg from a cargo-audit run's counts.
#
# Usage: update_audit_badge.sh <vulnerability-count> <warning-count>
#
# Fully offline - no network call, no third-party badge-rendering service.
# Same hand-rolled flat-badge SVG as update_coverage_badge.sh, so the
# committed image has no runtime dependency on an external renderer.
#
# The per-vulnerability detail (which crate, which advisory/CVE) goes to
# docs/audit-history.csv instead, written by write_audit_history.py - see
# that script and its call site in ci.yml.
set -euo pipefail

VULNS="${1:?vulnerability count required}"
WARNINGS="${2:?warning count required}"

if   [ "${VULNS}" -gt 0 ]; then
  VALUE="${VULNS} vulnerabilit$([ "${VULNS}" -eq 1 ] && echo y || echo ies)"
  color="#e05d44"  # red
elif [ "${WARNINGS}" -gt 0 ]; then
  VALUE="${WARNINGS} warning$([ "${WARNINGS}" -eq 1 ] || echo s)"
  color="#dfb317"  # yellow
else
  VALUE="clean"
  color="#4c1"      # bright green
fi

label="audit"
# Rough monospace-ish width estimate (~7px/char at font-size 11), padded.
label_w=$(( ${#label} * 7 + 20 ))
value_w=$(( ${#VALUE} * 7 + 20 ))
total_w=$(( label_w + value_w ))

mkdir -p docs
cat > docs/audit-badge.svg <<SVG
<svg xmlns="http://www.w3.org/2000/svg" width="${total_w}" height="20" role="img" aria-label="${label}: ${VALUE}">
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="${total_w}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="${label_w}" height="20" fill="#555"/>
    <rect x="${label_w}" width="${value_w}" height="20" fill="${color}"/>
    <rect width="${total_w}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="$(( label_w / 2 ))" y="14">${label}</text>
    <text x="$(( label_w + value_w / 2 ))" y="14">${VALUE}</text>
  </g>
</svg>
SVG

echo "Updated docs/audit-badge.svg (${VALUE})"
