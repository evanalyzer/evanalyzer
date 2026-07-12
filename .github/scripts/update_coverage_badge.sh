#!/usr/bin/env bash
#
# Regenerates docs/coverage-badge.svg and appends a row to
# docs/coverage-history.csv from a line-coverage percentage.
#
# Usage: update_coverage_badge.sh <percentage> <short-sha>
#
# Fully offline - no network call, no third-party badge-rendering service.
# The SVG is a plain flat badge (visually matching shields.io's own style,
# hand-rolled here so the committed image has no runtime dependency on an
# external renderer - it's just a static file GitHub serves from the repo).
set -euo pipefail

PCT="${1:?percentage required}"
SHA="${2:?short sha required}"
VALUE="${PCT}%"
INT_PCT="${PCT%.*}"

# Colour bands, loosely matching shields.io's own coverage colour scale.
color="#e05d44"  # red
if   [ "${INT_PCT}" -ge 90 ]; then color="#4c1"      # bright green
elif [ "${INT_PCT}" -ge 75 ]; then color="#97ca00"   # green
elif [ "${INT_PCT}" -ge 60 ]; then color="#dfb317"   # yellow
elif [ "${INT_PCT}" -ge 40 ]; then color="#fe7d37"   # orange
fi

label="coverage"
# Rough monospace-ish width estimate (~7px/char at font-size 11), padded.
label_w=$(( ${#label} * 7 + 20 ))
value_w=$(( ${#VALUE} * 7 + 20 ))
total_w=$(( label_w + value_w ))

mkdir -p docs
cat > docs/coverage-badge.svg <<SVG
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

history="docs/coverage-history.csv"
if [ ! -f "${history}" ]; then
  echo "date,commit,line_coverage_percent" > "${history}"
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${SHA},${PCT}" >> "${history}"

echo "Updated docs/coverage-badge.svg and ${history} (${VALUE} at ${SHA})"
