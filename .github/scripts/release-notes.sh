#!/usr/bin/env bash
# release-notes.sh — release notes of a tag from its conventional commits.
#
# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
#
# Usage: release-notes.sh [tag]   (default: $GITHUB_REF_NAME, else HEAD)
#
# Takes the commits since the previous version tag (test tags such as
# 0.1.0-test.11 are passed over, so their commits count towards the next
# real release) and sorts them by type: breaking changes (feat!: ...),
# features, performance, fixes. Everything else (refactor, test, docs, ci,
# build, style, chore, revert) is internal and left out. Scopes are dropped
# and the first letter is capitalised; the subject is used as written.
#
# The previous tag is the nearest one in the history, not the highest
# version - a stray tag on an older commit (e.g. a wrongly numbered
# 1.1.0-alpha.9) becomes the start of the range, so delete such tags.
#
# Writes Markdown to stdout. Needs the full history and tags
# (actions/checkout with fetch-depth: 0).

set -euo pipefail

tag="${1:-${GITHUB_REF_NAME:-}}"
if [[ -z "$tag" ]]; then
  tag="HEAD"
fi

prev="$(git describe --tags --abbrev=0 --match '[0-9]*' --match 'v[0-9]*' \
  --exclude '*-test.*' "${tag}^" 2>/dev/null || true)"
if [[ -n "$prev" ]]; then
  range="${prev}..${tag}"
else
  range="$tag"
fi

git log --no-merges --format='%s' "$range" | awk -v prev="$prev" -v tag="$tag" \
  -v repo="${GITHUB_REPOSITORY:-}" -v server="${GITHUB_SERVER_URL:-https://github.com}" '
function add(group, text) {
  text = toupper(substr(text, 1, 1)) substr(text, 2)
  items[group] = items[group] "- " text "\n"
}
{
  if (match($0, /^[a-z]+(\([a-z0-9,-]+\))?!?: /) == 0) next
  head = substr($0, 1, RLENGTH - 2)
  text = substr($0, RLENGTH + 1)
  breaking = (substr(head, length(head)) == "!")
  type = head
  sub(/[(!].*/, "", type)
  if (breaking) add("breaking", text)
  else if (type == "feat") add("feat", text)
  else if (type == "perf") add("perf", text)
  else if (type == "fix") add("fix", text)
}
END {
  n = split("breaking feat perf fix", order, " ")
  title["breaking"] = "Breaking changes"
  title["feat"] = "Features"
  title["perf"] = "Performance"
  title["fix"] = "Fixes"
  any = 0
  for (i = 1; i <= n; i++) {
    g = order[i]
    if (items[g] == "") continue
    if (any) printf "\n"
    printf "### %s\n\n%s", title[g], items[g]
    any = 1
  }
  if (!any) print "No user-facing changes."
  if (repo != "" && prev != "") {
    printf "\n**Full changelog:** %s/%s/compare/%s...%s\n", server, repo, prev, tag
  }
}'
