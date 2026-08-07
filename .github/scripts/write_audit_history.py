#!/usr/bin/env python3
"""Appends per-finding rows to docs/audit-history.csv from a
`cargo audit --json` report.

One row per (crate, advisory) rather than a single per-run count, so which
library and which RUSTSEC id/CVE was affected is visible in the history
itself, not just how many were found. Covers both vulnerabilities and
warnings (unmaintained/unsound/yanked) - see .github/workflows/ci.yml for
the call site.

Usage: write_audit_history.py <cargo-audit-json-file> <short-sha>
"""

import csv
import json
import pathlib
import sys

FIELDS = ["date", "commit", "kind", "crate", "version", "advisory_id", "cve", "cvss_vector", "title"]


def cve_alias(advisory):
    return next((a for a in advisory.get("aliases", []) if a.startswith("CVE-")), "")


def main():
    json_path, sha = sys.argv[1], sys.argv[2]
    data = json.loads(pathlib.Path(json_path).read_text())

    # cargo-audit's own report doesn't carry a timestamp - stamp it here,
    # same as update_coverage_badge.sh / update_audit_badge.sh do for their
    # own history rows.
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for v in data["vulnerabilities"]["list"]:
        adv, pkg = v["advisory"], v["package"]
        rows.append(
            [now, sha, "vulnerability", pkg["name"], pkg["version"], adv["id"], cve_alias(adv), adv.get("cvss") or "", adv["title"]]
        )

    for kind, items in data.get("warnings", {}).items():
        for w in items:
            pkg = w["package"]
            adv = w.get("advisory")
            if adv:
                rows.append(
                    [now, sha, kind, pkg["name"], pkg["version"], adv["id"], cve_alias(adv), adv.get("cvss") or "", adv["title"]]
                )
            else:
                # `yanked` warnings have no advisory - just a package that
                # was pulled from the registry.
                rows.append([now, sha, kind, pkg["name"], pkg["version"], "", "", "", f"{pkg['name']} {pkg['version']} is yanked"])

    if not rows:
        rows.append([now, sha, "clean", "", "", "", "", "", "No vulnerabilities or warnings found"])

    history = pathlib.Path("docs/audit-history.csv")
    is_new = not history.exists()
    with history.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(FIELDS)
        writer.writerows(rows)

    print(f"Appended {len(rows)} row(s) to {history}")


if __name__ == "__main__":
    main()
