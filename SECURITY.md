# Security Policy

## Supported Versions

EVAnalyzer is pre-1.0 and does not maintain long-term support branches. Security fixes are made against the latest release only; please upgrade before reporting an issue if you're not already on the [latest release](https://github.com/evanalyzer/evanalyzer/releases/latest).

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/evanalyzer/evanalyzer/security) of this repository.
2. Click **Report a vulnerability**.
3. Describe the issue, including steps to reproduce and, if known, the affected version(s).

You should receive an acknowledgement within a few days. We'll work with you to confirm the issue, assess severity, and agree on a disclosure timeline before any public advisory is published.

## Dependencies

EVAnalyzer's Rust dependencies are checked against the [RustSec advisory database](https://rustsec.org/) with `cargo audit` on every CI run (see the audit badge in [README.md](README.md) and history in [`docs/audit-history.csv`](docs/audit-history.csv)). If you find a vulnerability in a dependency rather than in EVAnalyzer's own code, please report it upstream to the dependency's maintainers as well as to us.
