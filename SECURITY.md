# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue.** Report privately via one of:

- **AMD Product Security portal:** https://www.amd.com/en/resources/product-security.html

Please include: description and impact, steps to reproduce, and affected versions or commits.

We aim to acknowledge reports within 1 business day.

## Scope

This policy covers code and configuration in this repository. For vulnerabilities in third-party dependencies, report upstream. For AMD product issues unrelated to this repo, use the [AMD Product Security portal](https://www.amd.com/en/resources/product-security.html).

## Automated security scanning

Every pull request is scanned by [`.github/workflows/pr-security-scan.yml`](.github/workflows/pr-security-scan.yml), and the whole repository is scanned weekly by [`.github/workflows/weekly-security-scan.yml`](.github/workflows/weekly-security-scan.yml). Both call the shared `ROCm/rocm-security-gh` `security-baseline.yml` reusable workflow (CodeQL, gitleaks, zizmor, bandit, and Trivy); HIPIFY's scanner configuration is in [`.github/scan_tools_configs/`](.github/scan_tools_configs/).

A scanner finding is not a vulnerability report. If a finding turns out to be an exploitable vulnerability, report it through the AMD Product Security portal above, not in a public issue or pull request.
