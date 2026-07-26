# Security policy

## Reporting

Please report vulnerabilities privately to the repository owner rather than
opening a public issue with exploit details.

## Dependency policy

The project pins Git dependencies to immutable commits and runs RustSec auditing
in CI. A dependency update must pass formatting, clippy, tests, rustdoc, and the
advisory scan before it is merged.

## Accepted upstream advisory

`RUSTSEC-2023-0071` affects the transitive `rsa` crate used by Azalea for
private-key signing. RustSec currently lists no patched version. The risk is a
network-observable timing side channel in contexts where an attacker can measure
private RSA operations.

Until an upstream replacement is available:

- do not treat this client as suitable for high-value long-lived RSA keys on
  hostile servers;
- keep Azalea and the RustSec advisory under review;
- do not add any other audit exemption;
- remove the CI exemption as soon as a patched dependency path exists.

The crate forbids project-owned `unsafe` code and does not read credentials or
execute commands. Optional path visualization writes only beneath the
operator-supplied `PF_PATH_DIR`; server-provided profile names are normalized to
single safe filenames before use.
