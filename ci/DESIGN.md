# Design record

The design record for this project — the decisions, the measured upstream
facts, the risk register and the list of open defects — is maintained privately
at <https://github.com/Eudox67/PasROCm-design>.

It is deliberately not part of what this repository ships. Nothing in the build
reads it, and no comment in the sources points into it: every measured fact
that a reader needs is stated where it is used, as a comment, a test name or a
line in `docs/`.

This stub exists so that the tooling, `docs/CI.md` and the commit messages have
one stable place to point at. For the public-facing summary of what the shim
guarantees, read `docs/INSTALL.md` (loader contract, per-SDK capabilities) and
`docs/CI.md` (validation levels, release flow).
