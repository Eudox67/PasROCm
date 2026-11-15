# Security policy

## Reporting a vulnerability

Please report privately rather than opening a public issue: use GitHub's
**Security → Report a vulnerability** on this repository, or contact the
maintainer through the address in the commit history.

Include the ROCm version, the GPU target (`gfxNNNN`), the shim's
`pasrocm_shim_build_info()` string, and a reproducer if you have one. The shim
loads native code and launches GPU kernels, so a memory-safety report is taken
seriously even when it needs an unusual input to trigger.

## What is signed

| artifact | signature | verify with |
|---|---|---|
| every release asset | GitHub build provenance attestation | `gh attestation verify <file> --repo Eudox67/PasROCm` |
| `SHA256SUMS` | keyless cosign signature + transparency-log entry | `cosign verify-blob --bundle SHA256SUMS.bundle …` |
| each library in a package | `.sha256` sidecar, checked by the loader at run time | automatic |
| release tags and commits | maintainer's SSH key (public half: `SIGNING-KEY.pub`) | `git verify-tag v0.1.0` |

`docs/VERIFY.md` has the exact commands and explains what each one proves.

### Two identities, kept apart

**Source** is signed by a person: the maintainer's SSH key. Its public half is
committed as `SIGNING-KEY.pub`, fingerprint:

```text
SHA256:Zw+/YJ2dBpc6zNbf1HoowngdlNNRI4ONYWmQSfQG60M  (ED25519)
```

The private half never leaves the maintainer's machine and is never committed,
copied into a package, or used by CI.

**Artifacts** are signed by a machine: the repository's own OIDC token, scoped
to the publishing job. There is no long-lived key in the pipeline -- nothing to
steal from CI and nothing to rotate. The personal key above plays no part in
producing or signing a release, and CI never sees it.

Keeping them separate means a compromised runner cannot forge the maintainer's
authorship, and a lost personal key does not invalidate the provenance of
already-published binaries.

## Repository hardening

In place:

- Workflow permissions are least-privilege: `contents: read` for builds; only
  the publishing job gets `contents: write`, `id-token: write` and
  `attestations: write`.
- Third-party actions are pinned by commit SHA, not by a moving tag.
- The GPU target is never inferred on a CI machine (`REQUIRE_ARCH=1`), so the
  runner a job lands on cannot change what an artifact contains.
- Builds are deterministic: no timestamps, no absolute paths, fixed archive
  order and ownership. The library SHA256 is the invariant across a rebuild.

Applied when the repository is made public (GitHub requires a public repo or a
paid plan for these):

- Branch protection on `main`: pull request required, the `plan` check required,
  no force-push, linear history.
- Protected `v*` tags, so a release tag cannot be moved or deleted.
- Secret scanning and push protection.
- Dependabot for the pinned GitHub Actions.
- `CODEOWNERS` covering `.github/` and `ci/`, so changes to the pipeline need
  maintainer review.
- A `release` environment with required reviewers, so a stolen token still
  cannot publish without a human approval.

## Scope

This project ships bindings and a thin native shim. It does not sandbox GPU
kernels: a caller who passes a bad pointer or a wrong length can crash the
process, exactly as in C. The loader's identity and integrity gates exist to
make *substituted or mismatched libraries* fail loudly, not to validate
arguments.
