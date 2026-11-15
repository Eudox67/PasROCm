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
| release tags and commits | maintainer's SSH key (public half: `SIGNING-KEY.pub`) | `git verify-tag vX.Y.Z` |

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

### Checklist for the public repository

These need a public repo (or a paid plan). Tick them off; none is a prerequisite
for publishing a release, but each closes a specific hole.

| # | setting | what it prevents | how |
|---|---|---|---|
| 1 | Secret scanning + push protection | a token committed by accident | Settings → Code security → enable both |
| 2 | Protected `v*` tags | a published release tag being moved or deleted | Settings → Rules → New tag ruleset, pattern `v*` |
| 3 | Dependabot for Actions | a pinned action going stale and unpatched | `.github/dependabot.yml` (committed) |
| 4 | `CODEOWNERS` on `.github/` and `ci/` | a pipeline change slipping through review | `.github/CODEOWNERS` (committed) |
| 5 | `release` environment with required reviewers | a stolen token publishing without a human | Settings → Environments → New `release`, required reviewers |
| 6 | Branch protection on `main` | a direct push bypassing CI | Settings → Branches → require PR + the `plan` check |

**Note on #6:** requiring pull requests also blocks the maintainer's own direct
pushes to `main`. Choose it deliberately -- it is a workflow change, not only a
security setting.

### Publishing a release

The only trigger is a signed tag:

```bash
git tag -s vX.Y.Z -m "PasROCm X.Y.Z"
git push origin vX.Y.Z
```

`shim-release.yml` then builds every (SDK line, target) pair, refuses any
package whose ABI check failed, attests each asset, signs `SHA256SUMS`, and
publishes. A pre-release tag (`vX.Y.Z-rc.1`) runs the identical path and marks
the release as a pre-release -- use it to rehearse.

## Scope

This project ships bindings and a thin native shim. It does not sandbox GPU
kernels: a caller who passes a bad pointer or a wrong length can crash the
process, exactly as in C. The loader's identity and integrity gates exist to
make *substituted or mismatched libraries* fail loudly, not to validate
arguments.
