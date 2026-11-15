# Verifying a download

Four different questions, four different tools. Use as many as you care about;
the first one needs nothing installed.

| question | tool | what it proves |
|---|---|---|
| Are these the bytes the project published? | `sha256sum` | integrity against the published checksum file |
| Did this file come from this repo, this commit, this workflow? | `gh attestation verify` | build provenance, signed by GitHub's OIDC identity |
| Who signed the checksums, and is that signature public? | `cosign verify-blob` | keyless signature, recorded in a public transparency log |
| Is this source tree exactly what the maintainer committed? | `git verify-tag` | the tag signature covers the commit, which covers every file |

## 1. Checksums (no extra tools)

```bash
sha256sum -c SHA256SUMS
```

Every asset is listed in `SHA256SUMS`. This catches a corrupted or truncated
download. It does **not** prove who produced the files: anyone who can replace
an asset can replace the checksum file too. That is what the next two steps are
for.

## 2. Build provenance (recommended)

```bash
gh attestation verify pasrocm-shim-0.1.0-linux-x86_64-rocm-7.2-gfx1100.tar.gz \
  --repo Eudox67/PasROCm
```

GitHub signs a statement binding each asset's digest to the repository, the
workflow file and the commit that produced it. No keys are involved and nothing
has to be trusted about the uploader: the signature comes from GitHub's own
OIDC identity for this repository's release workflow.

`gh` 2.49 or newer is required (`gh --version`). Install it from
<https://cli.github.com/> or your package manager.

## 3. The checksum signature (independent of GitHub)

```bash
cosign verify-blob \
  --bundle SHA256SUMS.bundle \
  --certificate-identity-regexp '^https://github.com/Eudox67/PasROCm/\.github/workflows/shim-release\.yml@refs/tags/v' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  SHA256SUMS
```

One signature covers the whole release, because `SHA256SUMS` lists every asset.
The bundle carries the certificate and the transparency-log entry, so this
verification works offline once you have downloaded it, and the log entry means
a signature cannot be quietly issued and hidden.

Install cosign from <https://docs.sigstore.dev/cosign/system_config/installation/>.

## 4. The source tree

Git tags and commits are signed with the maintainer's **SSH key**, whose public
half is committed here as [`SIGNING-KEY.pub`](../SIGNING-KEY.pub). Unlike GPG
there is no keyserver to fetch from, so tell git who is allowed to sign:

```bash
# one line: <email> <public key>
echo "145217249+Eudox67@users.noreply.github.com $(cat SIGNING-KEY.pub)" > allowed_signers
git config gpg.ssh.allowedSignersFile "$PWD/allowed_signers"

git verify-tag v0.1.0
```

Expected output:

```text
Good "git" signature for 145217249+Eudox67@users.noreply.github.com
with ED25519 key SHA256:Zw+/YJ2dBpc6zNbf1HoowngdlNNRI4ONYWmQSfQG60M
```

Check that fingerprint against the one published in `SECURITY.md` before
trusting the file you just downloaded -- a replaced repository would carry a
replaced key.

A tag signature covers the commit, and the commit covers every file in the tree,
so this is the strongest statement about the *source*: not just "these bytes are
intact" but "this exact tree is what the maintainer signed".

## Two signing systems, deliberately separate

| | Git tags and commits | Release artifacts |
|---|---|---|
| signed by | the maintainer's SSH key | GitHub's OIDC identity for this repository |
| key material | one personal key, public half in `SIGNING-KEY.pub` | none -- keyless, nothing to store or rotate |
| proves | *who wrote this source* | *which workflow built this binary* |
| verified with | `git verify-tag` | `gh attestation verify`, `cosign verify-blob` |

They answer different questions and are not interchangeable. A signed tag says
nothing about whether a published `.so` was built from it; an attestation says
nothing about whether a human approved the source. Use both.

## The library checks itself at load time

Every package ships `lib/libpasrocm_shim_<gfx>.so.sha256` next to the library.
If that file is present, the Pascal loader hashes the library before binding any
entry point and refuses it on a mismatch:

```
EShimIdentityError: PasROCm: .../libpasrocm_shim_gfx1100.so does not match its expected SHA256.
  expected: 136967e2...
  actual:   8db447dc...
```

This closes a hole the identity gate cannot: a library built for the *right* GPU
by someone else, sitting earlier in the search path. Target, wavefront and ABI
would all match; the hash does not.

| situation | behaviour |
|---|---|
| release package, untouched | sidecar matches, loads |
| library replaced or truncated | refused, both hashes named |
| `PASROCM_SHIM_SHA256=<hash>` set | that value wins over the sidecar |
| local build (`./build.sh`) | no sidecar, nothing hashed, loads normally |

To pin a library yourself, without a sidecar:

```bash
export PASROCM_SHIM_SHA256=$(sha256sum libpasrocm_shim_gfx1100.so | cut -d' ' -f1)
```

If you asked for a hash but the loader only has a bare file name (resolved by
`dlopen` through `LD_LIBRARY_PATH`), it refuses rather than skipping the check:
name the file with `PASROCM_SHIM` or its directory with `PASROCM_SHIM_DIR`.

## While the repository is private

Release assets of a private repository require authentication, and both
`gh attestation verify` and the shields.io badges fetch anonymously. Verification
by the general public therefore starts when the repository is made public.
