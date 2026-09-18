# Release Process

PTR has no stable release yet. The first release target is an alpha only after current CI, lifecycle tests and an end-to-end baseline run are reproducible.

## Release assets

The release workflow builds the Linux binaries, creates a SHA-256 checksum, exports Cargo dependency metadata and generates a CycloneDX 1.5 SBOM. GitHub's official `actions/attest@v4` creates Sigstore-backed build-provenance and SBOM attestations for the binary archive.

A tagged release therefore contains:

- `ptr-linux-x86_64.tar.gz`
- SHA-256 checksum
- CycloneDX SBOM + checksum
- Cargo dependency metadata
- GitHub/Sigstore provenance and SBOM attestations

This is release provenance, not a claim that the alpha is production-ready. Additional platforms and reproducible-build comparison should be added before a stable release.
