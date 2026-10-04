# PTR Atlas

An interactive map of every package, every module under `src/` and the verified flows of the
workspace. Example targets (such as `model/burn-a0/examples/` and
`crates/ptr-types/examples/`), benches and build scripts are not mapped, so the line and test
totals count `src/` only.
Open `atlas.html` in a browser. It is one self-contained file.

- `extract_atlas.py` reads Cargo manifests and source files into `structure.json`
  (lines, tests, modules, public items, imports, which declared dependencies are
  really imported).
- `descriptions/area_*.json` hold the module descriptions and the flows. Each flow
  step cites the file and line it was checked against. These were written by reading
  the code, so re-check them when the code moves.
- `build_atlas.py` merges both into `atlas.html` using `atlas_template.html`.

Imports are read from `crate::`, `self::` and `super::` paths, including grouped `use`
trees, with comments and string literals removed. A relative path such as
`pub use child::Item` (a re-export of a child module) and an inline `mod` block (which
changes what `super::` means) are not tracked.

The page records the commit it was built from and adds `+uncommitted` when the sources
differ from that commit. Commit source changes first, then regenerate, then commit the
generated files, so the recorded revision reproduces the page.

Regenerate from the repository root:

    python3 docs/atlas/extract_atlas.py . docs/atlas/structure.json
    python3 docs/atlas/build_atlas.py
