# PTR Atlas

An interactive map of every package, module and verified flow in the workspace.
Open `atlas.html` in a browser. It is one self-contained file.

- `extract_atlas.py` reads Cargo manifests and source files into `structure.json`
  (lines, tests, modules, public items, imports, which declared dependencies are
  really imported).
- `descriptions/area_*.json` hold the module descriptions and the flows. Each flow
  step cites the file and line it was checked against. These were written by reading
  the code, so re-check them when the code moves.
- `build_atlas.py` merges both into `atlas.html` using `atlas_template.html`.

Regenerate from the repository root:

    python3 docs/atlas/extract_atlas.py . docs/atlas/structure.json
    python3 docs/atlas/build_atlas.py
