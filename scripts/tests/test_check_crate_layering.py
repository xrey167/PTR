import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_crate_layering", ROOT / "scripts/check_crate_layering.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# The only violations the real catalog may carry. Shrinking this set is progress;
# adding to it needs a reviewed reason, not a quiet edit of crate-layers.toml.
PINNED_EXCEPTIONS = {("ptr-bench", "tokio-postgres")}

WORKSPACE = "/ws"


def package(name: str, top: str, deps: list, kinds: dict | None = None) -> dict:
    kinds = kinds or {}
    return {
        "name": name,
        "id": f"path+file://{WORKSPACE}/{top}/{name}#0.1.0",
        "manifest_path": f"{WORKSPACE}/{top}/{name}/Cargo.toml",
        "dependencies": [{"name": dep, "kind": kinds.get(dep)} for dep in deps],
    }


def metadata(*packages: dict) -> dict:
    return {
        "workspace_root": WORKSPACE,
        "workspace_members": [p["id"] for p in packages],
        "packages": list(packages),
    }


def catalog(**overrides) -> dict:
    base = {
        "layer": [
            {"name": "kernel", "rank": 0, "crates": ["ptr-types"]},
            {"name": "foundation", "rank": 1, "crates": ["ptr-ledger", "ptr-net"]},
            {"name": "domain", "rank": 2, "crates": ["ptr-pods"]},
        ],
        "backend_owner": [{"crate": "iroh", "owners": ["ptr-net"]}],
        "allowed_exception": [],
    }
    base.update(overrides)
    return base


def clean() -> list:
    return [
        package("ptr-types", "crates", []),
        package("ptr-ledger", "crates", ["ptr-types"]),
        package("ptr-net", "crates", ["ptr-types", "iroh"]),
        package("ptr-pods", "crates", ["ptr-ledger", "ptr-types"]),
    ]


class CrateLayeringTests(unittest.TestCase):
    def test_a_conforming_workspace_has_no_errors(self):
        self.assertEqual(mod.check(metadata(*clean()), catalog()), [])

    def test_a_dependency_on_a_higher_layer_is_an_error(self):
        packages = clean()
        packages[1] = package("ptr-ledger", "crates", ["ptr-types", "ptr-pods"])
        errors = mod.check(metadata(*packages), catalog())
        self.assertEqual(len(errors), 1)
        self.assertIn("ptr-ledger", errors[0])
        self.assertIn("higher layer", errors[0])

    def test_a_dependency_on_the_same_layer_is_allowed(self):
        packages = clean()
        packages[1] = package("ptr-ledger", "crates", ["ptr-types", "ptr-net"])
        self.assertEqual(mod.check(metadata(*packages), catalog()), [])

    def test_a_backend_crate_outside_its_owners_is_an_error(self):
        packages = clean()
        packages[3] = package("ptr-pods", "crates", ["ptr-types", "iroh"])
        errors = mod.check(metadata(*packages), catalog())
        self.assertEqual(len(errors), 1)
        self.assertIn("backend crate iroh", errors[0])
        self.assertIn("ptr-net", errors[0])

    def test_a_backend_pattern_covers_every_crate_with_that_prefix(self):
        packages = clean()
        packages[3] = package("ptr-pods", "crates", ["ptr-types", "candle-nn"])
        cat = catalog(backend_owner=[{"crate": "candle-*", "owners": []}])
        errors = mod.check(metadata(*packages), cat)
        self.assertEqual(len(errors), 1)
        self.assertIn("no crate", errors[0])

    def test_an_unplaced_crate_is_an_error(self):
        packages = clean() + [package("ptr-new", "crates", ["ptr-types"])]
        errors = mod.check(metadata(*packages), catalog())
        self.assertEqual(errors, ["ptr-new is not placed in any layer of crate-layers.toml"])

    def test_an_unplaced_binary_is_an_error(self):
        packages = clean() + [package("newbin", "bins", ["ptr-types"])]
        errors = mod.check(metadata(*packages), catalog())
        self.assertEqual(errors, ["newbin is not placed in any layer of crate-layers.toml"])

    def test_members_outside_crates_and_bins_need_no_placement(self):
        packages = clean() + [package("helper", "tools", ["ptr-types"])]
        self.assertEqual(mod.check(metadata(*packages), catalog()), [])

    def test_a_placed_crate_that_is_not_a_member_is_an_error(self):
        cat = catalog()
        cat["layer"][2]["crates"].append("ptr-ghost")
        errors = mod.check(metadata(*clean()), cat)
        self.assertEqual(
            errors, ["ptr-ghost is placed in crate-layers.toml but is not a workspace member"]
        )

    def test_a_crate_in_two_layers_is_an_error(self):
        cat = catalog()
        cat["layer"][2]["crates"].append("ptr-ledger")
        errors = mod.check(metadata(*clean()), cat)
        self.assertIn("ptr-ledger is placed in more than one layer", errors)

    def test_dev_dependencies_are_exempt(self):
        packages = clean()
        packages[1] = package(
            "ptr-ledger", "crates", ["ptr-types", "ptr-pods", "iroh"],
            {"ptr-pods": "dev", "iroh": "dev"},
        )
        self.assertEqual(mod.check(metadata(*packages), catalog()), [])

    def test_build_dependencies_are_checked(self):
        packages = clean()
        packages[1] = package("ptr-ledger", "crates", ["ptr-pods"], {"ptr-pods": "build"})
        self.assertEqual(len(mod.check(metadata(*packages), catalog())), 1)

    def test_a_listed_exception_silences_exactly_its_violation(self):
        packages = clean()
        packages[1] = package("ptr-ledger", "crates", ["ptr-pods", "iroh"])
        cat = catalog(
            allowed_exception=[{"from": "ptr-ledger", "to": "ptr-pods", "reason": "temporary"}]
        )
        errors = mod.check(metadata(*packages), cat)
        self.assertEqual(len(errors), 1)
        self.assertIn("backend crate iroh", errors[0])

    def test_an_exception_that_no_longer_matches_is_an_error(self):
        cat = catalog(
            allowed_exception=[{"from": "ptr-ledger", "to": "ptr-pods", "reason": "temporary"}]
        )
        errors = mod.check(metadata(*clean()), cat)
        self.assertEqual(len(errors), 1)
        self.assertIn("no longer matches a violation", errors[0])

    def test_an_exception_without_a_reason_is_an_error(self):
        packages = clean()
        packages[1] = package("ptr-ledger", "crates", ["ptr-pods"])
        cat = catalog(allowed_exception=[{"from": "ptr-ledger", "to": "ptr-pods", "reason": " "}])
        errors = mod.check(metadata(*packages), cat)
        self.assertEqual(errors, ["allowed_exception ptr-ledger -> ptr-pods has no reason"])

    def test_an_owner_that_is_not_a_member_is_an_error(self):
        cat = catalog(backend_owner=[{"crate": "iroh", "owners": ["ptr-net", "ptr-gone"]}])
        errors = mod.check(metadata(*clean()), cat)
        self.assertEqual(errors, ["backend iroh: owner ptr-gone is not a workspace member"])


class RealCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = mod.load_catalog()

    def test_exceptions_never_grow_beyond_the_pinned_set(self):
        listed = {(e["from"], e["to"]) for e in self.catalog.get("allowed_exception", [])}
        self.assertLessEqual(listed, PINNED_EXCEPTIONS)

    def test_every_exception_states_a_reason(self):
        for exception in self.catalog.get("allowed_exception", []):
            self.assertTrue(exception["reason"].strip(), exception)

    def test_ranks_are_unique_and_every_layer_is_named(self):
        layers = self.catalog["layer"]
        self.assertEqual(len({layer["rank"] for layer in layers}), len(layers))
        self.assertEqual(len({layer["name"] for layer in layers}), len(layers))

    def test_the_kernel_is_the_only_rank_zero_layer(self):
        rank_zero = [layer for layer in self.catalog["layer"] if layer["rank"] == 0]
        self.assertEqual([layer["crates"] for layer in rank_zero], [["ptr-types"]])

    def test_every_crate_directory_is_placed(self):
        placed = {crate for layer in self.catalog["layer"] for crate in layer["crates"]}
        on_disk = set()
        for top in ("crates", "bins"):
            for manifest in (ROOT / top).glob("*/Cargo.toml"):
                text = manifest.read_text(encoding="utf-8")
                name = next(
                    line.split("=", 1)[1].strip().strip('"')
                    for line in text.splitlines()
                    if line.startswith("name")
                )
                on_disk.add(name)
        self.assertEqual(on_disk - placed, set())
        self.assertEqual(placed - on_disk, set())


if __name__ == "__main__":
    unittest.main()
