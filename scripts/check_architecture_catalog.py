from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "research" / "catalogs"


def load(name: str) -> dict:
    return tomllib.loads((CATALOG / name).read_text(encoding="utf-8"))


def check() -> list[str]:
    errors: list[str] = []
    crates = {
        path.name
        for path in (ROOT / "crates").glob("ptr-*")
        if (path / "Cargo.toml").exists()
    }
    owners = crates | {"training"}
    evaluation_ids = {
        item["id"]
        for item in tomllib.loads(
            (ROOT / "evaluations" / "registry.toml").read_text(encoding="utf-8")
        ).get("component", [])
    }

    types = load("type-families.toml")
    allowed_type_status = set(types.get("allowed_status", []))
    families = types.get("family", [])
    family_ids: set[str] = set()
    for family in families:
        family_id = family.get("id")
        if not family_id:
            errors.append("type family without id")
            continue
        if family_id in family_ids:
            errors.append(f"duplicate type family: {family_id}")
        family_ids.add(family_id)
        if family.get("owner") not in owners:
            errors.append(f"{family_id}: unknown owner {family.get('owner')}")
        if family.get("status") not in allowed_type_status:
            errors.append(f"{family_id}: invalid status {family.get('status')}")
        for key in ["purpose", "implemented", "open", "invariants"]:
            if key not in family:
                errors.append(f"{family_id}: missing {key}")

    backends = load("backend-slots.toml")
    allowed_backend_status = set(backends.get("allowed_status", []))
    slots = backends.get("slot", [])
    slot_ids: set[str] = set()
    for slot in slots:
        slot_id = slot.get("id")
        if not slot_id:
            errors.append("backend slot without id")
            continue
        if slot_id in slot_ids:
            errors.append(f"duplicate backend slot: {slot_id}")
        slot_ids.add(slot_id)
        if slot.get("owner") not in owners:
            errors.append(f"{slot_id}: unknown owner {slot.get('owner')}")
        if slot.get("status") not in allowed_backend_status:
            errors.append(f"{slot_id}: invalid status {slot.get('status')}")
        evaluation = slot.get("evaluation", "")
        if evaluation and evaluation not in evaluation_ids:
            errors.append(f"{slot_id}: unknown evaluation {evaluation}")
        for key in ["contract", "candidates", "open_questions"]:
            if key not in slot:
                errors.append(f"{slot_id}: missing {key}")

    contracts = load("component-contracts.toml").get("component", [])
    contract_ids: set[str] = set()
    for component in contracts:
        component_id = component.get("id")
        if component_id not in crates:
            errors.append(f"component contract references unknown crate {component_id}")
            continue
        if component_id in contract_ids:
            errors.append(f"duplicate component contract: {component_id}")
        contract_ids.add(component_id)
        for family_id in component.get("type_families", []):
            if family_id not in family_ids:
                errors.append(f"{component_id}: unknown type family {family_id}")
        for slot_id in component.get("backend_slots", []):
            if slot_id not in slot_ids:
                errors.append(f"{component_id}: unknown backend slot {slot_id}")
        for key in ["ports", "type_families", "backend_slots", "open_decisions"]:
            if key not in component:
                errors.append(f"{component_id}: missing {key}")

    missing_contracts = sorted(crates - contract_ids)
    if missing_contracts:
        errors.append(f"missing component contracts: {missing_contracts}")

    rust_api = load("rust-api-layout.toml")
    api_entries = rust_api.get("crate", [])
    api_ids: set[str] = set()
    for entry in api_entries:
        crate_id = entry.get("id")
        if crate_id not in crates:
            errors.append(f"rust api layout references unknown crate {crate_id}")
            continue
        if crate_id in api_ids:
            errors.append(f"duplicate rust api layout: {crate_id}")
        api_ids.add(crate_id)
        for key in ["current_shape", "modules", "public_groups", "function_groups", "notes"]:
            if key not in entry:
                errors.append(f"{crate_id}: rust api layout missing {key}")

    missing_api = sorted(crates - api_ids)
    if missing_api:
        errors.append(f"missing rust api layouts: {missing_api}")

    return errors


def main() -> int:
    errors = check()
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    print(
        "OK: architecture catalog "
        f"{len(load('type-families.toml').get('family', []))} type families, "
        f"{len(load('backend-slots.toml').get('slot', []))} backend slots, "
        f"{len(load('component-contracts.toml').get('component', []))} component contracts, "
        f"{len(load('rust-api-layout.toml').get('crate', []))} Rust API layouts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
