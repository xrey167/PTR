# PTR Open Architecture Catalog

This area is the machine-readable inventory for architecture decisions that are intentionally **not frozen yet**.

PTR separates three questions:

1. **Which semantic/type families must exist?**
2. **Which replaceable backend slots must the architecture expose?**
3. **Which component owns each contract, and what must cross that boundary?**

The catalog is deliberately broader than the current implementation. A row marked `open` or `scaffold` is not a TODO to blindly implement; it is a design slot that must be evaluated before compatibility is frozen.

## Files

- `type-families.toml` — domain/runtime type families, current coverage and open additions.
- `backend-slots.toml` — replaceable infrastructure/model/backend slots and evaluation ownership.
- `component-contracts.toml` — required ports and cross-component contract responsibilities.

## Status vocabulary

- `implemented` — usable reference semantics exist.
- `prototype` — executable semantics exist but the contract is still evolving.
- `scaffold` — the slot/shape exists but is incomplete.
- `open` — required design area, exact shape not selected.
- `deferred` — intentionally postponed until evidence/scale requires it.

A backend candidate is never architectural authority. Provider-native types stay inside adapters.

Validate with:

```bash
python scripts/check_architecture_catalog.py
```
