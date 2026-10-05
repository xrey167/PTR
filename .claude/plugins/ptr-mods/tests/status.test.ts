import { describe, expect, test } from 'claude-code/testing'

import { buildStatusText, needsDocs, porcelainPaths } from '../hooks/lib'

describe('status text', () => {
  test('clean tree', () => {
    expect(buildStatusText('', 'main\n')).toBe('ptr main · 0 changed')
  })

  test('crate src without component.toml flags docs?', () => {
    const p = ' M crates/ptr-ledger/src/lib.rs\n?? notes.txt\n'
    expect(buildStatusText(p, 'feat/x')).toBe('ptr feat/x · 2 changed · docs?')
  })

  test('component.toml touched clears docs?', () => {
    const p = ' M crates/ptr-ledger/src/lib.rs\n M crates/ptr-ledger/component.toml\n'
    expect(buildStatusText(p, 'main')).toBe('ptr main · 2 changed')
  })

  test('Cargo.toml counts, per crate', () => {
    const p = ' M crates/a/Cargo.toml\n M crates/b/component.toml\n'
    expect(needsDocs(porcelainPaths(p))).toBe(true)
    expect(needsDocs(porcelainPaths(' M crates/b/component.toml\n M crates/b/Cargo.toml\n'))).toBe(false)
  })

  test('non-crate changes and renames', () => {
    expect(needsDocs(['docs/x.md', 'bins/ptrd/src/main.rs', 'crates/a/README.md'])).toBe(false)
    expect(porcelainPaths('R  old.rs -> crates/a/src/new.rs\n')).toEqual(['crates/a/src/new.rs'])
    expect(needsDocs(porcelainPaths('R  old.rs -> crates/a/src/new.rs\n'))).toBe(true)
  })

  test('quoted paths and untracked src dir', () => {
    expect(porcelainPaths('?? "a b.txt"\n')).toEqual(['a b.txt'])
    expect(needsDocs(porcelainPaths('?? crates/a/src/\n'))).toBe(true)
  })

  test('empty branch shows ?', () => {
    expect(buildStatusText('', '')).toBe('ptr ? · 0 changed')
  })
})
