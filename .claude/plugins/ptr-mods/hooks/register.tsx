import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { PtrModsRuns } from '../types'
import {
  EMPTY_RUNS_TEXT,
  LOG_DIR,
  buildStatusText,
  errText,
  formatTime,
  gpuLineFromRun,
  parsePid,
  pickNewestLog,
  pidLine,
  tailLines,
} from './lib'

type Engine = EngineInterface

const PANE = 'ptr-runs'
const REFRESH_MS = 30_000
const runs = atom({ plugin: 'ptr-mods', key: 'runs' } as const, null)

// ---------- /runs data ----------

async function pidAlive(
  $: Engine,
  pid: number,
): Promise<boolean | undefined> {
  try {
    const r = await $.process.run(['kill', '-0', String(pid)], { timeoutMs: 5000 })
    if (r.exitCode === 0) return true
    // EPERM means the process exists but belongs to someone else.
    return /not permitted/i.test(r.stderr)
  } catch {
    return undefined // no `kill` (Windows) or cannot start: unknown
  }
}

async function gpuLine($: Engine): Promise<string> {
  try {
    const r = await $.process.run(
      [
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
        '--format=csv,noheader,nounits',
      ],
      { timeoutMs: 8000 },
    )
    return gpuLineFromRun(r.exitCode, r.stdout)
  } catch {
    return gpuLineFromRun(undefined, '')
  }
}

export async function collectRuns($: Engine): Promise<PtrModsRuns> {
  const at = await $.clock.now()
  const root = await $.session.root()
  const dir = `${root.replace(/[\\/]$/, '')}/${LOG_DIR}`
  const base: PtrModsRuns = {
    at,
    dir: LOG_DIR,
    empty: null,
    log: null,
    pid: '',
    gpu: await gpuLine($),
    error: null,
  }
  try {
    if (!(await $.fs.exists(dir))) return { ...base, empty: EMPTY_RUNS_TEXT }
    const newest = pickNewestLog(await $.fs.list(dir))
    if (newest === undefined) return { ...base, empty: EMPTY_RUNS_TEXT }

    let tail: string[]
    try {
      tail = tailLines(await $.fs.read(`${dir}/${newest.name}`), 15)
    } catch (e) {
      tail = [`(cannot read log: ${errText(e)})`]
    }

    const pidPath = `${dir}/${newest.name.replace(/\.log$/, '')}.pid`
    let pid: string
    try {
      if (await $.fs.exists(pidPath)) {
        const n = parsePid(await $.fs.read(pidPath))
        pid = pidLine(n, n === undefined ? undefined : await pidAlive($, n), true)
      } else {
        pid = pidLine(undefined, undefined, false)
      }
    } catch (e) {
      pid = `PID: unknown (${errText(e)})`
    }
    return { ...base, log: { name: newest.name, mtimeMs: newest.mtimeMs, tail }, pid }
  } catch (e) {
    return { ...base, error: `refresh failed: ${errText(e)}` }
  }
}

async function refreshRuns($: Engine): Promise<void> {
  try {
    const view = await collectRuns($)
    await update($, runs, () => view)
  } catch {
    // never throw out of a hook
  }
}

// ---------- status line ----------

async function git($: Engine, argv: string[]): Promise<{ ok: boolean; out: string }> {
  try {
    const r = await $.process.run(['git', ...argv], { timeoutMs: 10_000 })
    return { ok: r.exitCode === 0, out: r.stdout }
  } catch {
    return { ok: false, out: '' }
  }
}

async function refreshStatus($: Engine): Promise<void> {
  try {
    const st = await git($, ['status', '--porcelain'])
    if (!st.ok) {
      $.ui.status(undefined) // not a git repo (or git missing)
      return
    }
    const br = await git($, ['rev-parse', '--abbrev-ref', 'HEAD'])
    $.ui.status(buildStatusText(st.out, br.ok ? br.out : '?'))
  } catch {
    try {
      $.ui.status(undefined)
    } catch {
      // ignore
    }
  }
}

// ---------- refresh timer ----------

// A reload drops the engine's timers and this variable together.
let timer: { cancel: () => void } | undefined

function stopTimer(): void {
  timer?.cancel()
  timer = undefined
}

async function tick($: Engine): Promise<void> {
  try {
    const open = (await $.ui.panes()).some(p => p.id === PANE)
    if (!open) return stopTimer()
    await refreshRuns($)
  } catch {
    // ignore
  }
}

function startTimer($: Engine): void {
  stopTimer()
  timer = $.clock.every(REFRESH_MS, () => {
    void tick($)
  })
}

// ---------- module ----------

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    try {
      await $.command.register({
        name: 'runs',
        description: 'Show the newest PTR training run (log tail, PID, GPU) in a pane',
      })
    } catch {
      // the status line still works without the command
    }
    const result = await next(e)
    await refreshStatus($)
    return result
  })

  on('command.run', { command: 'runs' }, async $ => {
    try {
      await refreshRuns($)
      const opened = await $.ui.open({ id: PANE, title: 'PTR Runs' })
      if (!opened.isPlaced) return { text: `PTR Runs pane not shown: ${opened.reason}` }
      startTimer($)
      return { text: 'PTR Runs pane opened (refreshes every 30 s).' }
    } catch (e) {
      return { text: `PTR Runs failed: ${errText(e)}` }
    }
  })

  on('tool.call', { tool: /^(Edit|Write|MultiEdit|Bash)$/ }, async ($, e, next) => {
    const ran = await next(e)
    await refreshStatus($)
    return ran
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e, next) => {
    try {
      const { Box, Button, Text } = $.ui.resolve(e)
      const view = await read($, runs)
      if (view === null) {
        return (
          <Box flexDirection="column">
            <Text dimColor>Loading…</Text>
            <Button key="refresh" label="Refresh" hotkey="r" onPress={() => refreshRuns($)} />
          </Box>
        )
      }
      return (
        <Box flexDirection="column">
          {view.empty !== null ? (
            <Text>{view.empty}</Text>
          ) : (
            <Box flexDirection="column">
              <Text bold>{view.log?.name ?? ''}</Text>
              <Text dimColor>
                {view.dir} · modified {view.log ? formatTime(view.log.mtimeMs) : '?'}
              </Text>
              <Text>{view.pid}</Text>
            </Box>
          )}
          <Text>{view.gpu}</Text>
          {view.error !== null && <Text color="red">{view.error}</Text>}
          {view.log !== null && (
            <Box flexDirection="column">
              <Text dimColor>── last {view.log.tail.length} lines ──</Text>
              {view.log.tail.map(line => (
                <Text wrap="truncate-end">{line}</Text>
              ))}
            </Box>
          )}
          <Text dimColor>updated {formatTime(view.at)}</Text>
          <Button key="refresh" label="Refresh" hotkey="r" onPress={() => refreshRuns($)} />
        </Box>
      )
    } catch {
      return next(e)
    }
  })
}
