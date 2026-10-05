// Pure helpers of ptr-mods: no `$`, so they are unit-tested directly.

export const LOG_DIR = 'training/runs/logs'

export const EMPTY_RUNS_TEXT =
  'No runs yet — start one with make smoke-gpu, see docs/DESKTOP_SETUP.md'

export const NO_GPU_TEXT = 'no NVIDIA GPU / nvidia-smi not found'

// ---------- status line ----------

/** Path of one `git status --porcelain` line (renames: the new path), unquoted. */
export function porcelainPath(line: string): string {
  let rest = line.slice(3)
  const arrow = rest.indexOf(' -> ')
  if (arrow >= 0 && (line[0] === 'R' || line[1] === 'R' || line[0] === 'C' || line[1] === 'C')) {
    rest = rest.slice(arrow + 4)
  }
  if (rest.length >= 2 && rest.startsWith('"') && rest.endsWith('"')) {
    rest = rest.slice(1, -1).replace(/\\(["\\])/g, '$1')
  }
  return rest
}

export function porcelainPaths(porcelain: string): string[] {
  return porcelain
    .split('\n')
    .filter(line => line.length > 3)
    .map(porcelainPath)
}

/** True when a crate's src/ or Cargo.toml changed but its component.toml did not. */
export function needsDocs(paths: readonly string[]): boolean {
  const touched = new Set<string>()
  const documented = new Set<string>()
  for (const p of paths) {
    const m = /^crates\/([^/]+)\/(.*)$/.exec(p)
    if (m === null) continue
    const crate = m[1] as string
    const rest = m[2] as string
    if (rest === 'component.toml') documented.add(crate)
    else if (rest === 'Cargo.toml' || rest === 'src' || rest.startsWith('src/')) touched.add(crate)
  }
  for (const crate of touched) if (!documented.has(crate)) return true
  return false
}

/** `ptr <branch> · <n> changed` plus ` · docs?` when component.toml lags. */
export function buildStatusText(porcelain: string, branch: string): string {
  const paths = porcelainPaths(porcelain)
  const name = branch.trim() === '' ? '?' : branch.trim()
  return `ptr ${name} · ${paths.length} changed${needsDocs(paths) ? ' · docs?' : ''}`
}

// ---------- GPU ----------

export type GpuInfo = { utilPct: number; usedMiB: number; totalMiB: number; tempC: number }

/** Parses `--format=csv,noheader,nounits` rows; malformed rows are skipped. */
export function parseGpuCsv(stdout: string): GpuInfo[] {
  const gpus: GpuInfo[] = []
  for (const line of stdout.split('\n')) {
    if (line.trim() === '') continue
    const f = line.split(',').map(s => Number(s.trim()))
    if (f.length < 4 || f.slice(0, 4).some(n => !Number.isFinite(n))) continue
    gpus.push({
      utilPct: f[0] as number,
      usedMiB: f[1] as number,
      totalMiB: f[2] as number,
      tempC: f[3] as number,
    })
  }
  return gpus
}

export function formatGpuLine(gpus: readonly GpuInfo[]): string {
  if (gpus.length === 0) return NO_GPU_TEXT
  return gpus
    .map(
      (g, i) =>
        `${gpus.length > 1 ? `GPU ${i}: ` : 'GPU: '}${g.utilPct}% util · ` +
        `${g.usedMiB}/${g.totalMiB} MiB · ${g.tempC}°C`,
    )
    .join('\n')
}

/** GPU line from an nvidia-smi run; `exitCode` undefined = could not start. */
export function gpuLineFromRun(exitCode: number | undefined, stdout: string): string {
  if (exitCode !== 0) return NO_GPU_TEXT
  return formatGpuLine(parseGpuCsv(stdout))
}

// ---------- logs and PID ----------

export type LogEntry = { name: string; kind: string; mtimeMs: number }

/** Newest `*.log` file by mtime (name breaks ties, descending). */
export function pickNewestLog(entries: readonly LogEntry[]): LogEntry | undefined {
  const logs = entries.filter(e => e.kind === 'file' && e.name.endsWith('.log'))
  logs.sort((a, b) => b.mtimeMs - a.mtimeMs || (a.name < b.name ? 1 : -1))
  return logs[0]
}

export function tailLines(text: string, n: number): string[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  while (lines.length > 0 && lines[lines.length - 1] === '') lines.pop()
  return lines.slice(-n)
}

export function parsePid(text: string): number | undefined {
  const m = /^\s*(\d+)\s*$/m.exec(text)
  if (m === null) return undefined
  const pid = Number(m[1])
  return Number.isSafeInteger(pid) && pid > 0 ? pid : undefined
}

export function pidLine(pid: number | undefined, alive: boolean | undefined, hasFile: boolean): string {
  if (!hasFile) return 'PID: no .pid file next to the log'
  if (pid === undefined) return 'PID: .pid file unreadable (unknown)'
  if (alive === undefined) return `PID ${pid}: unknown`
  return `PID ${pid}: ${alive ? 'alive' : 'not running'}`
}

export function errText(e: unknown): string {
  if (e instanceof Error) return e.message
  return String(e)
}

export function formatTime(ms: number): string {
  return new Date(ms).toISOString().replace('T', ' ').slice(0, 19) + 'Z'
}
