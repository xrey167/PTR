export type PtrModsRuns = {
  /** Epoch ms of the refresh that produced this view. */
  at: number
  /** Where the log directory was looked up (display only). */
  dir: string
  /** Empty-state text when there is no log, else null. */
  empty: string | null
  log: { name: string; mtimeMs: number; tail: string[] } | null
  /** Text for the PID line, e.g. "pid 123: alive". */
  pid: string
  /** Text for the GPU line. */
  gpu: string
  /** A failure of the refresh itself, shown as text. */
  error: string | null
}

declare module 'claude-code' {
  interface PluginState {
    'ptr-mods': { runs: PtrModsRuns | null }
  }
}
