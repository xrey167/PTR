import { describe, expect, test } from 'claude-code/testing'

import {
  EMPTY_RUNS_TEXT,
  NO_GPU_TEXT,
  gpuLineFromRun,
  parseGpuCsv,
  parsePid,
  pickNewestLog,
  pidLine,
  tailLines,
} from '../hooks/lib'

describe('gpu csv', () => {
  test('parses one row', () => {
    expect(parseGpuCsv('37, 1024, 24576, 55\n')).toEqual([
      { utilPct: 37, usedMiB: 1024, totalMiB: 24576, tempC: 55 },
    ])
  })

  test('formats the line, several GPUs, skips junk', () => {
    expect(gpuLineFromRun(0, '37, 1024, 24576, 55\n')).toBe('GPU: 37% util · 1024/24576 MiB · 55°C')
    expect(gpuLineFromRun(0, '1, 2, 3, 4\n5, 6, 7, 8\n')).toContain('GPU 1: 5% util')
    expect(parseGpuCsv('[N/A], 1, 2, 3\nfoo\n')).toEqual([])
  })

  test('failure or empty output gives the no-GPU text', () => {
    expect(gpuLineFromRun(127, '')).toBe(NO_GPU_TEXT)
    expect(gpuLineFromRun(undefined, '')).toBe(NO_GPU_TEXT)
    expect(gpuLineFromRun(0, '')).toBe(NO_GPU_TEXT)
  })
})

describe('runs helpers and empty state', () => {
  test('empty state text names the next step', () => {
    expect(EMPTY_RUNS_TEXT).toContain('make smoke-gpu')
    expect(EMPTY_RUNS_TEXT).toContain('docs/DESKTOP_SETUP.md')
  })

  test('newest log by mtime, only .log files', () => {
    const e = (name: string, mtimeMs: number, kind = 'file') => ({ name, kind, mtimeMs })
    expect(pickNewestLog([e('a.log', 1), e('b.log', 5), e('c.pid', 9), e('d.log', 7, 'dir')])?.name).toBe('b.log')
    expect(pickNewestLog([e('x.pid', 1)])).toBeUndefined()
    expect(pickNewestLog([])).toBeUndefined()
  })

  test('tail, pid parse and pid line', () => {
    expect(tailLines('a\nb\nc\n', 2)).toEqual(['b', 'c'])
    expect(parsePid('4242\n')).toBe(4242)
    expect(parsePid('nope')).toBeUndefined()
    expect(pidLine(4242, true, true)).toBe('PID 4242: alive')
    expect(pidLine(4242, false, true)).toBe('PID 4242: not running')
    expect(pidLine(4242, undefined, true)).toBe('PID 4242: unknown')
    expect(pidLine(undefined, undefined, false)).toContain('no .pid file')
  })
})
