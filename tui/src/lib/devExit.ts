import { readFileSync } from 'node:fs'

const ENTRY_SCRIPT = 'src/entry.tsx'

export function shouldTerminateDevWatcher(
  env: NodeJS.ProcessEnv = process.env
): boolean {
  return env.npm_lifecycle_event === 'dev'
}

export function isTsxWatchCmdline(cmdline: string): boolean {
  return (
    cmdline.includes('tsx') &&
    cmdline.includes('--watch') &&
    cmdline.includes(ENTRY_SCRIPT)
  )
}

export function findWatchAncestorFromProcfs(
  startPid: number,
  readFile: (path: string) => string
): number | null {
  let pid = startPid
  let hops = 0
  while (pid > 1 && hops < 8) {
    const cmdline = safeReadFile(readFile, `/proc/${pid}/cmdline`)
      ?.replace(/\0/g, ' ')
      .trim()
    if (cmdline && isTsxWatchCmdline(cmdline)) {
      return pid
    }
    const status = safeReadFile(readFile, `/proc/${pid}/status`)
    const match = status?.match(/^PPid:\s+(\d+)$/m)
    const parentPid = match?.[1] ? Number.parseInt(match[1], 10) : NaN
    if (!Number.isFinite(parentPid) || parentPid <= 1) {
      return null
    }
    pid = parentPid
    hops += 1
  }
  return null
}

export function terminateDevWatchRunner(): boolean {
  if (!shouldTerminateDevWatcher()) {
    return false
  }
  const target = findWatchAncestorFromProcfs(process.ppid, (path) =>
    readFileSync(path, 'utf8')
  )
  if (!target) {
    return false
  }
  try {
    process.kill(target, 'SIGTERM')
    return true
  } catch {
    return false
  }
}

function safeReadFile(
  readFile: (path: string) => string,
  path: string
): string | null {
  try {
    return readFile(path)
  } catch {
    return null
  }
}
