import { describe, expect, it } from 'vitest'

import {
  findWatchAncestorFromProcfs,
  isTsxWatchCmdline,
  shouldTerminateDevWatcher
} from '../lib/devExit.js'

describe('devExit', () => {
  it('detects tsx watch entry command lines', () => {
    expect(
      isTsxWatchCmdline(
        '/usr/bin/node /workspace/Aether/tui/node_modules/.bin/tsx --watch src/entry.tsx'
      )
    ).toBe(true)
    expect(
      isTsxWatchCmdline(
        '/usr/bin/node --require tsx/dist/preflight.cjs --import loader.mjs src/entry.tsx'
      )
    ).toBe(false)
  })

  it('only enables watcher termination under npm run dev', () => {
    expect(shouldTerminateDevWatcher({ npm_lifecycle_event: 'dev' })).toBe(true)
    expect(shouldTerminateDevWatcher({ npm_lifecycle_event: 'start' })).toBe(false)
    expect(shouldTerminateDevWatcher({})).toBe(false)
  })

  it('finds the nearest tsx watch ancestor in procfs data', () => {
    const files = new Map<string, string>([
      ['/proc/200/cmdline', '/usr/bin/node\0src/entry.tsx\0'],
      ['/proc/200/status', 'Name:\tnode\nPPid:\t150\n'],
      [
        '/proc/150/cmdline',
        '/usr/bin/node\0/workspace/Aether/tui/node_modules/.bin/tsx\0--watch\0src/entry.tsx\0'
      ],
      ['/proc/150/status', 'Name:\tnode\nPPid:\t100\n'],
      ['/proc/100/cmdline', 'npm\0run\0dev\0'],
      ['/proc/100/status', 'Name:\tnpm\nPPid:\t1\n']
    ])
    const readFile = (path: string) => {
      const value = files.get(path)
      if (value === undefined) {
        throw new Error(`missing ${path}`)
      }
      return value
    }
    expect(findWatchAncestorFromProcfs(200, readFile)).toBe(150)
  })
})
