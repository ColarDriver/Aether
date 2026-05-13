import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from 'ink-testing-library'

import { PermissionModal } from '../overlays/PermissionModal.js'
import type { PermissionRequestParams } from '../gatewayTypes.js'
import { OVERLAY_PRIORITY, overlayActions, type OverlayState } from '../store/overlayStore.js'
import {
  permissionRulesActions,
  permissionRulesState
} from '../store/permissionRulesStore.js'

const SETTLE_MS = 80
const flush = (ms: number = SETTLE_MS): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms))

let nextCreatedAt = 1
function makeOverlay(
  payload: PermissionRequestParams,
  onDismiss = vi.fn()
): OverlayState<PermissionRequestParams> {
  const id = `srv_perm_${nextCreatedAt}`
  const overlay: OverlayState<PermissionRequestParams> = {
    kind: 'permission',
    id,
    payload,
    createdAt: nextCreatedAt++,
    priority: OVERLAY_PRIORITY.permission,
    onDismiss
  }
  overlayActions.push(overlay)
  return overlay
}

const SHELL_PARAMS: PermissionRequestParams = {
  session_id: 'ses_1',
  run_id: 'run_1',
  request: {
    tool_call_id: 'tc_1',
    tool_name: 'shell',
    arguments: { command: 'ls -la /tmp' },
    category: 'shell',
    risk: 'medium',
    preview: { title: 'Run shell command', command: 'ls -la /tmp' },
    allow_session: true
  },
  deadline_ms: 0
}

const DIFF_PARAMS: PermissionRequestParams = {
  session_id: 'ses_1',
  run_id: 'run_1',
  request: {
    tool_call_id: 'tc_2',
    tool_name: 'file_edit',
    arguments: { path: 'src/foo/bar.ts' },
    category: 'edit',
    risk: 'high',
    preview: {
      title: 'Edit src/foo/bar.ts',
      diff: '--- a/src/foo/bar.ts\n+++ b/src/foo/bar.ts\n@@ -1 +1 @@\n-old\n+new\n'
    },
    allow_session: true
  },
  deadline_ms: 0
}

describe('PermissionModal — shell command preview', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    permissionRulesActions.resetForTests()
    nextCreatedAt = 1
  })
  afterEach(() => {
    overlayActions.resetForTests()
    permissionRulesActions.resetForTests()
  })

  it('Y commits allow_once', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)
    expect(lastFrame()).toContain('Permission · shell')
    expect(lastFrame()).toContain('$ ls -la /tmp')

    stdin.write('y')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'allow_once' })
    unmount()
  })

  it('N commits deny', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)
    stdin.write('n')
    await flush()
    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'deny' })
    unmount()
  })

  it('ESC commits deny', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)
    stdin.write('\u001B')
    await flush()
    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'deny' })
    unmount()
  })

  it('S opens the prefix editor; Enter confirms with the suggested prefix and writes the cache', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('s')
    await flush()
    expect(lastFrame()).toContain('Allow for session matching:')
    expect(lastFrame()).toContain('command prefix:')

    stdin.write('\r')
    await flush()

    expect(dismiss).toHaveBeenCalledOnce()
    const args = dismiss.mock.calls[0]
    expect(args).toBeDefined()
    expect(args?.[0]).toBe('commit')
    expect(args?.[1]).toMatchObject({
      type: 'allow_session',
      rule: {
        tool_name: 'shell',
        behavior: 'allow',
        scope: 'session',
        command_prefix: 'ls'
      }
    })

    expect(permissionRulesState.get().rules).toHaveLength(1)
    expect(permissionRulesState.get().rules[0]).toMatchObject({
      toolName: 'shell',
      commandPrefix: 'ls'
    })
    unmount()
  })

  it('ESC inside the prefix editor returns to the choice footer instead of denying', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('s')
    await flush()
    expect(lastFrame()).toContain('command prefix:')

    stdin.write('\u001B')
    await flush()
    expect(lastFrame()).toContain('allow once')
    expect(dismiss).not.toHaveBeenCalled()
    unmount()
  })
})

describe('PermissionModal — diff preview', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    permissionRulesActions.resetForTests()
    nextCreatedAt = 1
  })
  afterEach(() => {
    overlayActions.resetForTests()
    permissionRulesActions.resetForTests()
  })

  it('renders coloured diff lines from preview.diff', () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(DIFF_PARAMS, dismiss)
    const { lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)
    expect(lastFrame()).toContain('-old')
    expect(lastFrame()).toContain('+new')
    unmount()
  })

  it('S in a diff request suggests the parent directory as path prefix', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(DIFF_PARAMS, dismiss)
    const { stdin, lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('s')
    await flush()
    expect(lastFrame()).toContain('path prefix:')
    expect(lastFrame()).toContain('src/foo/')
    unmount()
  })
})

describe('PermissionModal — cached rule banner', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    permissionRulesActions.resetForTests()
    nextCreatedAt = 1
  })

  it('shows a banner when a previously-set session rule matches', () => {
    permissionRulesActions.add({ toolName: 'shell', commandPrefix: 'ls' })
    const overlay = makeOverlay(SHELL_PARAMS)
    const { lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)
    expect(lastFrame()).toContain('matched cached session rule')
    unmount()
  })
})
