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
      path: 'src/foo/bar.ts',
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

  it('renders the Python-style question + numbered options', () => {
    const overlay = makeOverlay(SHELL_PARAMS)
    const { lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)
    const frame = lastFrame()
    expect(frame).toContain('Run shell command')
    expect(frame).toContain('$ ls -la /tmp')
    expect(frame).toContain('Do you want to run this command?')
    expect(frame).toContain('1. Yes')
    expect(frame).toContain('2. Yes, allow this command prefix in this session')
    expect(frame).toContain('3. No')
    expect(frame).not.toContain('Permission · shell')
    unmount()
  })

  it('Enter on the default selection commits allow_once', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('\r')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'allow_once' })
    unmount()
  })

  it('number key 1 picks allow_once immediately', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('1')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'allow_once' })
    unmount()
  })

  it('number key 3 commits deny', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('3')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'deny' })
    unmount()
  })

  it('ESC commits abort (matches Python user_abort branch)', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'abort' })
    unmount()
  })

  it('Ctrl-C commits abort', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'abort' })
    unmount()
  })

  it('Down arrow navigates to allow_session; Enter commits it with no rule on the wire', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('[B')
    await flush()
    stdin.write('\r')
    await flush()

    expect(dismiss).toHaveBeenCalledOnce()
    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'allow_session' })

    // Local cache still derives a sensible rule so the banner can fire on the
    // next prompt before the engine round-trips its own session rule. The
    // prefix `ls -la` mirrors Python `_shell_command_prefix` taking the first
    // two whitespace-separated tokens.
    expect(permissionRulesState.get().rules).toEqual([
      { toolName: 'shell', commandPrefix: 'ls -la' }
    ])
    unmount()
  })

  it('Up arrow wraps to the last option', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('[A')
    await flush()
    stdin.write('\r')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'deny' })
    unmount()
  })

  it('Left/Right arrows still navigate', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('[C') // right
    await flush()
    stdin.write('\r')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'allow_session' })
    unmount()
  })

  it('number key 2 picks allow_session immediately without opening an editor', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(SHELL_PARAMS, dismiss)
    const { stdin, lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('2')
    await flush()

    expect(dismiss).toHaveBeenCalledOnce()
    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'allow_session' })
    expect(lastFrame()).not.toContain('Allow for session matching:')
    expect(lastFrame()).not.toContain('command prefix:')
    unmount()
  })
})

describe('PermissionModal — file_edit question', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    permissionRulesActions.resetForTests()
    nextCreatedAt = 1
  })

  it('renders the tool-specific edit question with the basename', () => {
    const overlay = makeOverlay(DIFF_PARAMS)
    const { lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)
    expect(lastFrame()).toContain('Do you want to make this edit to bar.ts?')
    expect(lastFrame()).toContain('-old')
    expect(lastFrame()).toContain('+new')
    expect(lastFrame()).toContain('2. Yes, allow edits in this path during this session')
    unmount()
  })

  it('allow_session on a file_edit caches a path-prefix rule', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(DIFF_PARAMS, dismiss)
    const { stdin, unmount } = render(<PermissionModal overlay={overlay} />)

    stdin.write('2')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { type: 'allow_session' })
    expect(permissionRulesState.get().rules).toEqual([
      { toolName: 'file_edit', pathPrefix: 'src/foo/bar.ts' }
    ])
    unmount()
  })
})

describe('PermissionModal — allow_session is hidden when allow_session=false', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    permissionRulesActions.resetForTests()
    nextCreatedAt = 1
  })

  it('shows only Yes / No when the request opts out of session-allow', () => {
    const overlay = makeOverlay({
      ...SHELL_PARAMS,
      request: { ...SHELL_PARAMS.request, allow_session: false }
    })
    const { lastFrame, unmount } = render(<PermissionModal overlay={overlay} />)
    const frame = lastFrame()
    expect(frame).toContain('1. Yes')
    expect(frame).toContain('2. No')
    expect(frame).not.toContain('allow this command prefix in this session')
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
