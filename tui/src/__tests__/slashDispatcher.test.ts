import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GatewayClient } from '../gatewayClient.js'
import type { SessionInfo, SlashCommandInfo } from '../gatewayTypes.js'
import { chatActions } from '../store/chatStore.js'
import { overlayActions, overlayStack } from '../store/overlayStore.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'
import {
  dispatchSlash,
  isSlashCommandLine,
  parseSlash,
  splitArgs,
  type SlashCtx
} from '../slash/dispatcher.js'

describe('slash dispatcher', () => {
  beforeEach(() => {
    chatActions.resetForTests()
    sessionActions.resetForTests()
    overlayActions.resetForTests()
  })

  it('parses quoted args', () => {
    expect(splitArgs('/system "be precise"')).toEqual(['/system', 'be precise'])
    expect(parseSlash('/resume abc')).toEqual({ name: '/resume', args: ['abc'] })
  })

  it('does not treat absolute paths as slash commands', () => {
    expect(isSlashCommandLine('/tmp/example')).toBe(false)
    expect(parseSlash('/tmp/example')).toBeNull()
  })

  it('/help pushes the HelpOverlay onto the overlay stack', async () => {
    const ctx = makeCtx({
      catalog: [{ name: '/help', description: 'Show help', category: 'local' }]
    })

    await expect(dispatchSlash('/help', ctx)).resolves.toEqual({ kind: 'noop' })
    const stack = overlayStack.get()
    expect(stack).toHaveLength(1)
    expect(stack[0]?.kind).toBe('help')
  })

  it('/help --no-overlay falls back to a transcript note', async () => {
    const ctx = makeCtx({
      catalog: [{ name: '/help', description: 'Show help', category: 'local' }]
    })

    await expect(dispatchSlash('/help --no-overlay', ctx)).resolves.toMatchObject({
      kind: 'note',
      text: expect.stringContaining('/help')
    })
  })

  it('creates a new session for /new', async () => {
    const info = makeSessionInfo({ session_id: 'new-session' })
    const ctx = makeCtx({
      createSession: vi.fn(async () => info)
    })

    await expect(dispatchSlash('/new', ctx)).resolves.toMatchObject({
      kind: 'session',
      info
    })
  })

  it('/plan enters plan mode', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1' }))
    const request = vi.fn(async (method: string) => {
      if (method === 'plan.mode_set') {
        return { session_id: 's1', mode: 'plan' }
      }
      return {}
    })
    const ctx = makeCtx({ request: request as unknown as GatewayClient['request'] })

    await expect(dispatchSlash('/plan', ctx)).resolves.toEqual({
      kind: 'note',
      level: 'success',
      text: 'Enabled plan mode'
    })
    expect(request).toHaveBeenCalledWith('plan.mode_set', {
      session_id: 's1',
      mode: 'plan'
    })
    expect(sessionState.get().mode).toBe('plan')
  })

  it('/plan with a description enters plan mode and returns a query continuation', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1' }))
    const request = vi.fn(async () => ({ session_id: 's1', mode: 'plan' }))
    const ctx = makeCtx({ request: request as unknown as GatewayClient['request'] })

    await expect(dispatchSlash('/plan add auth flow', ctx)).resolves.toEqual({
      kind: 'query',
      query: 'add auth flow',
      note: 'Enabled plan mode',
      level: 'success'
    })
  })

  it('/plan in plan mode renders the current plan', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1', mode: 'plan' }))
    const request = vi.fn(async (method: string) => {
      if (method === 'plan.current') {
        return {
          session_id: 's1',
          mode: 'plan',
          plan_path: '/tmp/plan.md',
          has_plan: true,
          plan_content: '# Plan\n- add auth'
        }
      }
      return {}
    })
    const ctx = makeCtx({ request: request as unknown as GatewayClient['request'] })

    const result = await dispatchSlash('/plan', ctx)
    expect(result).toMatchObject({
      kind: 'note',
      level: 'info',
      text: expect.stringContaining('# Plan\n- add auth')
    })
    if (result.kind === 'note') {
      expect(result.text).toContain('Current Plan')
      expect(result.text).toContain('/tmp/plan.md')
    }
  })

  it('/plan open outside plan mode only enables plan mode', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1', mode: 'agent' }))
    const openFile = vi.fn(async () => undefined)
    const request = vi.fn(async (method: string) => {
      if (method === 'plan.mode_set') {
        return { session_id: 's1', mode: 'plan' }
      }
      return {}
    })
    const ctx = makeCtx({
      request: request as unknown as GatewayClient['request'],
      openFile
    })

    await expect(dispatchSlash('/plan open', ctx)).resolves.toEqual({
      kind: 'note',
      level: 'success',
      text: 'Enabled plan mode'
    })
    expect(openFile).not.toHaveBeenCalled()
  })

  it('/plan open reports no plan when the artifact is missing', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1', mode: 'plan' }))
    const request = vi.fn(async () => ({
      session_id: 's1',
      mode: 'plan',
      plan_path: null,
      has_plan: false,
      plan_content: null
    }))
    const ctx = makeCtx({ request: request as unknown as GatewayClient['request'] })

    await expect(dispatchSlash('/plan open', ctx)).resolves.toMatchObject({
      kind: 'note',
      level: 'warn',
      text: 'No plan written yet.'
    })
  })

  it('/plan open opens the artifact path when present', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1', mode: 'plan' }))
    const openFile = vi.fn(async () => undefined)
    const request = vi.fn(async () => ({
      session_id: 's1',
      mode: 'plan',
      plan_path: '/tmp/s1.md',
      has_plan: true,
      plan_content: 'plan'
    }))
    const ctx = makeCtx({
      request: request as unknown as GatewayClient['request'],
      openFile
    })

    await expect(dispatchSlash('/plan open', ctx)).resolves.toMatchObject({
      kind: 'note',
      level: 'success',
      text: 'Opened plan: /tmp/s1.md'
    })
    expect(openFile).toHaveBeenCalledWith('/tmp/s1.md')
  })

  it('/clear clears the current plan state when a session is active', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1', mode: 'plan' }))
    const request = vi.fn(async () => ({
      session_id: 's1',
      mode: 'agent',
      has_plan: false
    }))
    const ctx = makeCtx({ request: request as unknown as GatewayClient['request'] })

    await expect(dispatchSlash('/clear', ctx)).resolves.toEqual({ kind: 'clear' })
    expect(request).toHaveBeenCalledWith('plan.clear', { session_id: 's1' })
    expect(sessionState.get().mode).toBe('agent')
  })

  it('requests cancellation for /interrupt', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1' }))
    const request = vi.fn(async () => ({ ok: true }))
    const ctx = makeCtx({ request: request as unknown as GatewayClient['request'] })

    await expect(dispatchSlash('/interrupt', ctx)).resolves.toMatchObject({
      kind: 'note',
      text: 'interrupt'
    })
    expect(request).toHaveBeenCalledWith('agent.cancel', { session_id: 's1' })
  })

  it('/model updates the current session and applies discovered base_url', async () => {
    sessionActions.setSession(makeSessionInfo({ session_id: 's1', model: 'old' }))
    const updated = makeSessionInfo({
      session_id: 's1',
      model: 'kimi-k2.6',
      base_url: 'http://example.test/v1'
    })
    const request = vi.fn(async (method: string) => {
      if (method === 'providers.models') {
        return {
          models: [{ id: 'kimi-k2.6', display_name: 'kimi-k2.6' }],
          discovery: {
            kind: 'live',
            base_url: 'http://example.test',
            suggested_base_url: 'http://example.test/v1'
          }
        }
      }
      if (method === 'session.update') {
        return { session_id: 's1', info: updated }
      }
      return {}
    })
    const ctx = makeCtx({ request: request as unknown as GatewayClient['request'] })

    await expect(dispatchSlash('/model kimi-k2.6', ctx)).resolves.toMatchObject({
      kind: 'session',
      info: updated
    })
    expect(request).toHaveBeenCalledWith('session.update', {
      session_id: 's1',
      provider: 'openai',
      model: 'kimi-k2.6',
      base_url: 'http://example.test/v1'
    })
  })

  it('returns readable note for unknown commands', async () => {
    await expect(dispatchSlash('/missing', makeCtx())).resolves.toMatchObject({
      kind: 'note',
      level: 'warn'
    })
  })
})

function makeCtx(input: {
  catalog?: SlashCommandInfo[]
  request?: GatewayClient['request']
  createSession?: SlashCtx['createSession']
  openFile?: SlashCtx['openFile']
} = {}): SlashCtx {
  return {
    client: {
      request: input.request ?? (vi.fn(async () => ({})) as unknown as GatewayClient['request'])
    },
    catalog: input.catalog ?? [],
    getSession: () => sessionState.get(),
    createSession: input.createSession ?? vi.fn(async () => makeSessionInfo()),
    setSystemOverride: sessionActions.setSystemOverride,
    toggleVerbose: chatActions.toggleVerbose,
    ...(input.openFile ? { openFile: input.openFile } : {})
  }
}

function makeSessionInfo(input: Partial<SessionInfo> = {}): SessionInfo {
  return {
    session_id: input.session_id ?? 's1',
    created_at: 0,
    updated_at: 0,
    provider: input.provider ?? 'openai',
    model: input.model ?? 'gpt-4o',
    ...(input.base_url !== undefined ? { base_url: input.base_url } : {}),
    ...(input.mode !== undefined ? { mode: input.mode } : {})
  }
}
