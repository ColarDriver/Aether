// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, getSessionToken, request, setBaseUrl, setSessionToken } from './client'

describe('api client', () => {
  beforeEach(() => {
    setBaseUrl('http://aether.test')
    setSessionToken('token-1')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('sends the Aether session token header', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const result = await request<{ ok: boolean }>('GET', '/api/status')

    expect(result.ok).toBe(true)
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect((init.headers as Record<string, string>)['X-Aether-Session-Token']).toBe('token-1')
  })

  it('encodes documentation path segments without flattening slashes', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ path: 'sprint 20/00_overview.md' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.doc('sprint 20/00_overview.md')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/docs/sprint%2020/00_overview.md',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('posts plan clear requests for web clear commands', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ session_id: 's1', mode: 'agent', has_plan: false }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.clearPlan('s1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/plan/s1/clear',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('fetches task observer messages with a limit', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ task_id: 'task-1', messages: [], pending_messages: [], delivered_messages: [], total_count: 0, truncated: false }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.taskMessages('task-1', { limit: 25 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/tasks/task-1/messages?limit=25',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('fetches child task message streams with limits', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ task_id: 'task-1', streams: [], total_count: 0, truncated: false }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.taskChildMessages('task-1', { limit: 10, perTaskLimit: 5 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/tasks/task-1/children/messages?limit=10&per_task_limit=5',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('fetches task result artifacts', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ task_id: 'task-1', result_path: '/tmp/result.json', result: { summary: 'ok' } }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.taskResult('task-1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/tasks/task-1/result',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('refreshes the bootstrap token and retries once after a 401', async () => {
    setSessionToken('stale-token')
    let privateCalls = 0
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === 'http://aether.test/api/bootstrap') {
        return new Response(JSON.stringify({ session_token: 'fresh-token', auth_enabled: true, web: { enabled: true } }), { status: 200 })
      }
      if (url === 'http://aether.test/api/private') {
        privateCalls += 1
        if (privateCalls === 1) {
          expect((init?.headers as Record<string, string>)['X-Aether-Session-Token']).toBe('stale-token')
          return new Response(JSON.stringify({ error: { message: 'Unauthorized' } }), { status: 401 })
        }
        expect((init?.headers as Record<string, string>)['X-Aether-Session-Token']).toBe('fresh-token')
        return new Response(JSON.stringify({ ok: true }), { status: 200 })
      }
      return new Response('', { status: 404 })
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await request<{ ok: boolean }>('GET', '/api/private')

    expect(result.ok).toBe(true)
    expect(getSessionToken()).toBe('fresh-token')
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('maps structured errors to ApiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ error: { message: 'bad request' } }), {
          status: 400,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    await expect(request('GET', '/api/private')).rejects.toMatchObject({
      name: 'ApiError',
      status: 400,
      message: 'bad request',
    } satisfies Partial<ApiError>)
  })
})
