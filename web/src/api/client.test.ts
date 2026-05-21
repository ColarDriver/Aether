// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, request, setBaseUrl, setSessionToken } from './client'

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
