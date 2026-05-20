// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, request, setBaseUrl, setSessionToken } from './client'

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
