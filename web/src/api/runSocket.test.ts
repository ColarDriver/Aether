// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getSessionToken, setBaseUrl, setSessionToken } from './client'
import { buildRunSocketUrl, RunSocketClient } from './runSocket'
import type { RunSocketFrame } from './types'

class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 3

  readyState = FakeWebSocket.CONNECTING
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null

  constructor(public url: string) {
    fakeSockets.push(this)
  }

  send(value: string) {
    this.sent.push(value)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.()
  }

  open() {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.()
  }

  serverClose() {
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.()
  }

  serverReject() {
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.()
  }

  serverMessage(frame: RunSocketFrame) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }
}

let fakeSockets: FakeWebSocket[] = []

describe('run socket client', () => {
  beforeEach(() => {
    setBaseUrl('http://127.0.0.1:9120')
    setSessionToken(null)
    fakeSockets = []
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('builds the run websocket URL from the REST base URL', () => {
    expect(buildRunSocketUrl()).toBe('ws://127.0.0.1:9120/api/runs/ws')
  })

  it('includes the local web token when configured', () => {
    setSessionToken('abc')
    expect(buildRunSocketUrl()).toBe('ws://127.0.0.1:9120/api/runs/ws?token=abc')
  })

  it('queues frames until the socket opens and dispatches server frames', () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    const client = new RunSocketClient()
    const handler = vi.fn()
    client.onFrame(handler)

    client.send({ type: 'run.cancel', payload: { session_id: 'session-1' } })

    expect(fakeSockets).toHaveLength(1)
    expect(fakeSockets[0].sent).toEqual([])

    fakeSockets[0].open()
    expect(JSON.parse(fakeSockets[0].sent[0])).toMatchObject({ type: 'run.cancel' })

    fakeSockets[0].serverMessage({ type: 'ready' })
    expect(handler).toHaveBeenCalledWith({ type: 'ready' })

    client.disconnect()
  })

  it('refreshes the bootstrap token after a pre-open websocket rejection', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ session_token: 'fresh-token', auth_enabled: true, web: { enabled: true } }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )))
    setSessionToken('stale-token')
    const client = new RunSocketClient()

    client.connect()
    expect(fakeSockets[0].url).toBe('ws://127.0.0.1:9120/api/runs/ws?token=stale-token')
    fakeSockets[0].serverReject()

    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(getSessionToken()).toBe('fresh-token')
    expect(fakeSockets).toHaveLength(2)
    expect(fakeSockets[1].url).toBe('ws://127.0.0.1:9120/api/runs/ws?token=fresh-token')

    client.disconnect()
  })

  it('keeps a queued run start while refreshing a rejected websocket token', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ session_token: 'fresh-token', auth_enabled: true, web: { enabled: true } }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )))
    setSessionToken('stale-token')
    const client = new RunSocketClient()

    client.startRun('session-1', 'hello after restart')
    expect(fakeSockets[0].sent).toEqual([])
    fakeSockets[0].serverReject()

    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(fakeSockets).toHaveLength(2)
    fakeSockets[1].open()

    expect(JSON.parse(fakeSockets[1].sent[0])).toMatchObject({
      type: 'run.start',
      payload: { session_id: 'session-1', user_message: 'hello after restart' },
    })

    client.disconnect()
  })

  it('bootstraps a token after a pre-open rejection without an existing token', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ session_token: 'fresh-token', auth_enabled: true, web: { enabled: true } }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )))
    const client = new RunSocketClient()

    client.connect()
    expect(fakeSockets[0].url).toBe('ws://127.0.0.1:9120/api/runs/ws')
    fakeSockets[0].serverReject()

    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(getSessionToken()).toBe('fresh-token')
    expect(fakeSockets).toHaveLength(2)
    expect(fakeSockets[1].url).toBe('ws://127.0.0.1:9120/api/runs/ws?token=fresh-token')

    client.disconnect()
  })

  it('reconnects after an unexpected close', () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeWebSocket)
    const client = new RunSocketClient()

    client.connect()
    fakeSockets[0].open()
    fakeSockets[0].serverClose()

    expect(fakeSockets).toHaveLength(1)
    vi.advanceTimersByTime(1000)
    expect(fakeSockets).toHaveLength(2)

    client.disconnect()
  })

  it('sends heartbeat pings while connected', () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeWebSocket)
    const client = new RunSocketClient()

    client.connect()
    fakeSockets[0].open()
    vi.advanceTimersByTime(30_000)

    expect(fakeSockets[0].sent.map((item) => JSON.parse(item).type)).toContain('ping')

    client.disconnect()
  })

  it('nests approval responses under the result field', () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    const client = new RunSocketClient()

    client.connect()
    fakeSockets[0].open()
    client.respondApproval('approval-1', { confirmed: true, answers: { q: 'yes' } })

    expect(JSON.parse(fakeSockets[0].sent.at(-1) ?? '{}')).toEqual({
      type: 'approval.respond',
      payload: {
        prompt_id: 'approval-1',
        result: { confirmed: true, answers: { q: 'yes' } },
      },
    })

    client.disconnect()
  })

  it('sends run attachments in the run.start payload', () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    const client = new RunSocketClient()

    client.connect()
    fakeSockets[0].open()
    client.startRun('session-1', 'look at this', [
      { type: 'image', name: 'plot.png', mimeType: 'image/png', data: 'data:image/png;base64,abc' },
    ])

    expect(JSON.parse(fakeSockets[0].sent.at(-1) ?? '{}')).toMatchObject({
      type: 'run.start',
      payload: {
        session_id: 'session-1',
        user_message: 'look at this',
        attachments: [
          { type: 'image', name: 'plot.png', mimeType: 'image/png', data: 'data:image/png;base64,abc' },
        ],
      },
    })

    client.disconnect()
  })
})
