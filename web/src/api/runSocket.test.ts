// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { setBaseUrl, setSessionToken } from './client'
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

  it('reconnects after an unexpected close', () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeWebSocket)
    const client = new RunSocketClient()

    client.connect()
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
})
