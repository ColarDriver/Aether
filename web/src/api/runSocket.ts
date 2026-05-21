import { getBaseUrl, getSessionToken } from './client'
import type { RunAttachment, RunSocketFrame } from './types'

type FrameHandler = (frame: RunSocketFrame) => void

export class RunSocketClient {
  private ws: WebSocket | null = null
  private handlers = new Set<FrameHandler>()
  private pending: RunSocketFrame[] = []
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private reconnectAttempt = 0
  private pingTimer: ReturnType<typeof setInterval> | null = null
  private closed = false

  connect() {
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return
    }
    this.closed = false
    const ws = new WebSocket(buildRunSocketUrl())
    this.ws = ws
    ws.onopen = () => {
      this.reconnectAttempt = 0
      this.startPing()
      while (this.pending.length > 0 && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(this.pending.shift()))
      }
    }
    ws.onmessage = (event) => {
      const frame = parseFrame(event.data)
      if (!frame) return
      for (const handler of this.handlers) handler(frame)
    }
    ws.onclose = () => {
      this.stopPing()
      if (!this.closed) this.scheduleReconnect()
    }
  }

  disconnect() {
    this.closed = true
    this.stopPing()
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = null
    this.ws?.close()
    this.ws = null
    this.pending = []
  }

  send(frame: RunSocketFrame) {
    if (!this.ws || this.ws.readyState === WebSocket.CLOSED) this.connect()
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(frame))
      return
    }
    this.pending.push(frame)
  }

  onFrame(handler: FrameHandler) {
    this.handlers.add(handler)
    return () => this.handlers.delete(handler)
  }

  startRun(sessionId: string, userMessage: string, attachments: RunAttachment[] = []) {
    const id = crypto.randomUUID()
    this.send({
      type: 'run.start',
      id,
      payload: {
        session_id: sessionId,
        user_message: userMessage,
        ...(attachments.length > 0 ? { attachments } : {}),
      },
    })
    return id
  }

  cancelRun(sessionId: string, runId?: string) {
    this.send({ type: 'run.cancel', payload: { session_id: sessionId, run_id: runId } })
  }

  respondPermission(promptId: string, decision: Record<string, unknown>) {
    this.send({ type: 'permission.respond', payload: { prompt_id: promptId, decision } })
  }

  respondApproval(promptId: string, result: Record<string, unknown>) {
    this.send({ type: 'approval.respond', payload: { prompt_id: promptId, result } })
  }

  private startPing() {
    this.stopPing()
    this.pingTimer = setInterval(() => this.send({ type: 'ping' }), 30_000)
  }

  private stopPing() {
    if (this.pingTimer) clearInterval(this.pingTimer)
    this.pingTimer = null
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return
    const delay = Math.min(1000 * 2 ** this.reconnectAttempt, 30_000)
    this.reconnectAttempt += 1
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.connect()
    }, delay)
  }
}

export function buildRunSocketUrl() {
  const base = getBaseUrl()
  const url = new URL(base || window.location.origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  const prefix = url.pathname === '/' ? '' : url.pathname.replace(/\/$/, '')
  url.pathname = prefix + '/api/runs/ws'
  const token = getSessionToken()
  if (token) url.searchParams.set('token', token)
  return url.toString()
}

function parseFrame(value: unknown): RunSocketFrame | null {
  try {
    const parsed = JSON.parse(String(value))
    if (parsed && typeof parsed === 'object' && typeof parsed.type === 'string') return parsed as RunSocketFrame
  } catch {
    return null
  }
  return null
}

export const runSocket = new RunSocketClient()
