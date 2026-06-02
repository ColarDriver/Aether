import { getBaseUrl, getSessionToken, refreshSessionTokenFromBootstrapDocument } from './client'
import type { RunAttachment, RunSocketFrame } from './types'

type FrameHandler = (frame: RunSocketFrame) => void

const PING_INTERVAL_MS = 15_000
// How long to wait for a `pong` after a `ping` before treating the socket as
// dead. A half-open connection (network drop, proxy idle-kill, server-side
// detach) keeps readyState === OPEN, so `onclose` never fires and the stream
// silently stalls. Without this watchdog the only recovery is a manual refresh.
const PONG_TIMEOUT_MS = 10_000

export class RunSocketClient {
  private ws: WebSocket | null = null
  private handlers = new Set<FrameHandler>()
  private pending: RunSocketFrame[] = []
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private reconnectAttempt = 0
  private pingTimer: ReturnType<typeof setInterval> | null = null
  private pongTimer: ReturnType<typeof setTimeout> | null = null
  private closed = false
  private tokenRefresh: Promise<void> | null = null

  connect() {
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return
    }
    this.closed = false
    this.emit({ type: 'socket.connecting' })
    const ws = new WebSocket(buildRunSocketUrl())
    this.ws = ws
    let opened = false
    ws.onopen = () => {
      opened = true
      this.reconnectAttempt = 0
      this.emit({ type: 'socket.open' })
      this.startPing()
      while (this.pending.length > 0 && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(this.pending.shift()))
      }
    }
    ws.onmessage = (event) => {
      const frame = parseFrame(event.data)
      if (!frame) {
        // onmessage fired but the payload didn't parse into a frame. If the
        // console goes quiet here while data is still on the wire, the freeze is
        // a frontend parse drop, not a stalled stream.
        console.warn('[socket] onmessage unparsed', typeof event.data, String(event.data).slice(0, 80))
        return
      }
      // Any inbound traffic proves the socket is alive; a pong specifically
      // answers our heartbeat. Either way, clear the liveness watchdog.
      this.clearPongTimeout()
      this.emit(frame)
    }
    ws.onclose = () => {
      this.stopPing()
      if (this.closed) return
      this.emit({ type: 'socket.closed', payload: { reconnecting: true, opened } })
      if (!opened) {
        this.recoverSessionTokenAndReconnect()
        return
      }
      this.scheduleReconnect()
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

  private emit(frame: RunSocketFrame) {
    for (const handler of this.handlers) handler(frame)
  }

  startRun(sessionId: string, userMessage: string, attachments: RunAttachment[] = []) {
    const id = typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('').replace(/(.{8})(.{4})(.{4})(.{4})(.{12})/, '$1-$2-$3-$4-$5')
    this.send({
      type: 'run.start',
      id,
      payload: {
        session_id: sessionId,
        user_message: userMessage,
        options: { workspace_checkpoint: true },
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
    this.pingTimer = setInterval(() => {
      console.log('[socket] ping (readyState=' + (this.ws?.readyState ?? 'null') + ')')
      this.send({ type: 'ping' })
      this.armPongTimeout()
    }, PING_INTERVAL_MS)
  }

  private stopPing() {
    if (this.pingTimer) clearInterval(this.pingTimer)
    this.pingTimer = null
    this.clearPongTimeout()
  }

  private armPongTimeout() {
    if (this.pongTimer) return
    this.pongTimer = setTimeout(() => {
      this.pongTimer = null
      this.handleMissedPong()
    }, PONG_TIMEOUT_MS)
  }

  private clearPongTimeout() {
    if (this.pongTimer) clearTimeout(this.pongTimer)
    this.pongTimer = null
  }

  // The socket looks OPEN but the peer went silent (half-open connection):
  // frames have silently stopped arriving. Tear the dead socket down and
  // reconnect ourselves — `onclose` won't fire on its own. The fresh socket
  // gets a `ready` frame, which drives the transcript backfill, so the user
  // never has to refresh to recover the stalled stream.
  private handleMissedPong() {
    console.warn('[socket] missed pong -> tearing down and reconnecting')
    const ws = this.ws
    this.ws = null
    this.stopPing()
    if (ws) {
      ws.onopen = null
      ws.onmessage = null
      ws.onclose = null
      try {
        ws.close()
      } catch {
        // ignore — the underlying connection is already gone
      }
    }
    if (this.closed) return
    this.emit({ type: 'socket.closed', payload: { reconnecting: true, opened: true } })
    this.reconnectAttempt = 0
    this.scheduleReconnect()
  }

  private recoverSessionTokenAndReconnect() {
    if (this.tokenRefresh) return
    const previousToken = getSessionToken()
    this.tokenRefresh = refreshSessionTokenFromBootstrapDocument()
      .then((nextToken) => {
        if (this.closed) return
        if (nextToken && nextToken !== previousToken) {
          this.reconnectAttempt = 0
          this.connect()
          return
        }
        this.scheduleReconnect()
      })
      .catch(() => {
        if (!this.closed) this.scheduleReconnect()
      })
      .finally(() => {
        this.tokenRefresh = null
      })
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
