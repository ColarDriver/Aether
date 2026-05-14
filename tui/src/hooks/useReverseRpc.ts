import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type {
  ApprovalRequestParams,
  GatewayEvent,
  GatewayServerRequest,
  JsonObject,
  PermissionRequestParams
} from '../gatewayTypes.js'
import {
  OVERLAY_PRIORITY,
  overlayActions,
  type DismissReason,
  type OverlayKind
} from '../store/overlayStore.js'

// The server enforces its own timeout on reverse RPCs (default 120s in
// reverse_rpc.py). The client-side timer here is a UX safety net: if a modal
// is dismissed because its `deadline_ms` elapsed we want to send a deny
// response immediately so the engine thread unblocks fast.
const TIMEOUT_GRACE_MS = 250

export interface ReverseRpcResponder {
  respond(id: string, result: unknown): void
}

/**
 * Subscribes to the gateway event stream and converts every
 * `gateway.server_request` frame into an overlay push. The overlay's
 * `onDismiss` callback is what writes the JSON-RPC response back through the
 * client, so dismissing an overlay (for any reason) is the single point where
 * the reverse-RPC loop closes.
 *
 * Until PR 8b ships full ApprovalModal / PermissionModal, every reverse RPC
 * lands in a placeholder overlay; that placeholder still answers with the
 * conservative deny payload so the engine never hangs.
 */
export function useReverseRpc(client: GatewayClient): void {
  useEffect(() => {
    const handler = (event: GatewayEvent) => {
      if (event.type !== 'gateway.server_request') {
        return
      }
      handleReverseRpcEvent(event as GatewayServerRequest, client)
    }
    client.on('event', handler)
    return () => {
      client.off('event', handler)
    }
  }, [client])
}

/**
 * Pure routing logic for a single reverse-RPC frame. Exported for unit tests.
 *
 * The overlay's `onDismiss` callback owns:
 *   - calling `client.respond(id, payload)` exactly once
 *   - falling back to `defaultDeny(method)` when no payload is supplied
 *
 * Tests can drive this function directly with a stub responder; the React hook
 * is a thin subscriber on top.
 */
export function handleReverseRpcEvent(
  event: GatewayServerRequest,
  client: ReverseRpcResponder
): void {
  const { id, method, params } = event

  if (method === 'approval.request') {
    pushReverseRpcOverlay({
      kind: 'approval',
      id,
      method,
      payload: (params ?? {}) as ApprovalRequestParams,
      deadlineMs: getDeadlineMs(params),
      client
    })
    return
  }

  if (method === 'permission.request') {
    pushReverseRpcOverlay({
      kind: 'permission',
      id,
      method,
      payload: (params ?? {}) as PermissionRequestParams,
      deadlineMs: getDeadlineMs(params),
      client
    })
    return
  }

  // Unknown server-initiated method: respond with empty result so the gateway
  // does not block forever on a method we have not taught the TUI yet.
  client.respond(id, {})
}

interface PushOptions<TPayload> {
  kind: OverlayKind
  id: string
  method: string
  payload: TPayload
  deadlineMs: number | null
  client: ReverseRpcResponder
}

function pushReverseRpcOverlay<TPayload>(opts: PushOptions<TPayload>): void {
  let responded = false

  const respond = (payload: unknown): void => {
    if (responded) {
      return
    }
    responded = true
    try {
      opts.client.respond(opts.id, payload)
    } catch {
      // If respond throws (process gone, etc.) there is nothing useful to do
      // from inside the dismiss callback; the gateway will surface the loss
      // through its own error path.
    }
  }

  overlayActions.push({
    kind: opts.kind,
    id: opts.id,
    payload: opts.payload,
    createdAt: Date.now(),
    priority: OVERLAY_PRIORITY[opts.kind],
    onDismiss(reason: DismissReason, result?: unknown) {
      const payload =
        reason === 'commit' && result !== undefined ? result : defaultDeny(opts.method)
      respond(payload)
    }
  })

  if (opts.deadlineMs && opts.deadlineMs > 0) {
    const total = opts.deadlineMs + TIMEOUT_GRACE_MS
    setTimeout(() => {
      if (!overlayActions.has(opts.id)) {
        return
      }
      overlayActions.dismiss(opts.id, 'timeout')
    }, total)
  }
}

/**
 * Default conservative response for a reverse RPC that the user did not
 * explicitly answer (cancel / timeout / unknown method).
 *
 * Mirrors the payload shapes consumed in
 * aether/gateway/handlers/response_methods.py so the gateway can complete the
 * pending Future without raising.
 */
export function defaultDeny(method: string): JsonObject {
  if (method === 'approval.request') {
    return { confirmed: false }
  }
  if (method === 'permission.request') {
    // Mirrors Python `_reject_active_permission` / `_abort_all_permission_requests`
    // in `aether/cli/app.py`: a non-user-driven dismiss (process exit, panel
    // unmount, deadline) maps to ABORT with no user-supplied feedback.
    // `feedback` is the wire field consumed by `_decision_from_wire`; the
    // earlier `reason` key was simply dropped by Pydantic.
    return { type: 'abort', feedback: 'overlay dismissed' }
  }
  return {}
}

function getDeadlineMs(params: unknown): number | null {
  if (params && typeof params === 'object' && 'deadline_ms' in (params as object)) {
    const raw = (params as { deadline_ms?: unknown }).deadline_ms
    if (typeof raw === 'number' && Number.isFinite(raw) && raw > 0) {
      return raw
    }
  }
  return null
}
