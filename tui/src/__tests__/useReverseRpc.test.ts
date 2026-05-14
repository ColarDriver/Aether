import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type {
  ApprovalRequestParams,
  GatewayServerRequest,
  PermissionRequestParams
} from '../gatewayTypes.js'
import {
  defaultDeny,
  handleReverseRpcEvent,
  type ReverseRpcResponder
} from '../hooks/useReverseRpc.js'
import { overlayActions, overlayStack, topOverlay } from '../store/overlayStore.js'

function makeResponder() {
  return {
    respond: vi.fn<ReverseRpcResponder['respond']>()
  }
}

const APPROVAL_PARAMS: ApprovalRequestParams = {
  kind: 'plan',
  session_id: 'ses_1',
  run_id: 'run_1',
  plan_text: 'do the thing',
  questions: [],
  deadline_ms: 0
}

const PERMISSION_PARAMS: PermissionRequestParams = {
  session_id: 'ses_1',
  run_id: 'run_1',
  request: {
    tool_call_id: 'tc_1',
    tool_name: 'shell',
    arguments: { command: 'ls' },
    category: 'shell',
    risk: 'medium',
    allow_session: true
  },
  deadline_ms: 0
}

describe('useReverseRpc — handleReverseRpcEvent', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('routes approval.request into an approval overlay', () => {
    const client = makeResponder()
    const event: GatewayServerRequest = {
      type: 'gateway.server_request',
      id: 'srv_app_1',
      method: 'approval.request',
      params: APPROVAL_PARAMS
    }

    handleReverseRpcEvent(event, client)

    expect(overlayStack.get()).toHaveLength(1)
    const top = topOverlay()
    expect(top?.kind).toBe('approval')
    expect(top?.id).toBe('srv_app_1')
    expect(top?.payload).toBe(APPROVAL_PARAMS)
    expect(client.respond).not.toHaveBeenCalled()
  })

  it('routes permission.request into a permission overlay', () => {
    const client = makeResponder()
    const event: GatewayServerRequest = {
      type: 'gateway.server_request',
      id: 'srv_perm_2',
      method: 'permission.request',
      params: PERMISSION_PARAMS
    }

    handleReverseRpcEvent(event, client)

    const top = topOverlay()
    expect(top?.kind).toBe('permission')
    expect(top?.id).toBe('srv_perm_2')
  })

  it('cancel dismiss responds with the conservative deny payload', () => {
    const client = makeResponder()
    handleReverseRpcEvent(
      {
        type: 'gateway.server_request',
        id: 'srv_app_1',
        method: 'approval.request',
        params: APPROVAL_PARAMS
      },
      client
    )

    overlayActions.dismissTop('cancel')

    expect(client.respond).toHaveBeenCalledOnce()
    expect(client.respond).toHaveBeenCalledWith('srv_app_1', { confirmed: false })
  })

  it('commit dismiss forwards the user-supplied payload', () => {
    const client = makeResponder()
    handleReverseRpcEvent(
      {
        type: 'gateway.server_request',
        id: 'srv_perm_3',
        method: 'permission.request',
        params: PERMISSION_PARAMS
      },
      client
    )

    overlayActions.dismissTop('commit', { type: 'allow_session', rule: { foo: 'bar' } })

    expect(client.respond).toHaveBeenCalledWith('srv_perm_3', {
      type: 'allow_session',
      rule: { foo: 'bar' }
    })
  })

  it('client-side timeout fires defaultDeny when deadline_ms elapses without a user answer', () => {
    const client = makeResponder()
    handleReverseRpcEvent(
      {
        type: 'gateway.server_request',
        id: 'srv_perm_4',
        method: 'permission.request',
        params: { ...PERMISSION_PARAMS, deadline_ms: 1000 }
      },
      client
    )

    expect(client.respond).not.toHaveBeenCalled()

    vi.advanceTimersByTime(999)
    expect(client.respond).not.toHaveBeenCalled()

    vi.advanceTimersByTime(500)
    expect(client.respond).toHaveBeenCalledOnce()
    expect(client.respond).toHaveBeenCalledWith('srv_perm_4', {
      type: 'abort',
      feedback: 'overlay dismissed'
    })
  })

  it('a manual dismiss before the deadline disarms the timeout', () => {
    const client = makeResponder()
    handleReverseRpcEvent(
      {
        type: 'gateway.server_request',
        id: 'srv_app_5',
        method: 'approval.request',
        params: { ...APPROVAL_PARAMS, deadline_ms: 1000 }
      },
      client
    )

    overlayActions.dismissTop('commit', { confirmed: true, answers: [] })
    expect(client.respond).toHaveBeenCalledOnce()
    expect(client.respond).toHaveBeenCalledWith('srv_app_5', {
      confirmed: true,
      answers: []
    })

    vi.advanceTimersByTime(5000)
    expect(client.respond).toHaveBeenCalledOnce()
  })

  it('responds with an empty result for unknown server-initiated methods', () => {
    const client = makeResponder()
    handleReverseRpcEvent(
      {
        type: 'gateway.server_request',
        id: 'srv_misc_9',
        method: 'mystery.method',
        params: { foo: 'bar' }
      },
      client
    )

    expect(client.respond).toHaveBeenCalledOnce()
    expect(client.respond).toHaveBeenCalledWith('srv_misc_9', {})
    expect(overlayStack.get()).toHaveLength(0)
  })

  it('only responds once even if the overlay is dismissed twice', () => {
    const client = makeResponder()
    handleReverseRpcEvent(
      {
        type: 'gateway.server_request',
        id: 'srv_app_6',
        method: 'approval.request',
        params: APPROVAL_PARAMS
      },
      client
    )

    const overlay = overlayActions.get('srv_app_6')
    expect(overlay).not.toBeNull()
    overlay?.onDismiss('cancel')
    overlay?.onDismiss('cancel')

    expect(client.respond).toHaveBeenCalledOnce()
  })
})

describe('defaultDeny', () => {
  it('approval.request → confirmed:false', () => {
    expect(defaultDeny('approval.request')).toEqual({ confirmed: false })
  })

  it('permission.request → type:abort with feedback (matches Python user_abort branch)', () => {
    expect(defaultDeny('permission.request')).toEqual({
      type: 'abort',
      feedback: 'overlay dismissed'
    })
  })

  it('unknown method → empty object', () => {
    expect(defaultDeny('mystery.thing')).toEqual({})
  })
})
