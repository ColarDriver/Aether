import type { RunSocketFrame } from '../api/types'
import type { ApprovalRequestBlock, ChatBlock, PermissionRequestBlock, TokenUsage } from './blocks'

export type RunStatusSnapshot = {
  runId: string
  sessionId: string
  state: string
  detail?: string | null
  elapsedMs?: number
  tokens?: TokenUsage
}

export type ChatRenderState = {
  blocksBySession: Record<string, ChatBlock[]>
  activeRunId: string | null
  tokenUsageByRun: Record<string, TokenUsage>
  statusByRun: Record<string, RunStatusSnapshot>
  pendingPermissionBlock: PermissionRequestBlock | null
  pendingApprovalBlock: ApprovalRequestBlock | null
}

export function createChatRenderState(): ChatRenderState {
  return {
    blocksBySession: {},
    activeRunId: null,
    tokenUsageByRun: {},
    statusByRun: {},
    pendingPermissionBlock: null,
    pendingApprovalBlock: null,
  }
}

export function frameSessionId(frame: RunSocketFrame): string {
  if (typeof frame.payload?.session_id === 'string') return frame.payload.session_id
  const request = frame.payload?.request
  if (isRecord(request) && typeof request.session_id === 'string') return request.session_id
  return ''
}

export function frameRunId(frame: RunSocketFrame): string {
  return typeof frame.payload?.run_id === 'string' ? frame.payload.run_id : ''
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
