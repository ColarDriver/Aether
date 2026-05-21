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
  return typeof frame.payload?.session_id === 'string' ? frame.payload.session_id : ''
}

export function frameRunId(frame: RunSocketFrame): string {
  return typeof frame.payload?.run_id === 'string' ? frame.payload.run_id : ''
}
