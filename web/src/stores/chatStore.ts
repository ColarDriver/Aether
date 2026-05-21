import { create } from 'zustand'
import { api } from '../api/client'
import { runSocket } from '../api/runSocket'
import type { RunSocketFrame } from '../api/types'
import type { ChatBlock, RunStatusSnapshot, TokenUsage } from '../chat-rendering'
import { normalizeTranscript, reduceRunFrame, resolvePromptInBlocks } from '../chat-rendering'

export type PermissionPrompt = {
  promptId: string
  runId: string
  request: {
    tool_name?: string
    tool_call_id?: string
    arguments?: Record<string, unknown>
    category?: string
    risk?: string
    reason?: string | null
    allow_session?: boolean
    preview?: {
      title?: string
      subtitle?: string | null
      body?: string | null
      diff?: string | null
      path?: string | null
      command?: string | null
      metadata?: Record<string, unknown>
    } | null
  }
}

export type ApprovalPrompt = {
  promptId: string
  kind: string
  sessionId: string
  runId: string
  planText?: string | null
  planPath?: string | null
  questions: Array<Record<string, unknown>>
}

type ChatState = {
  connected: boolean
  frames: RunSocketFrame[]
  activeRunId: string | null
  blocksBySession: Record<string, ChatBlock[]>
  tokenUsageByRun: Record<string, TokenUsage>
  statusByRun: Record<string, RunStatusSnapshot>
  pendingPermission: PermissionPrompt | null
  pendingApproval: ApprovalPrompt | null
  loadTranscript: (sessionId: string) => Promise<void>
  connect: () => void
  startRun: (sessionId: string, message: string) => string
  cancelRun: (sessionId: string) => void
  appendLocalNotice: (sessionId: string, content: string) => void
  appendLocalError: (sessionId: string, message: string) => void
  respondPermission: (decision: Record<string, unknown>) => void
  respondApproval: (result: Record<string, unknown>) => void
}

let unsubscribeSocket: (() => void) | null = null
let localBlockSequence = 0

export const useChatStore = create<ChatState>((set, get) => ({
  connected: false,
  frames: [],
  activeRunId: null,
  blocksBySession: {},
  tokenUsageByRun: {},
  statusByRun: {},
  pendingPermission: null,
  pendingApproval: null,
  loadTranscript: async (sessionId) => {
    const { messages } = await api.sessionMessages(sessionId)
    const blocks = normalizeTranscript(sessionId, messages)
    set((state) => ({
      blocksBySession: {
        ...state.blocksBySession,
        [sessionId]: blocks,
      },
    }))
  },
  connect: () => {
    runSocket.connect()
    if (unsubscribeSocket) return
    unsubscribeSocket = runSocket.onFrame((frame) => {
      set((state) => ({
        frames: [...state.frames, frame],
        connected: frame.type === 'ready' ? true : state.connected,
        activeRunId:
          frame.type === 'run.accepted' && typeof frame.payload?.run_id === 'string'
            ? frame.payload.run_id
            : state.activeRunId,
      }))
      applyRenderFrame(frame, set)
      applyPromptFrame(frame, set)
    })
  },
  startRun: (sessionId, message) => {
    get().connect()
    const runId = runSocket.startRun(sessionId, message)
    const timestamp = Date.now()
    const userBlock: ChatBlock = {
      id: 'user-' + timestamp,
      sessionId,
      runId,
      timestamp,
      source: 'optimistic',
      kind: 'user_message',
      content: message,
    }
    set((state) => ({
      blocksBySession: {
        ...state.blocksBySession,
        [sessionId]: [...(state.blocksBySession[sessionId] ?? []), userBlock],
      },
      activeRunId: runId,
    }))
    return runId
  },
  cancelRun: (sessionId) => {
    runSocket.cancelRun(sessionId, get().activeRunId ?? undefined)
  },
  appendLocalNotice: (sessionId, content) => {
    set((state) => appendLocalBlock(state, {
      id: nextLocalBlockId('notice'),
      sessionId,
      timestamp: Date.now(),
      source: 'optimistic',
      kind: 'system_notice',
      content,
    }))
  },
  appendLocalError: (sessionId, message) => {
    set((state) => appendLocalBlock(state, {
      id: nextLocalBlockId('error'),
      sessionId,
      timestamp: Date.now(),
      source: 'optimistic',
      kind: 'error',
      message,
      code: 'web_slash_command',
    }))
  },
  respondPermission: (decision) => {
    const prompt = get().pendingPermission
    if (!prompt) return
    runSocket.respondPermission(prompt.promptId, decision)
    set((state) => ({
      pendingPermission: null,
      blocksBySession: resolvePromptInBlocks(state.blocksBySession, prompt.promptId, decision),
    }))
  },
  respondApproval: (result) => {
    const prompt = get().pendingApproval
    if (!prompt) return
    runSocket.respondApproval(prompt.promptId, result)
    set((state) => ({
      pendingApproval: null,
      blocksBySession: resolvePromptInBlocks(state.blocksBySession, prompt.promptId, result),
    }))
  },
}))

function nextLocalBlockId(prefix: string): string {
  localBlockSequence += 1
  return prefix + '-' + Date.now() + '-' + localBlockSequence
}

function appendLocalBlock(state: ChatState, block: ChatBlock): Partial<ChatState> {
  return {
    blocksBySession: {
      ...state.blocksBySession,
      [block.sessionId]: [...(state.blocksBySession[block.sessionId] ?? []), block],
    },
  }
}

function applyPromptFrame(
  frame: RunSocketFrame,
  set: (partial: ChatState | Partial<ChatState> | ((state: ChatState) => ChatState | Partial<ChatState>)) => void,
) {
  const payload = frame.payload ?? {}
  const sessionId = asString(payload.session_id)
  const runId = asString(payload.run_id)
  if (!sessionId) return

  if (frame.type === 'permission.requested') {
    const request = asRecord(payload.request)
    set({
      pendingPermission: {
        promptId: asString(payload.prompt_id),
        runId,
        request: {
          tool_name: asString(request.tool_name),
          tool_call_id: asString(request.tool_call_id),
          arguments: asRecord(request.arguments),
          category: asString(request.category),
          risk: asString(request.risk),
          reason: asString(request.reason) || null,
          allow_session: Boolean(request.allow_session),
          preview: previewFromUnknown(request.preview),
        },
      },
    })
    return
  }

  if (frame.type === 'approval.requested') {
    set({
      pendingApproval: {
        promptId: asString(payload.prompt_id),
        kind: asString(payload.kind) || 'plan',
        sessionId,
        runId,
        planText: asString(payload.plan_text) || null,
        planPath: asString(payload.plan_path) || null,
        questions: Array.isArray(payload.questions) ? payload.questions.filter(isRecord) : [],
      },
    })
    return
  }

  if (frame.type === 'prompt.resolved') {
    set({ pendingPermission: null, pendingApproval: null })
  }
}

function applyRenderFrame(
  frame: RunSocketFrame,
  set: (partial: ChatState | Partial<ChatState> | ((state: ChatState) => ChatState | Partial<ChatState>)) => void,
) {
  set((state) => {
    const next = reduceRunFrame(
      {
        blocksBySession: state.blocksBySession,
        activeRunId: state.activeRunId,
        tokenUsageByRun: state.tokenUsageByRun,
        statusByRun: state.statusByRun,
        pendingPermissionBlock: null,
        pendingApprovalBlock: null,
      },
      frame,
    )
    return {
      blocksBySession: next.blocksBySession,
      activeRunId: next.activeRunId,
      tokenUsageByRun: next.tokenUsageByRun,
      statusByRun: next.statusByRun,
    }
  })
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function previewFromUnknown(value: unknown): PermissionPrompt['request']['preview'] {
  if (!isRecord(value)) return null
  return {
    title: asString(value.title),
    subtitle: asString(value.subtitle) || null,
    body: asString(value.body) || null,
    diff: asString(value.diff) || null,
    path: asString(value.path) || null,
    command: asString(value.command) || null,
    metadata: asRecord(value.metadata),
  }
}
