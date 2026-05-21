import { create } from 'zustand'
import { api } from '../api/client'
import { runSocket } from '../api/runSocket'
import type { RunSocketFrame, TranscriptMessage } from '../api/types'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant' | 'tool' | 'system'
  text: string
  isStreaming?: boolean
  isError?: boolean
}

export type ToolBlock = {
  id: string
  sessionId: string
  runId: string
  toolCallId: string
  toolName: string
  arguments: Record<string, unknown>
  status: 'running' | 'finished'
  content?: string
  isError?: boolean
  metadata?: Record<string, unknown>
}

type TokenUsage = {
  input_tokens?: number
  output_tokens?: number
  cache_read_tokens?: number
  cache_write_tokens?: number
}

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
  messagesBySession: Record<string, ChatMessage[]>
  toolsBySession: Record<string, ToolBlock[]>
  tokenUsageByRun: Record<string, TokenUsage>
  pendingPermission: PermissionPrompt | null
  pendingApproval: ApprovalPrompt | null
  loadTranscript: (sessionId: string) => Promise<void>
  connect: () => void
  startRun: (sessionId: string, message: string) => string
  cancelRun: (sessionId: string) => void
  respondPermission: (decision: Record<string, unknown>) => void
  respondApproval: (result: Record<string, unknown>) => void
}

let unsubscribeSocket: (() => void) | null = null

export const useChatStore = create<ChatState>((set, get) => ({
  connected: false,
  frames: [],
  activeRunId: null,
  messagesBySession: {},
  toolsBySession: {},
  tokenUsageByRun: {},
  pendingPermission: null,
  pendingApproval: null,
  loadTranscript: async (sessionId) => {
    const { messages } = await api.sessionMessages(sessionId)
    set((state) => ({
      messagesBySession: {
        ...state.messagesBySession,
        [sessionId]: messages.map((message, index) => transcriptToChatMessage(message, index)),
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
      applyRunFrame(frame, set)
    })
  },
  startRun: (sessionId, message) => {
    get().connect()
    const userMessage: ChatMessage = {
      id: 'user-' + Date.now(),
      role: 'user',
      text: message,
    }
    set((state) => ({
      messagesBySession: {
        ...state.messagesBySession,
        [sessionId]: [...(state.messagesBySession[sessionId] ?? []), userMessage],
      },
    }))
    const runId = runSocket.startRun(sessionId, message)
    set({ activeRunId: runId })
    return runId
  },
  cancelRun: (sessionId) => {
    runSocket.cancelRun(sessionId, get().activeRunId ?? undefined)
  },
  respondPermission: (decision) => {
    const prompt = get().pendingPermission
    if (!prompt) return
    runSocket.respondPermission(prompt.promptId, decision)
    set({ pendingPermission: null })
  },
  respondApproval: (result) => {
    const prompt = get().pendingApproval
    if (!prompt) return
    runSocket.respondApproval(prompt.promptId, result)
    set({ pendingApproval: null })
  },
}))

function transcriptToChatMessage(message: TranscriptMessage, index: number): ChatMessage {
  return {
    id: 'persisted-' + index,
    role: message.role,
    text: message.text ?? '',
    isError: message.is_error,
  }
}

function applyRunFrame(
  frame: RunSocketFrame,
  set: (partial: ChatState | Partial<ChatState> | ((state: ChatState) => ChatState | Partial<ChatState>)) => void,
) {
  const payload = frame.payload ?? {}
  const sessionId = asString(payload.session_id)
  const runId = asString(payload.run_id)
  if (!sessionId) return

  if (frame.type === 'assistant.delta') {
    const text = asString(payload.text)
    if (!text) return
    set((state) => ({
      messagesBySession: {
        ...state.messagesBySession,
        [sessionId]: appendAssistantDelta(state.messagesBySession[sessionId] ?? [], runId, text),
      },
    }))
    return
  }

  if (frame.type === 'run.finished' || frame.type === 'run.cancelled' || frame.type === 'run.failed') {
    set((state) => ({
      activeRunId: frame.type === 'run.failed' ? state.activeRunId : null,
      messagesBySession: {
        ...state.messagesBySession,
        [sessionId]: finishAssistantMessage(state.messagesBySession[sessionId] ?? [], runId, frame.type),
      },
    }))
    return
  }

  if (frame.type === 'tool.started') {
    const toolCallId = asString(payload.tool_call_id)
    if (!toolCallId) return
    set((state) => ({
      toolsBySession: {
        ...state.toolsBySession,
        [sessionId]: [
          ...(state.toolsBySession[sessionId] ?? []),
          {
            id: runId + '-' + toolCallId,
            sessionId,
            runId,
            toolCallId,
            toolName: asString(payload.tool_name) || 'tool',
            arguments: asRecord(payload.arguments),
            status: 'running',
          },
        ],
      },
    }))
    return
  }

  if (frame.type === 'tool.finished') {
    const toolCallId = asString(payload.tool_call_id)
    set((state) => ({
      toolsBySession: {
        ...state.toolsBySession,
        [sessionId]: (state.toolsBySession[sessionId] ?? []).map((tool) =>
          tool.toolCallId === toolCallId
            ? {
                ...tool,
                status: 'finished',
                content: asString(payload.content),
                isError: Boolean(payload.is_error),
                metadata: asRecord(payload.metadata),
              }
            : tool,
        ),
      },
    }))
    return
  }

  if (frame.type === 'token.usage' && runId) {
    set((state) => ({
      tokenUsageByRun: {
        ...state.tokenUsageByRun,
        [runId]: {
          input_tokens: asNumber(payload.input_tokens),
          output_tokens: asNumber(payload.output_tokens),
          cache_read_tokens: asNumber(payload.cache_read_tokens),
          cache_write_tokens: asNumber(payload.cache_write_tokens),
        },
      },
    }))
    return
  }

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

function appendAssistantDelta(messages: ChatMessage[], runId: string, text: string): ChatMessage[] {
  const last = messages[messages.length - 1]
  if (last?.role === 'assistant' && last.isStreaming) {
    return [...messages.slice(0, -1), { ...last, text: last.text + text }]
  }
  return [...messages, { id: 'assistant-' + (runId || Date.now()), role: 'assistant', text, isStreaming: true }]
}

function finishAssistantMessage(messages: ChatMessage[], runId: string, eventType: string): ChatMessage[] {
  const last = messages[messages.length - 1]
  if (last?.role === 'assistant' && last.isStreaming) {
    return [...messages.slice(0, -1), { ...last, isStreaming: false, isError: eventType === 'run.failed' }]
  }
  if (eventType === 'run.failed') {
    return [...messages, { id: 'assistant-error-' + runId, role: 'assistant', text: 'Run failed.', isError: true }]
  }
  return messages
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined
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
