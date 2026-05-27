import { create } from 'zustand'
import { api } from '../api/client'
import { runSocket } from '../api/runSocket'
import type { RunSocketFrame, TaskSummary, TranscriptMessage } from '../api/types'
import type { ChatAttachment, ChatBlock, RunStatusSnapshot, TokenUsage } from '../chat-rendering'
import { frameSessionId, normalizeTranscript, reduceRunFrame, resolvePromptInBlocks } from '../chat-rendering'
import { useSessionStore } from './sessionStore'
import { useTaskStore } from './taskStore'

export type PermissionPrompt = {
  promptId: string
  sessionId: string
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

export type SocketConnectionState = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'disconnected'

type ChatState = {
  connected: boolean
  socketState: SocketConnectionState
  socketDetail: string | null
  frames: RunSocketFrame[]
  activeRunId: string | null
  blocksBySession: Record<string, ChatBlock[]>
  tokenUsageByRun: Record<string, TokenUsage>
  statusByRun: Record<string, RunStatusSnapshot>
  pendingPermission: PermissionPrompt | null
  pendingApproval: ApprovalPrompt | null
  pendingPermissionQueue: PermissionPrompt[]
  pendingApprovalQueue: ApprovalPrompt[]
  loadTranscript: (sessionId: string) => Promise<void>
  replaceTranscript: (sessionId: string, messages: TranscriptMessage[]) => void
  connect: () => void
  startRun: (sessionId: string, message: string, attachments?: ChatAttachment[]) => string
  cancelRun: (sessionId: string) => void
  appendLocalNotice: (sessionId: string, content: string) => void
  appendLocalError: (sessionId: string, message: string) => void
  clearSession: (sessionId: string) => void
  respondPermission: (decision: Record<string, unknown>) => void
  respondApproval: (result: Record<string, unknown>) => void
}

let unsubscribeSocket: (() => void) | null = null
let localBlockSequence = 0

export const useChatStore = create<ChatState>((set, get) => ({
  connected: false,
  socketState: 'idle',
  socketDetail: null,
  frames: [],
  activeRunId: null,
  blocksBySession: {},
  tokenUsageByRun: {},
  statusByRun: {},
  pendingPermission: null,
  pendingApproval: null,
  pendingPermissionQueue: [],
  pendingApprovalQueue: [],
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
  replaceTranscript: (sessionId, messages) => {
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
        connected: frame.type === 'ready' || frame.type === 'socket.open'
          ? true
          : frame.type === 'socket.closed'
            ? false
            : state.connected,
        ...socketConnectionStateFromFrame(frame, state),
        activeRunId:
          frame.type === 'run.accepted' && typeof frame.payload?.run_id === 'string'
            ? frame.payload.run_id
            : state.activeRunId,
      }))
      applyRenderFrame(frame, set)
      applyPromptFrame(frame, set)
      applyTaskFrame(frame)
      if (frame.type === 'ready') void backfillActiveSessionAfterReady(set)
    })
  },
  startRun: (sessionId, message, attachments = []) => {
    get().connect()
    const runId = runSocket.startRun(sessionId, message, attachments)
    const timestamp = Date.now()
    const userBlock: ChatBlock = {
      id: 'user-' + timestamp,
      sessionId,
      runId,
      timestamp,
      source: 'optimistic',
      kind: 'user_message',
      content: message,
      ...(attachments.length > 0 ? { attachments } : {}),
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
  clearSession: (sessionId) => {
    set((state) => {
      const permissionQueue = state.pendingPermissionQueue.filter((prompt) => prompt.sessionId !== sessionId)
      const approvalQueue = state.pendingApprovalQueue.filter((prompt) => prompt.sessionId !== sessionId)
      return {
        blocksBySession: {
          ...state.blocksBySession,
          [sessionId]: [],
        },
        pendingPermissionQueue: permissionQueue,
        pendingApprovalQueue: approvalQueue,
        pendingPermission: permissionQueue[0] ?? null,
        pendingApproval: approvalQueue[0] ?? null,
        activeRunId: null,
      }
    })
  },
  respondPermission: (decision) => {
    const prompt = get().pendingPermission
    if (!prompt) return
    runSocket.respondPermission(prompt.promptId, decision)
    set((state) => {
      const queue = state.pendingPermissionQueue.filter((item) => item.promptId !== prompt.promptId)
      return {
        pendingPermissionQueue: queue,
        pendingPermission: queue[0] ?? null,
        blocksBySession: resolvePromptInBlocks(state.blocksBySession, prompt.promptId, decision),
      }
    })
  },
  respondApproval: (result) => {
    const prompt = get().pendingApproval
    if (!prompt) return
    runSocket.respondApproval(prompt.promptId, result)
    set((state) => {
      const queue = state.pendingApprovalQueue.filter((item) => item.promptId !== prompt.promptId)
      return {
        pendingApprovalQueue: queue,
        pendingApproval: queue[0] ?? null,
        blocksBySession: resolvePromptInBlocks(state.blocksBySession, prompt.promptId, result),
      }
    })
  },
}))

function socketConnectionStateFromFrame(frame: RunSocketFrame, state: ChatState): Pick<ChatState, 'socketState' | 'socketDetail'> {
  if (frame.type === 'ready' || frame.type === 'socket.open') {
    return { socketState: 'connected', socketDetail: null }
  }
  if (frame.type === 'socket.connecting') {
    return { socketState: state.connected ? 'connected' : 'connecting', socketDetail: 'Opening run stream' }
  }
  if (frame.type === 'socket.closed') {
    const payload = frame.payload ?? {}
    const reconnecting = payload.reconnecting !== false
    const detail = typeof payload.reason === 'string' && payload.reason.trim()
      ? payload.reason.trim()
      : reconnecting
        ? 'Reconnecting run stream'
        : 'Run stream disconnected'
    return {
      socketState: reconnecting ? 'reconnecting' : 'disconnected',
      socketDetail: detail,
    }
  }
  return { socketState: state.socketState, socketDetail: state.socketDetail }
}

function nextLocalBlockId(prefix: string): string {
  localBlockSequence += 1
  return prefix + '-' + Date.now() + '-' + localBlockSequence
}

function enqueuePermissionPrompt(state: ChatState, prompt: PermissionPrompt): Partial<ChatState> {
  const queue = replaceOrAppendPrompt(state.pendingPermissionQueue, prompt)
  return { pendingPermissionQueue: queue, pendingPermission: queue[0] ?? null }
}

function enqueueApprovalPrompt(state: ChatState, prompt: ApprovalPrompt): Partial<ChatState> {
  const queue = replaceOrAppendPrompt(state.pendingApprovalQueue, prompt)
  return { pendingApprovalQueue: queue, pendingApproval: queue[0] ?? null }
}

function replaceOrAppendPrompt<T extends { promptId: string }>(queue: T[], prompt: T): T[] {
  const index = queue.findIndex((item) => item.promptId === prompt.promptId)
  if (index < 0) return [...queue, prompt]
  return queue.map((item, itemIndex) => itemIndex === index ? prompt : item)
}

function resolvePromptQueues(state: ChatState, promptId: string): Partial<ChatState> {
  if (!promptId) return {}
  const permissionQueue = state.pendingPermissionQueue.filter((prompt) => prompt.promptId !== promptId)
  const approvalQueue = state.pendingApprovalQueue.filter((prompt) => prompt.promptId !== promptId)
  return {
    pendingPermissionQueue: permissionQueue,
    pendingApprovalQueue: approvalQueue,
    pendingPermission: permissionQueue[0] ?? null,
    pendingApproval: approvalQueue[0] ?? null,
  }
}

function appendLocalBlock(state: ChatState, block: ChatBlock): Partial<ChatState> {
  return {
    blocksBySession: {
      ...state.blocksBySession,
      [block.sessionId]: [...(state.blocksBySession[block.sessionId] ?? []), block],
    },
  }
}

async function backfillActiveSessionAfterReady(
  set: (partial: ChatState | Partial<ChatState> | ((state: ChatState) => ChatState | Partial<ChatState>)) => void,
): Promise<void> {
  const sessionId = useSessionStore.getState().activeSessionId
  if (!sessionId) return

  const [messagesResult, tasksResult] = await Promise.allSettled([
    api.sessionMessages(sessionId),
    api.sessionTasks(sessionId, { limit: 100 }),
    useSessionStore.getState().loadSessions(),
  ]).then((results) => [results[0], results[1]] as const)

  if (messagesResult.status === 'fulfilled') {
    const backfilled = normalizeTranscript(sessionId, messagesResult.value.messages)
    set((state) => ({
      blocksBySession: {
        ...state.blocksBySession,
        [sessionId]: mergeBackfilledBlocks(state.blocksBySession[sessionId] ?? [], backfilled),
      },
    }))
  }

  if (tasksResult.status === 'fulfilled') {
    useTaskStore.getState().applyTaskSnapshot(sessionId, tasksResult.value.tasks)
  }
}

function mergeBackfilledBlocks(existing: ChatBlock[], backfilled: ChatBlock[]): ChatBlock[] {
  if (existing.length === 0) return backfilled
  const next = [...backfilled]
  for (const block of existing) {
    if (block.source === 'transcript') continue
    if (hasEquivalentBackfilledBlock(backfilled, block)) continue
    next.push(block)
  }
  return next
}

function hasEquivalentBackfilledBlock(backfilled: ChatBlock[], block: ChatBlock): boolean {
  if (block.kind === 'user_message') {
    return backfilled.some((item) => item.kind === 'user_message' && item.content === block.content)
  }
  if (block.kind === 'assistant_message') {
    return backfilled.some((item) => item.kind === 'assistant_message' && item.content === block.content)
  }
  if (block.kind === 'tool_call' || block.kind === 'tool_result' || block.kind === 'ask_user_question') {
    return backfilled.some((item) => (
      (item.kind === 'tool_call' || item.kind === 'tool_result' || item.kind === 'ask_user_question')
      && item.toolCallId === block.toolCallId
    ))
  }
  if (block.kind === 'diff') {
    return backfilled.some((item) => item.kind === 'diff' && item.path === block.path && item.diff === block.diff)
  }
  return false
}


function applyTaskFrame(frame: RunSocketFrame) {
  if (frame.type !== 'tasks.snapshot') return
  const payload = frame.payload ?? {}
  const sessionId = asString(payload.session_id)
  const tasks = Array.isArray(payload.tasks) ? payload.tasks.filter(isRecord) as TaskSummary[] : []
  if (!sessionId) return
  useTaskStore.getState().applyTaskSnapshot(sessionId, tasks)
}

function applyPromptFrame(
  frame: RunSocketFrame,
  set: (partial: ChatState | Partial<ChatState> | ((state: ChatState) => ChatState | Partial<ChatState>)) => void,
) {
  const payload = frame.payload ?? {}

  if (isPromptTerminalFrame(frame.type)) {
    const promptId = asString(payload.prompt_id)
    set((state) => resolvePromptQueues(state, promptId))
    return
  }

  const sessionId = frameSessionId(frame)
  const runId = asString(payload.run_id)
  if (!sessionId) return

  if (frame.type === 'permission.requested') {
    const request = asRecord(payload.request)
    const prompt: PermissionPrompt = {
      promptId: asString(payload.prompt_id),
      sessionId,
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
    }
    set((state) => enqueuePermissionPrompt(state, prompt))
    return
  }

  if (frame.type === 'approval.requested') {
    const prompt: ApprovalPrompt = {
      promptId: asString(payload.prompt_id),
      kind: asString(payload.kind) || 'plan',
      sessionId,
      runId,
      planText: asString(payload.plan_text) || null,
      planPath: asString(payload.plan_path) || null,
      questions: Array.isArray(payload.questions) ? payload.questions.filter(isRecord) : [],
    }
    set((state) => enqueueApprovalPrompt(state, prompt))
    return
  }
}

function isPromptTerminalFrame(type: string): boolean {
  return type === 'prompt.resolved'
    || type === 'prompt.stale'
    || type === 'prompt.expired'
    || type === 'prompt.disconnected'
    || type === 'prompt.missing'
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
