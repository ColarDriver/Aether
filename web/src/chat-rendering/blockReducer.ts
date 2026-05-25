import type { RunSocketFrame } from '../api/types'
import type {
  ApprovalRequestBlock,
  ChatBlock,
  PermissionPreview,
  PermissionRequestBlock,
  TokenUsage,
  ToolCallBlock,
} from './blocks'
import { answersFromMetadata, extractDiffFromMetadata, parseAskUserQuestions, recordFromUnknown, stringFromUnknown } from './content'
import type { ChatRenderState, RunStatusSnapshot } from './runState'
import { tokenUsageFromRecord } from './tokens'
import { frameRunId, frameSessionId } from './runState'

export function reduceRunFrame(state: ChatRenderState, frame: RunSocketFrame): ChatRenderState {
  const payload = frame.payload ?? {}
  const sessionId = frameSessionId(frame)
  const runId = frameRunId(frame)

  if (frame.type === 'run.accepted') {
    return {
      ...state,
      activeRunId: runId || state.activeRunId,
    }
  }

  if (frame.type === 'prompt.resolved') {
    const promptId = stringFromUnknown(payload.prompt_id)
    const result = recordFromUnknown(payload.result)
    return {
      ...state,
      blocksBySession: resolvePromptInBlocks(
        state.blocksBySession,
        promptId,
        Object.keys(result).length > 0 ? result : payload,
      ),
      pendingPermissionBlock: state.pendingPermissionBlock?.promptId === promptId ? null : state.pendingPermissionBlock,
      pendingApprovalBlock: state.pendingApprovalBlock?.promptId === promptId ? null : state.pendingApprovalBlock,
    }
  }

  if (!sessionId) return state

  const blocks = state.blocksBySession[sessionId] ?? []

  if (frame.type === 'assistant.delta') {
    const text = stringFromUnknown(payload.text)
    if (!text) return state
    return withSessionBlocks(state, sessionId, appendAssistantDelta(blocks, sessionId, runId, text, sequence(frame)))
  }

  if (frame.type === 'reasoning.delta') {
    const text = stringFromUnknown(payload.text)
    if (!text) return state
    return withSessionBlocks(state, sessionId, appendThinkingDelta(blocks, sessionId, runId, text, sequence(frame)))
  }

  if (frame.type === 'run.status' || frame.type === 'loop.state' || frame.type === 'silent.progress') {
    const snapshot = statusSnapshotFromFrame(state.statusByRun[runId], frame, sessionId, runId)
    return {
      ...withSessionBlocks(state, sessionId, upsertStreamingStatus(blocks, snapshot)),
      statusByRun: {
        ...state.statusByRun,
        [runId]: snapshot,
      },
    }
  }

  if (frame.type === 'token.usage' && runId) {
    const usage = tokenUsageFromPayload(payload)
    const previous = state.statusByRun[runId]
    const snapshot: RunStatusSnapshot = {
      runId,
      sessionId,
      state: previous?.state ?? 'responding',
      detail: previous?.detail ?? null,
      elapsedMs: previous?.elapsedMs,
      tokens: usage,
    }
    return {
      ...withSessionBlocks(state, sessionId, upsertStreamingStatus(blocks, snapshot)),
      tokenUsageByRun: {
        ...state.tokenUsageByRun,
        [runId]: usage,
      },
      statusByRun: {
        ...state.statusByRun,
        [runId]: snapshot,
      },
    }
  }

  if (frame.type === 'tool.started') {
    const toolCallId = stringFromUnknown(payload.tool_call_id)
    if (!toolCallId) return state
    return withSessionBlocks(
      state,
      sessionId,
      upsertToolStarted(blocks, {
        sessionId,
        runId,
        toolCallId,
        toolName: stringFromUnknown(payload.tool_name) || 'tool',
        args: recordFromUnknown(payload.arguments),
        iteration: numberOrUndefined(payload.iteration),
        timestamp: timestampFor(frame),
      }),
    )
  }

  if (frame.type === 'tool.finished') {
    const toolCallId = stringFromUnknown(payload.tool_call_id)
    if (!toolCallId) return state
    return withSessionBlocks(
      state,
      sessionId,
      upsertToolFinished(blocks, {
        sessionId,
        runId,
        toolCallId,
        toolName: stringFromUnknown(payload.tool_name) || 'tool',
        content: stringFromUnknown(payload.content),
        isError: Boolean(payload.is_error),
        metadata: recordFromUnknown(payload.metadata),
        iteration: numberOrUndefined(payload.iteration),
        timestamp: timestampFor(frame),
      }),
    )
  }

  if (frame.type === 'permission.requested') {
    const request = recordFromUnknown(payload.request)
    const block: PermissionRequestBlock = {
      id: 'permission-' + stringFromUnknown(payload.prompt_id),
      sessionId,
      runId,
      timestamp: timestampFor(frame),
      source: 'live',
      kind: 'permission_request',
      promptId: stringFromUnknown(payload.prompt_id),
      toolCallId: stringOrNull(request.tool_call_id),
      toolName: stringFromUnknown(request.tool_name) || 'tool',
      arguments: recordFromUnknown(request.arguments),
      category: stringOrNull(request.category),
      risk: stringOrNull(request.risk),
      reason: stringOrNull(request.reason),
      preview: previewFromUnknown(request.preview),
      allowSession: Boolean(request.allow_session),
      state: 'pending',
    }
    return {
      ...withSessionBlocks(state, sessionId, upsertBlockById(blocks, block)),
      activeRunId: runId || state.activeRunId,
      pendingPermissionBlock: block,
    }
  }

  if (frame.type === 'approval.requested') {
    const block: ApprovalRequestBlock = {
      id: 'approval-' + stringFromUnknown(payload.prompt_id),
      sessionId,
      runId,
      timestamp: timestampFor(frame),
      source: 'live',
      kind: 'approval_request',
      promptId: stringFromUnknown(payload.prompt_id),
      approvalKind: stringFromUnknown(payload.kind) || 'plan',
      planText: stringOrNull(payload.plan_text),
      planPath: stringOrNull(payload.plan_path),
      questions: parseAskUserQuestions({ questions: Array.isArray(payload.questions) ? payload.questions : [] }),
      state: 'pending',
    }
    return {
      ...withSessionBlocks(state, sessionId, upsertBlockById(blocks, block)),
      activeRunId: runId || state.activeRunId,
      pendingApprovalBlock: block,
    }
  }

  if (frame.type === 'run.result') {
    const metadata = recordFromUnknown(payload.metadata)
    const usage = tokenUsageFromRecord(recordFromUnknown(payload.usage))
    const hasUsage = Object.values(usage).some((value) => typeof value === 'number')
    const nextBlocks = applyRunResultBlocks(
      blocks,
      sessionId,
      runId,
      stringFromUnknown(payload.final_text),
      metadata,
      timestampFor(frame),
    )
    const previous = state.statusByRun[runId]
    return {
      ...withSessionBlocks(state, sessionId, nextBlocks),
      activeRunId: state.activeRunId === runId ? null : state.activeRunId,
      ...(hasUsage ? {
        tokenUsageByRun: {
          ...state.tokenUsageByRun,
          [runId]: usage,
        },
        statusByRun: {
          ...state.statusByRun,
          [runId]: {
            runId,
            sessionId,
            state: 'idle',
            detail: previous?.detail ?? null,
            elapsedMs: previous?.elapsedMs,
            tokens: usage,
          },
        },
      } : {}),
    }
  }

  if (frame.type === 'run.finished' || frame.type === 'run.cancelled') {
    return {
      ...withSessionBlocks(state, sessionId, finishRunBlocks(blocks, runId)),
      activeRunId: state.activeRunId === runId ? null : state.activeRunId,
    }
  }

  if (frame.type === 'run.failed') {
    return {
      ...withSessionBlocks(state, sessionId, failRunBlocks(blocks, sessionId, runId, stringFromUnknown(payload.message))),
      activeRunId: state.activeRunId === runId ? null : state.activeRunId,
    }
  }

  return state
}

export function resolvePromptInBlocks(
  blocksBySession: Record<string, ChatBlock[]>,
  promptId: string,
  result: Record<string, unknown>,
): Record<string, ChatBlock[]> {
  if (!promptId) return blocksBySession
  const next: Record<string, ChatBlock[]> = {}
  for (const [sessionId, blocks] of Object.entries(blocksBySession)) {
    next[sessionId] = blocks.map((block) => {
      if (block.kind === 'permission_request' && block.promptId === promptId) {
        const decision = recordFromUnknown(result.decision)
        const rawType = stringFromUnknown(result.type || decision.type || result.decision)
        const state = rawType.includes('deny') ? 'denied' : rawType.includes('abort') ? 'aborted' : 'allowed'
        return { ...block, state }
      }
      if (block.kind === 'approval_request' && block.promptId === promptId) {
        const confirmed = result.confirmed === true
        const answers = recordFromUnknown(result.answers)
        return {
          ...block,
          state: Object.keys(answers).length > 0 ? 'answered' : confirmed ? 'approved' : 'rejected',
        }
      }
      return block
    })
  }
  return next
}

function withSessionBlocks(state: ChatRenderState, sessionId: string, blocks: ChatBlock[]): ChatRenderState {
  return {
    ...state,
    blocksBySession: {
      ...state.blocksBySession,
      [sessionId]: blocks,
    },
  }
}

function appendAssistantDelta(blocks: ChatBlock[], sessionId: string, runId: string, text: string, timestamp: number): ChatBlock[] {
  const index = findLastIndex(blocks, (block) =>
    block.kind === 'assistant_message' && block.runId === runId && Boolean(block.isStreaming),
  )
  if (index >= 0) {
    const block = blocks[index]
    if (block?.kind === 'assistant_message') {
      return replaceAt(blocks, index, { ...block, content: block.content + text })
    }
  }
  return [
    ...blocks,
    {
      id: 'assistant-' + (runId || timestamp),
      sessionId,
      runId,
      timestamp,
      source: 'live',
      kind: 'assistant_message',
      content: text,
      isStreaming: true,
    },
  ]
}

function appendThinkingDelta(blocks: ChatBlock[], sessionId: string, runId: string, text: string, timestamp: number): ChatBlock[] {
  const id = 'thinking-' + (runId || timestamp)
  const index = blocks.findIndex((block) => block.id === id)
  if (index >= 0) {
    const block = blocks[index]
    if (block?.kind === 'thinking') {
      return replaceAt(blocks, index, {
        ...block,
        content: block.content + text,
        isActive: true,
        sequence: timestamp,
      })
    }
  }
  return [
    ...blocks,
    {
      id,
      sessionId,
      runId,
      timestamp,
      source: 'live',
      kind: 'thinking',
      content: text,
      isActive: true,
      sequence: timestamp,
    },
  ]
}

function upsertToolStarted(
  blocks: ChatBlock[],
  input: {
    sessionId: string
    runId: string
    toolCallId: string
    toolName: string
    args: Record<string, unknown>
    iteration?: number
    timestamp: number
  },
): ChatBlock[] {
  const existingIndex = findToolIndex(blocks, input.toolCallId)
  const block: ChatBlock = isAskUserQuestionTool(input.toolName)
    ? {
        id: input.runId + '-' + input.toolCallId,
        sessionId: input.sessionId,
        runId: input.runId,
        timestamp: input.timestamp,
        source: 'live',
        kind: 'ask_user_question',
        toolCallId: input.toolCallId,
        questions: parseAskUserQuestions(input.args),
        state: 'pending',
      }
    : {
        id: input.runId + '-' + input.toolCallId,
        sessionId: input.sessionId,
        runId: input.runId,
        timestamp: input.timestamp,
        source: 'live',
        kind: 'tool_call',
        toolCallId: input.toolCallId,
        toolName: input.toolName,
        arguments: input.args,
        status: 'running',
        iteration: input.iteration,
      }
  return existingIndex >= 0 ? replaceAt(blocks, existingIndex, block) : [...blocks, block]
}

function upsertToolFinished(
  blocks: ChatBlock[],
  input: {
    sessionId: string
    runId: string
    toolCallId: string
    toolName: string
    content: string
    isError: boolean
    metadata: Record<string, unknown>
    iteration?: number
    timestamp: number
  },
): ChatBlock[] {
  let next = blocks.map((block) => {
    if (block.kind === 'tool_call' && block.toolCallId === input.toolCallId) {
      return {
        ...block,
        status: input.isError ? 'failed' : 'finished',
      } satisfies ToolCallBlock
    }
    if (block.kind === 'ask_user_question' && block.toolCallId === input.toolCallId) {
      const answers = answersFromMetadata(input.metadata)
      return {
        ...block,
        state: input.isError ? 'cancelled' : 'answered',
        ...(Object.keys(answers).length > 0 ? { answers } : {}),
      } satisfies ChatBlock
    }
    return block
  })

  const result: ChatBlock = {
    id: input.runId + '-result-' + input.toolCallId,
    sessionId: input.sessionId,
    runId: input.runId,
    timestamp: input.timestamp,
    source: 'live',
    kind: 'tool_result',
    toolCallId: input.toolCallId,
    toolName: input.toolName,
    content: input.content,
    isError: input.isError,
    metadata: input.metadata,
  }
  next = upsertBlockById(next, result)

  const diff = extractDiffFromMetadata(input.metadata)
  if (diff) {
    next = upsertBlockById(next, {
      id: input.runId + '-diff-' + input.toolCallId,
      sessionId: input.sessionId,
      runId: input.runId,
      timestamp: input.timestamp,
      source: 'live',
      kind: 'diff',
      origin: 'tool_result',
      ...diff,
    })
  }
  return next
}

function upsertStreamingStatus(blocks: ChatBlock[], snapshot: RunStatusSnapshot): ChatBlock[] {
  const id = 'status-' + snapshot.runId
  const block: ChatBlock = {
    id,
    sessionId: snapshot.sessionId,
    runId: snapshot.runId,
    timestamp: Date.now(),
    source: 'live',
    kind: 'streaming_status',
    state: snapshot.state,
    detail: snapshot.detail,
    elapsedMs: snapshot.elapsedMs,
    tokens: snapshot.tokens,
  }
  return upsertBlockById(blocks, block)
}

function applyRunResultBlocks(
  blocks: ChatBlock[],
  sessionId: string,
  runId: string,
  finalText: string,
  metadata: Record<string, unknown>,
  timestamp: number,
): ChatBlock[] {
  const hasMetadata = Object.keys(metadata).length > 0
  const finished = finishRunBlocks(blocks, runId)
  const assistantIndex = findLastIndex(finished, (block) => block.kind === 'assistant_message' && block.runId === runId)
  if (assistantIndex >= 0) {
    const block = finished[assistantIndex]
    if (block?.kind !== 'assistant_message') return finished
    const content = finalText && (!block.content || finalText.startsWith(block.content)) ? finalText : block.content
    return replaceAt(finished, assistantIndex, {
      ...block,
      content,
      isStreaming: false,
      ...(hasMetadata ? { metadata: mergeMetadata(block.metadata, metadata) } : {}),
    })
  }
  if (!finalText) return finished
  return [
    ...finished,
    {
      id: 'assistant-' + (runId || timestamp),
      sessionId,
      runId,
      timestamp,
      source: 'live',
      kind: 'assistant_message',
      content: finalText,
      isStreaming: false,
      ...(hasMetadata ? { metadata } : {}),
    },
  ]
}

function mergeMetadata(current: Record<string, unknown> | undefined, incoming: Record<string, unknown>): Record<string, unknown> {
  return { ...(current ?? {}), ...incoming }
}

function finishRunBlocks(blocks: ChatBlock[], runId: string): ChatBlock[] {
  return blocks
    .map((block) => {
      if (block.runId !== runId) return block
      if (block.kind === 'assistant_message') return { ...block, isStreaming: false }
      if (block.kind === 'thinking') return { ...block, isActive: false }
      return block
    })
    .filter((block) => !(block.runId === runId && block.kind === 'streaming_status'))
}

function failRunBlocks(blocks: ChatBlock[], sessionId: string, runId: string, message: string): ChatBlock[] {
  const finished = finishRunBlocks(blocks, runId).map((block) =>
    block.kind === 'assistant_message' && block.runId === runId
      ? { ...block, isStreaming: false, isError: true }
      : block,
  )
  const hasError = finished.some((block) => block.kind === 'error' && block.runId === runId)
  if (hasError) return finished
  return [
    ...finished,
    {
      id: 'error-' + (runId || Date.now()),
      sessionId,
      runId,
      timestamp: Date.now(),
      source: 'live',
      kind: 'error',
      message: message || 'Run failed.',
      code: 'run_failed',
    },
  ]
}

function statusSnapshotFromFrame(
  previous: RunStatusSnapshot | undefined,
  frame: RunSocketFrame,
  sessionId: string,
  runId: string,
): RunStatusSnapshot {
  const payload = frame.payload ?? {}
  if (frame.type === 'run.status') {
    return {
      runId,
      sessionId,
      state: stringFromUnknown(payload.kind) || previous?.state || 'responding',
      detail: stringOrNull(payload.detail),
      elapsedMs: previous?.elapsedMs,
      tokens: previous?.tokens,
    }
  }
  if (frame.type === 'loop.state') {
    return {
      runId,
      sessionId,
      state: previous?.state || 'responding',
      detail: stringFromUnknown(payload.state) || previous?.detail || null,
      elapsedMs: previous?.elapsedMs,
      tokens: previous?.tokens,
    }
  }
  return {
    runId,
    sessionId,
    state: previous?.state || 'thinking',
    detail: previous?.detail || 'working',
    elapsedMs: previous?.elapsedMs,
    tokens: previous?.tokens,
  }
}

function tokenUsageFromPayload(payload: Record<string, unknown>): TokenUsage {
  return tokenUsageFromRecord(payload)
}

function upsertBlockById(blocks: ChatBlock[], block: ChatBlock): ChatBlock[] {
  const index = blocks.findIndex((item) => item.id === block.id)
  return index >= 0 ? replaceAt(blocks, index, block) : [...blocks, block]
}

function findToolIndex(blocks: ChatBlock[], toolCallId: string): number {
  return blocks.findIndex((block) =>
    (block.kind === 'tool_call' || block.kind === 'ask_user_question') && block.toolCallId === toolCallId,
  )
}

function previewFromUnknown(value: unknown): PermissionPreview | null {
  const record = recordFromUnknown(value)
  if (Object.keys(record).length === 0) return null
  return {
    title: stringOrUndefined(record.title),
    subtitle: stringOrNull(record.subtitle),
    body: stringOrNull(record.body),
    diff: stringOrNull(record.diff),
    path: stringOrNull(record.path),
    command: stringOrNull(record.command),
    metadata: recordFromUnknown(record.metadata),
  }
}

function isAskUserQuestionTool(toolName: string): boolean {
  return toolName.toLowerCase() === 'ask_user_question' || toolName === 'AskUserQuestion'
}

function timestampFor(frame: RunSocketFrame): number {
  return typeof frame.transport_sequence === 'number' ? frame.transport_sequence : Date.now()
}

function sequence(frame: RunSocketFrame): number {
  return typeof frame.payload?.sequence === 'number' ? frame.payload.sequence : timestampFor(frame)
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined
}

function stringOrNull(value: unknown): string | null {
  return typeof value === 'string' && value ? value : null
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === 'string' && value ? value : undefined
}

function replaceAt<T>(items: T[], index: number, value: T): T[] {
  return [...items.slice(0, index), value, ...items.slice(index + 1)]
}

function findLastIndex<T>(items: T[], predicate: (item: T) => boolean): number {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (predicate(items[index]!)) return index
  }
  return -1
}
