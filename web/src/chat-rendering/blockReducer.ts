import type { RunSocketFrame } from '../api/types'
import type {
  ApprovalRequestBlock,
  ChatBlock,
  ErrorDiagnosticDetail,
  PermissionPreview,
  PermissionRequestBlock,
  TokenUsage,
  ToolCallBlock,
} from './blocks'
import { answersFromMetadata, extractDiffFromMetadata, parseAskUserQuestions, recordFromUnknown, stringFromUnknown } from './content'
import type { ChatRenderState, RunStatusSnapshot } from './runState'
import { tokenUsageFromRecord, tokenUsageTotal } from './tokens'
import { frameRunId, frameSessionId } from './runState'

export function reduceRunFrame(state: ChatRenderState, frame: RunSocketFrame): ChatRenderState {
  const payload = frame.payload ?? {}
  const sessionId = frameSessionId(frame)
  const runId = frameRunId(frame)

  if (frame.type === 'run.accepted') {
    if (!sessionId || !runId) {
      return {
        ...state,
        activeRunId: runId || state.activeRunId,
      }
    }
    const blocks = state.blocksBySession[sessionId] ?? []
    const snapshot: RunStatusSnapshot = {
      runId,
      sessionId,
      state: 'starting',
      detail: 'run accepted',
    }
    return {
      ...withSessionBlocks(state, sessionId, upsertStreamingStatus(blocks, snapshot)),
      activeRunId: runId,
      statusByRun: {
        ...state.statusByRun,
        [runId]: snapshot,
      },
    }
  }

  if (isPromptTerminalFrame(frame.type)) {
    const promptId = stringFromUnknown(payload.prompt_id)
    const result = recordFromUnknown(payload.result)
    const terminalState = promptTerminalState(frame.type)
    return {
      ...state,
      blocksBySession: resolvePromptInBlocks(
        state.blocksBySession,
        promptId,
        Object.keys(result).length > 0 ? result : payload,
        terminalState,
        stringFromUnknown(payload.reason),
      ),
      pendingPermissionBlock: state.pendingPermissionBlock?.promptId === promptId ? null : state.pendingPermissionBlock,
      pendingApprovalBlock: state.pendingApprovalBlock?.promptId === promptId ? null : state.pendingApprovalBlock,
    }
  }

  if (!sessionId) return state

  const blocks = state.blocksBySession[sessionId] ?? []

  if (frame.type === 'run.started') {
    if (!runId) return state
    const snapshot: RunStatusSnapshot = {
      runId,
      sessionId,
      state: 'starting',
      detail: 'run started',
      tokens: state.statusByRun[runId]?.tokens,
    }
    return {
      ...withSessionBlocks(state, sessionId, upsertStreamingStatus(blocks, snapshot)),
      activeRunId: runId || state.activeRunId,
      statusByRun: {
        ...state.statusByRun,
        [runId]: snapshot,
      },
    }
  }

  if (frame.type === 'run.cancel.accepted') {
    if (!runId) return state
    const cancelled = payload.cancelled !== false
    const snapshot: RunStatusSnapshot = {
      runId,
      sessionId,
      state: cancelled ? 'cancelling' : 'idle',
      detail: cancelled ? 'interrupt requested' : 'no active run',
      tokens: state.statusByRun[runId]?.tokens,
    }
    return {
      ...withSessionBlocks(state, sessionId, upsertStreamingStatus(blocks, snapshot)),
      statusByRun: {
        ...state.statusByRun,
        [runId]: snapshot,
      },
    }
  }

  if (frame.type === 'assistant.delta') {
    const text = stringFromUnknown(payload.text)
    if (!text) return state
    const nextBlocks = appendAssistantDelta(blocks, sessionId, runId, text, sequence(frame))
    return withEstimatedTokenUsage(withSessionBlocks(state, sessionId, nextBlocks), sessionId, runId, nextBlocks)
  }

  if (frame.type === 'reasoning.delta') {
    const text = stringFromUnknown(payload.text)
    if (!text) return state
    const thinkingBlocks = appendThinkingDelta(blocks, sessionId, runId, text, sequence(frame))
    // Extended-thinking tokens are streaming — surface the genuine `thinking`
    // phase in the activity bar (the model-wait window shows as `requesting`).
    const previous = state.statusByRun[runId]
    const snapshot: RunStatusSnapshot = {
      runId,
      sessionId,
      state: 'thinking',
      detail: previous?.detail ?? null,
      elapsedMs: previous?.elapsedMs,
      tokens: previous?.tokens,
    }
    const nextBlocks = upsertStreamingStatus(thinkingBlocks, snapshot)
    // Seed statusByRun with the thinking snapshot first so the downstream
    // token estimator preserves the `thinking` state while topping up tokens.
    const withThinking: ChatRenderState = {
      ...state,
      statusByRun: { ...state.statusByRun, [runId]: snapshot },
    }
    return withEstimatedTokenUsage(withSessionBlocks(withThinking, sessionId, nextBlocks), sessionId, runId, nextBlocks)
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
    let nextBlocks = applyRunResultBlocks(
      blocks,
      sessionId,
      runId,
      stringFromUnknown(payload.final_text),
      metadata,
      timestampFor(frame),
    )
    if (stringFromUnknown(payload.exit_reason) === 'error') {
      nextBlocks = failRunBlocks(
        nextBlocks,
        sessionId,
        runId,
        errorDiagnosticFromRunResult(metadata, payload),
      )
    }
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

  if (frame.type === 'run.failed' || frame.type === 'error') {
    const message = stringFromUnknown(payload.message) || stringFromUnknown(payload.error) || 'Run failed.'
    const code = stringFromUnknown(payload.code) || (frame.type === 'error' ? 'run_error' : 'run_failed')
    const diagnostic = frame.type === 'error'
      ? errorDiagnosticFromServiceFrame(payload, message, code)
      : { message, code }
    return {
      ...withSessionBlocks(state, sessionId, failRunBlocks(blocks, sessionId, runId, diagnostic)),
      activeRunId: !runId || state.activeRunId === runId ? null : state.activeRunId,
    }
  }

  return state
}

export function resolvePromptInBlocks(
  blocksBySession: Record<string, ChatBlock[]>,
  promptId: string,
  result: Record<string, unknown>,
  terminalState?: 'stale' | 'expired' | 'disconnected' | 'missing',
  statusMessage?: string | null,
): Record<string, ChatBlock[]> {
  if (!promptId) return blocksBySession
  const next: Record<string, ChatBlock[]> = {}
  for (const [sessionId, blocks] of Object.entries(blocksBySession)) {
    next[sessionId] = blocks.map((block) => {
      if (block.kind === 'permission_request' && block.promptId === promptId) {
        if (terminalState) {
          return { ...block, state: terminalState, statusMessage }
        }
        const decision = recordFromUnknown(result.decision)
        const rawType = stringFromUnknown(result.type || decision.type || result.decision)
        const state = rawType.includes('deny') ? 'denied' : rawType.includes('abort') ? 'aborted' : 'allowed'
        return { ...block, state }
      }
      if (block.kind === 'approval_request' && block.promptId === promptId) {
        if (terminalState) {
          return { ...block, state: terminalState, statusMessage }
        }
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

function isPromptTerminalFrame(type: string): boolean {
  return type === 'prompt.resolved'
    || type === 'prompt.stale'
    || type === 'prompt.expired'
    || type === 'prompt.disconnected'
    || type === 'prompt.missing'
}

function promptTerminalState(type: string): 'stale' | 'expired' | 'disconnected' | 'missing' | undefined {
  if (type === 'prompt.stale') return 'stale'
  if (type === 'prompt.expired') return 'expired'
  if (type === 'prompt.disconnected') return 'disconnected'
  if (type === 'prompt.missing') return 'missing'
  return undefined
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

function withEstimatedTokenUsage(state: ChatRenderState, sessionId: string, runId: string, blocks: ChatBlock[]): ChatRenderState {
  if (!runId) return state
  const previous = state.statusByRun[runId]
  const tokens = estimatedTokenUsageForRun(blocks, runId, previous?.tokens)
  if (tokenUsageTotal(tokens) <= 0) return state
  const snapshot: RunStatusSnapshot = {
    runId,
    sessionId,
    state: previous?.state ?? 'responding',
    detail: previous?.detail ?? null,
    elapsedMs: previous?.elapsedMs,
    tokens,
  }
  return {
    ...state,
    tokenUsageByRun: {
      ...state.tokenUsageByRun,
      [runId]: tokens,
    },
    statusByRun: {
      ...state.statusByRun,
      [runId]: snapshot,
    },
  }
}

// Token estimates are recomputed on every streaming delta. Re-scanning and
// regex-counting the run's entire assistant+reasoning text each time is O(text)
// per delta -> O(text^2) over a run, which stalls the main thread mid-stream.
// Finalized blocks keep their identity (the reducer only replaces the active
// streaming block), so cache each block's estimate by reference: prior blocks
// hit the cache and only the growing active block is re-counted.
const blockTokenEstimateCache = new WeakMap<ChatBlock, number>()

function estimateBlockTokens(block: ChatBlock): number {
  const cached = blockTokenEstimateCache.get(block)
  if (cached !== undefined) return cached
  const content = block.kind === 'assistant_message' || block.kind === 'thinking' ? block.content : ''
  const tokens = estimateTextTokens(content)
  blockTokenEstimateCache.set(block, tokens)
  return tokens
}

function estimatedTokenUsageForRun(blocks: ChatBlock[], runId: string, previous?: TokenUsage): TokenUsage {
  let estimatedOutput = 0
  let estimatedReasoning = 0
  for (const block of blocks) {
    if (block.runId !== runId) continue
    if (block.kind === 'assistant_message') estimatedOutput += estimateBlockTokens(block)
    else if (block.kind === 'thinking') estimatedReasoning += estimateBlockTokens(block)
  }
  return {
    input_tokens: positiveNumber(previous?.input_tokens),
    output_tokens: positiveNumber(Math.max(previous?.output_tokens ?? 0, estimatedOutput)),
    cache_read_tokens: positiveNumber(previous?.cache_read_tokens),
    cache_write_tokens: positiveNumber(previous?.cache_write_tokens),
    reasoning_tokens: positiveNumber(Math.max(previous?.reasoning_tokens ?? 0, estimatedReasoning)),
    total_tokens: positiveNumber(previous?.total_tokens),
  }
}

function estimateTextTokens(text: string): number {
  const normalized = text.trim()
  if (!normalized) return 0
  const cjk = normalized.match(/[\u3400-\u9fff]/g)?.length ?? 0
  const nonCjk = normalized.replace(/[\u3400-\u9fff]/g, '')
  const wordLike = nonCjk.match(/[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g)?.length ?? 0
  return Math.max(1, Math.ceil(cjk + wordLike * 0.75))
}

function positiveNumber(value: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
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
  if (!finalText) {
    if (hasMetadata) {
      const resultIndex = findLastIndex(finished, (block) => block.kind === 'tool_result' && block.runId === runId)
      const block = finished[resultIndex]
      if (block?.kind === 'tool_result') {
        return replaceAt(finished, resultIndex, {
          ...block,
          metadata: mergeMetadata(block.metadata, metadata),
        })
      }
    }
    return finished
  }
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

type RunErrorDiagnostic = {
  message: string
  code?: string | null
  details?: ErrorDiagnosticDetail[]
  suggestions?: string[]
}

function failRunBlocks(blocks: ChatBlock[], sessionId: string, runId: string, diagnostic: RunErrorDiagnostic): ChatBlock[] {
  const message = diagnostic.message || 'Run failed.'
  const code = diagnostic.code || 'run_failed'
  const finished = finishRunBlocks(blocks, runId).map((block) =>
    block.kind === 'assistant_message' && block.runId === runId
      ? { ...block, isStreaming: false, isError: true }
      : block,
  )
  const hasError = finished.some((block) => block.kind === 'error' && block.runId === runId)
  if (hasError) {
    return finished.map((block) => {
      if (block.kind !== 'error' || block.runId !== runId) return block
      return {
        ...block,
        message: preferredErrorMessage(block.message, message),
        code: block.code || code,
        details: mergeErrorDetails(block.details, diagnostic.details),
        suggestions: mergeStrings(block.suggestions, diagnostic.suggestions),
      }
    })
  }
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
      code,
      ...(diagnostic.details && diagnostic.details.length > 0 ? { details: diagnostic.details } : {}),
      ...(diagnostic.suggestions && diagnostic.suggestions.length > 0 ? { suggestions: diagnostic.suggestions } : {}),
    },
  ]
}

function preferredErrorMessage(current: string, incoming: string): string {
  if (!incoming) return current
  if (!current) return incoming
  if (incoming === current) return current
  if (incoming.length > current.length) return incoming
  return current
}

function errorDiagnosticFromRunResult(metadata: Record<string, unknown>, payload: Record<string, unknown>): RunErrorDiagnostic {
  const error = recordFromUnknown(metadata.error)
  const message = stringFromUnknown(error.message) || stringFromUnknown(payload.error) || 'Run failed.'
  const details: ErrorDiagnosticDetail[] = []
  const status = numberOrUndefined(error.status_code)
  if (status !== undefined) details.push({ label: 'HTTP status', value: String(status) })
  const errorMetadata = recordFromUnknown(error.metadata)
  const url = stringFromUnknown(errorMetadata.url)
  if (url) details.push({ label: 'Endpoint', value: url })
  const method = stringFromUnknown(errorMetadata.method)
  if (method) details.push({ label: 'Method', value: method })
  const phase = stringFromUnknown(errorMetadata.phase)
  if (phase) details.push({ label: 'Phase', value: phase })
  if (error.is_network_error === true) details.push({ label: 'Network', value: 'transport-level failure' })
  const retryAfter = numberOrUndefined(error.retry_after_seconds)
  if (retryAfter !== undefined) details.push({ label: 'Retry after', value: formatSeconds(retryAfter) })
  const body = stringFromUnknown(error.body_summary)
  if (body) details.push({ label: 'Response body', value: body })
  return {
    message,
    code: stringFromUnknown(error.type) || 'run_error',
    details,
    suggestions: errorSuggestions({ status, url, body, message, network: error.is_network_error === true }),
  }
}

function errorDiagnosticFromServiceFrame(
  payload: Record<string, unknown>,
  message: string,
  code: string,
): RunErrorDiagnostic {
  const detailsRecord = recordFromUnknown(payload.details)
  const details = Object.entries(detailsRecord)
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([key, value]) => ({
      label: detailLabel(key),
      value: detailValue(value),
    }))
    .filter((detail) => detail.value.length > 0)
  return {
    message,
    code,
    details,
    suggestions: serviceErrorSuggestions(code, message),
  }
}

function errorSuggestions(input: { status?: number; url: string; body: string; message: string; network: boolean }): string[] {
  const haystack = [input.body, input.message].join('\n').toLowerCase()
  const suggestions: string[] = []
  if (input.status === 404) {
    suggestions.push('Check the provider base URL and API mode. OpenAI-compatible chat calls usually need a /v1-compatible endpoint that serves /chat/completions.')
    suggestions.push('Verify the selected model exists on that endpoint; some gateways return 404 for unknown models.')
  } else if (input.status === 401 || input.status === 403) {
    suggestions.push('Check the active provider credential and whether this key can use the selected model.')
  } else if (input.status === 429) {
    suggestions.push('The provider rate limit was hit. Wait or switch to a provider/model with available quota.')
  } else if (input.status === 413 || haystack.includes('context length') || haystack.includes('maximum context')) {
    suggestions.push('The request is too large for the model context window. Compress context, clear old transcript, or select a larger-context model.')
  } else if (input.network) {
    suggestions.push('The backend could not complete the network request. Check provider host reachability, proxy settings, and timeout configuration.')
  }
  if (input.url && !input.url.includes('/chat/completions') && input.url.includes('/v1')) {
    suggestions.push('This request did not target /chat/completions; confirm the provider API mode matches the configured endpoint.')
  }
  return mergeStrings(undefined, suggestions) ?? []
}

function serviceErrorSuggestions(code: string, message: string): string[] {
  const normalized = (code + '\n' + message).toLowerCase()
  if (normalized.includes('run_already_active') || normalized.includes('already active')) {
    return ['Wait for the active run to finish, or cancel it before starting a new run in this session.']
  }
  if (normalized.includes('invalid_run_start')) {
    return ['Send a non-empty prompt or attach at least one file before starting a run.']
  }
  return []
}

function detailLabel(key: string): string {
  return key
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function detailValue(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function mergeErrorDetails(
  left: ErrorDiagnosticDetail[] | undefined,
  right: ErrorDiagnosticDetail[] | undefined,
): ErrorDiagnosticDetail[] | undefined {
  const seen = new Set<string>()
  const merged: ErrorDiagnosticDetail[] = []
  for (const detail of [...(left ?? []), ...(right ?? [])]) {
    if (!detail.label || !detail.value) continue
    const key = detail.label + '\n' + detail.value
    if (seen.has(key)) continue
    seen.add(key)
    merged.push(detail)
  }
  return merged.length > 0 ? merged : undefined
}

function mergeStrings(left: string[] | undefined, right: string[] | undefined): string[] | undefined {
  const seen = new Set<string>()
  const merged: string[] = []
  for (const value of [...(left ?? []), ...(right ?? [])]) {
    const normalized = value.trim()
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    merged.push(normalized)
  }
  return merged.length > 0 ? merged : undefined
}

function formatSeconds(value: number): string {
  if (!Number.isFinite(value)) return String(value)
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 }) + 's'
}

function statusSnapshotFromFrame(
  previous: RunStatusSnapshot | undefined,
  frame: RunSocketFrame,
  sessionId: string,
  runId: string,
): RunStatusSnapshot {
  const payload = frame.payload ?? {}
  if (frame.type === 'run.status') {
    // The backend reports `thinking` while it is about to call the model
    // (before_llm / after_tool) — that is the request-in-flight wait, shown
    // as `requesting`. Genuine extended thinking arrives via reasoning.delta.
    const kind = stringFromUnknown(payload.kind)
    const state = kind === 'thinking' ? 'requesting' : kind || previous?.state || 'responding'
    return {
      runId,
      sessionId,
      state,
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
  // silent.progress carries function-call argument fragments — the model is
  // streaming the tool-call input, distinct from executing the tool.
  return {
    runId,
    sessionId,
    state: 'tool_input',
    detail: previous?.detail || null,
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
