import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { GatewayEvent } from '../gatewayTypes.js'
import { theme } from '../lib/theme.js'
import { shouldClearTodos, todosFromArgs } from '../lib/todos.js'
import { activityActions, type TurnSummary } from '../store/activityStore.js'
import { chatActions, verboseMode } from '../store/chatStore.js'
import { reasoningActions } from '../store/reasoningStore.js'
import { sessionActions } from '../store/sessionStore.js'
import { toolGroupActions } from '../store/toolGroupStore.js'

// The gateway can fire two events for the same root cause when an LLM call
// fails: one from `_GatewayEventMiddleware.on_error` (prefixed with the loop
// state — `LLM_CALL: provider HTTP 404…`) and one from the outer
// `except Exception` (the bare message). Show the user a single clean line.
const ERROR_DEDUPE_WINDOW_MS = 4000
const LOOP_STATE_PREFIX_RE = /^[A-Z][A-Z0-9_]*:\s+/

interface RecentError {
  text: string
  ts: number
}
let lastError: RecentError | null = null

function pushDedupedError(message: string): void {
  const normalised = stripLoopStatePrefix(message).trim()
  if (!normalised) {
    return
  }
  const now = Date.now()
  if (lastError && now - lastError.ts < ERROR_DEDUPE_WINDOW_MS) {
    if (
      lastError.text === normalised ||
      lastError.text.endsWith(normalised) ||
      normalised.endsWith(lastError.text)
    ) {
      // Keep the most-informative version (the longer of the two) by
      // overwriting the previous note rather than pushing a duplicate.
      lastError.ts = now
      return
    }
  }
  lastError = { text: normalised, ts: now }
  chatActions.pushNote(normalised, 'error')

  const hint = buildErrorHint(normalised)
  if (hint) {
    chatActions.pushNote(hint, 'warn')
  }
}

/**
 * Surface a small actionable hint when an error message looks like a
 * misconfiguration the user can fix without reading the stack trace —
 * provider 4xx, missing API key, unknown model, etc. Returns null when no
 * hint is appropriate so the caller can stay quiet.
 */
function buildErrorHint(message: string): string | null {
  const lower = message.toLowerCase()
  if (lower.includes('http 401') || lower.includes('http 403') || lower.includes('unauthor')) {
    return 'hint: check API key (OPENAI_API_KEY / ANTHROPIC_API_KEY) or provider config'
  }
  if (lower.includes('http 404') || lower.includes('not found') || lower.includes('unknown model')) {
    // 404 on chat completions almost always means one of:
    // (a) model id is wrong  →  user should run /model
    // (b) base_url is missing the API root prefix (e.g. /v1)  →  user
    //     should re-export OPENAI_BASE_URL with /v1 and restart.
    // We can't tell which without inspecting the request; mention both
    // so the user has a path forward in either case.
    return (
      'hint: 404 from provider — try /model to confirm the model name, ' +
      'or check OPENAI_BASE_URL has the right /v1 (or equivalent) suffix'
    )
  }
  if (lower.includes('http 429') || lower.includes('rate limit')) {
    return 'hint: rate-limited — wait a moment or switch providers with /model'
  }
  if (lower.includes('connection') || lower.includes('timeout') || lower.includes('econnrefused')) {
    return 'hint: provider unreachable — check OPENAI_BASE_URL / network'
  }
  return null
}

function stripLoopStatePrefix(message: string): string {
  return message.replace(LOOP_STATE_PREFIX_RE, '')
}

// Gate against duplicate terminal events for the same agent run. The gateway
// may emit multiple Error notifications for one underlying failure (once from
// the middleware `_GatewayEventMiddleware.on_error` and once from the
// top-level exception handler in `agent_methods.agent_run`). Without this
// guard we double-call `endTurn`, double-increment `turnErrors`, and push two
// per-turn footers ("● failed · 1 error · 33.5s" then "● failed · 2 errors").
let lastFinishedRunId: string | null = null

export function resetGatewayEventDedupeForTests(): void {
  lastError = null
  lastFinishedRunId = null
}

export function useGatewayEvents(client: GatewayClient): void {
  useEffect(() => {
    const handler = (event: GatewayEvent) => {
      applyGatewayEvent(event)
    }
    client.on('event', handler)
    return () => {
      client.off('event', handler)
    }
  }, [client])
}

export function applyGatewayEvent(event: GatewayEvent): void {
  switch (event.type) {
    case 'gateway.ready':
      sessionActions.setGatewayReady(event)
      break
    case 'gateway.error':
      pushDedupedError(event.message)
      break
    case 'gateway.stderr':
      if (verboseMode.get()) {
        chatActions.pushNote(event.line, 'warn')
      }
      break
    case 'gateway.protocol_error':
      chatActions.pushNote(`protocol error: ${event.reason}`, 'error')
      break
    case 'gateway.server_request':
      // Routed by useReverseRpc into overlayStore; intentionally not handled
      // here so the chat transcript does not get spammed with internal RPCs.
      break
    case 'text.delta':
      chatActions.appendAssistant(event.run_id, event.text)
      activityActions.addResponseChars(event.text.length)
      activityActions.setStatus('responding')
      break
    case 'tool.call': {
      if (event.tool_name === 'todo_write') {
        const todos = todosFromArgs(event.arguments)
        activityActions.setTodos(shouldClearTodos(todos) ? [] : todos)
      }
      const dispatch = toolGroupActions.startCall({
        toolCallId: event.tool_call_id,
        toolName: event.tool_name,
        args: event.arguments,
        iteration: event.iteration
      })
      chatActions.pushToolCall({
        id: event.tool_call_id,
        toolName: event.tool_name,
        args: event.arguments,
        iteration: event.iteration,
        coalesce: dispatch.isExplore
      })
      activityActions.setStatus('tool_use', event.tool_name)
      break
    }
    case 'tool.result':
      toolGroupActions.finishCall({
        toolCallId: event.tool_call_id,
        isError: Boolean(event.is_error)
      })
      chatActions.pushToolResult({
        toolCallId: event.tool_call_id,
        toolName: event.tool_name,
        text: event.content,
        isError: Boolean(event.is_error),
        ...(event.metadata ? { metadata: event.metadata } : {})
      })
      activityActions.bumpToolCounter(Boolean(event.is_error))
      activityActions.setStatus('thinking')
      break
    case 'status':
      sessionActions.setStatus(event.kind, event.detail ?? null)
      activityActions.setStatus(event.kind, event.detail ?? null)
      break
    case 'usage':
      sessionActions.addUsage({
        inputTokens: event.input_tokens,
        outputTokens: event.output_tokens,
        cacheReadTokens: event.cache_read_tokens ?? 0,
        cacheWriteTokens: event.cache_write_tokens ?? 0
      })
      activityActions.addUsage({
        input: event.input_tokens,
        output: event.output_tokens,
        cacheRead: event.cache_read_tokens ?? 0,
        cacheWrite: event.cache_write_tokens ?? 0
      })
      break
    case 'iteration.start':
      // Each iteration boundary flushes the prior explore burst so its
      // past-tense Explored summary lands in scrollback before any new
      // tool call header.
      flushAndPushGroup()
      activityActions.setIteration(event.iteration)
      activityActions.setStatus('thinking')
      // A new iteration means a new turn is active — clear the
      // last-finished marker so the next terminal event for this run
      // (or the next run) is not silently swallowed.
      if (event.run_id !== lastFinishedRunId) {
        lastFinishedRunId = null
      }
      break
    case 'iteration.end':
      flushAndPushGroup()
      activityActions.setIteration(event.iteration)
      break
    case 'loop.state':
      activityActions.setLoopState(event.state)
      break
    case 'reasoning.delta':
      reasoningActions.appendDelta(event.text)
      break
    case 'done': {
      if (event.run_id === lastFinishedRunId) {
        break
      }
      lastFinishedRunId = event.run_id
      flushAndPushGroup()
      activityActions.flushUsage()
      chatActions.finishAssistant(event.run_id, event.final_text)
      sessionActions.setStatus('idle')
      const summary = activityActions.endTurn('done')
      pushTurnFooter('done', summary, event.exit_reason ?? null)
      reasoningActions.clear()
      lastError = null
      break
    }
    case 'cancelled': {
      if (event.run_id === lastFinishedRunId) {
        break
      }
      lastFinishedRunId = event.run_id
      flushAndPushGroup()
      activityActions.flushUsage()
      chatActions.finishAssistant(event.run_id, event.partial_text)
      sessionActions.setStatus('idle')
      const summary = activityActions.endTurn('cancelled')
      pushTurnFooter('cancelled', summary, sanitiseTurnExitReason(event.reason ?? null))
      reasoningActions.clear()
      lastError = null
      break
    }
    case 'error': {
      // The error message itself is always deduplicated (pushDedupedError) so
      // the user sees one line per failure. But the turn-finalisation work
      // (endTurn, footer, counter bumps) must only run for the FIRST error
      // event of a given run_id — otherwise the second emit produces a
      // bogus "● failed · 2 errors" footer with durationMs=0.
      if (event.run_id === lastFinishedRunId) {
        pushDedupedError(event.message)
        break
      }
      lastFinishedRunId = event.run_id
      // Drop the active explore tracker without flushing — its past-tense
      // summary would be misleading after a fatal mid-iteration.
      toolGroupActions.discardActive()
      activityActions.flushUsage()
      chatActions.finishAssistant(event.run_id)
      activityActions.bumpErrorCounter()
      pushDedupedError(event.message)
      sessionActions.setStatus('idle')
      const summary = activityActions.endTurn('error')
      pushTurnFooter('error', summary, null)
      reasoningActions.clear()
      break
    }
  }
}

function flushAndPushGroup(): void {
  const flushed = toolGroupActions.flushActive()
  if (flushed) {
    chatActions.pushToolGroup(flushed)
  }
}

/**
 * Append the small "● done · 2 iter · 1 tool · 1.2s" line that mirrors the
 * Python TUI's per-turn footer. Mode determines the verb + colour level.
 *
 * - done      → info, "done"
 * - cancelled → warn, "cancelled"
 * - error     → error, "failed"
 *
 * In verbose mode (or on any non-success terminal state) we also append the
 * engine's `exit_reason` — matches Python `ui.py:2219` which keeps the
 * footer terse for normal users but exposes the engine jargon when the
 * operator has opted in.
 */
function pushTurnFooter(
  mode: 'done' | 'cancelled' | 'error',
  summary: TurnSummary,
  exitReason: string | null
): void {
  const verb = mode === 'done' ? 'done' : mode === 'cancelled' ? 'cancelled' : 'failed'
  const lead =
    mode === 'done'
      ? theme.icon('success') || '✓'
      : mode === 'cancelled'
        ? theme.icon('interrupt') || '⏹'
        : theme.icon('error') || '✗'
  const parts: string[] = [`${lead} ${verb}`]
  if (summary.iterations > 0) {
    const iterIcon = theme.icon('iter') || '↻'
    parts.push(`${iterIcon} ${summary.iterations} iter${summary.iterations === 1 ? '' : 's'}`)
  }
  if (summary.tools > 0) {
    const toolIcon = theme.icon('tool') || '⚙'
    parts.push(`${toolIcon} ${summary.tools} tool${summary.tools === 1 ? '' : 's'}`)
  }
  if (summary.errors > 0) {
    parts.push(`${summary.errors} error${summary.errors === 1 ? '' : 's'}`)
  }
  if (summary.durationMs > 0) {
    parts.push(formatDurationShort(summary.durationMs))
  }
  if (exitReason && (verboseMode.get() || mode !== 'done')) {
    parts.push(exitReason.toLowerCase().replace(/_/g, ' '))
  }
  const level: 'info' | 'warn' | 'error' =
    mode === 'error' ? 'error' : mode === 'cancelled' ? 'warn' : 'info'
  chatActions.pushNote(parts.join(' · '), level)
}

function sanitiseTurnExitReason(exitReason: string | null): string | null {
  if (!exitReason) {
    return null
  }
  const normalised = exitReason.trim().toLowerCase().replace(/[_\s]+/g, '-')
  if (normalised === 'user-interrupt') {
    return null
  }
  return exitReason
}

function formatDurationShort(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`
  }
  const seconds = ms / 1000
  if (seconds < 60) {
    return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`
  }
  const minutes = Math.floor(seconds / 60)
  const remaining = Math.round(seconds % 60)
  return `${minutes}m${remaining.toString().padStart(2, '0')}s`
}
