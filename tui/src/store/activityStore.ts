import { atom } from 'nanostores'

import type { TodoItem } from '../lib/todos.js'

export type ActivityStatus =
  | 'idle'
  | 'starting'
  | 'thinking'
  | 'responding'
  | 'tool_use'
  | 'cancelled'
  | 'error'

export interface ActivityState {
  status: ActivityStatus
  statusDetail: string | null
  /** Stable per-turn seed used to pick the Python-style spinner verb. */
  turnVerbIndex: number
  iteration: number
  maxIterations: number | null
  tokensIn: number
  tokensOut: number
  cacheRead: number
  cacheWrite: number
  loopState: string | null
  /** When non-null, the activity bar starts a wall-clock timer for "thought for Ns". */
  thinkingStartedAt: number | null
  /** Set when at least one streaming response token has arrived this turn. */
  responseStartedAt: number | null
  /** Index that drives spinner/shimmer animation; UI ticks this on a 100ms loop. */
  animationTick: number
  /**
   * Per-turn rolling stats — used by the per-turn footer note that prints
   * after a `done`/`cancelled`/`error` event. Reset by `beginTurn`.
   */
  turnStartedAt: number | null
  turnIterations: number
  turnTools: number
  turnErrors: number
  /**
   * Streamed character count this turn — used by the live activity-bar token
   * counter. This includes visible text deltas and count-only progress for
   * non-visible control-plane output such as tool-call argument fragments.
   */
  responseChars: number
  /**
   * Snapshot of the most recently completed turn — populated by `endTurn`
   * and read by `/stats`. Mirrors Python `ReplState.ui.stats` which
   * survives turn boundaries so `/stats` can show last-turn metrics.
   */
  lastTurn: TurnSummary | null
  /** Current session todo list, updated from todo_write tool calls. */
  todos: TodoItem[]
  /**
   * Latch set the moment the user requests an interrupt (ESC during a busy
   * turn). Mirrors Python `app.py:_interrupt_visual_pending` — the gateway
   * confirmation (cancelled/error event) takes time to round-trip, so we
   * hide the activity-bar spinner and surface an "interrupting…" hint in
   * the footer immediately. Cleared by the next `beginTurn` / `endTurn`.
   */
  interruptPending: boolean
}

const initialState: ActivityState = {
  status: 'starting',
  statusDetail: null,
  turnVerbIndex: 0,
  iteration: 0,
  maxIterations: null,
  tokensIn: 0,
  tokensOut: 0,
  cacheRead: 0,
  cacheWrite: 0,
  loopState: null,
  thinkingStartedAt: null,
  responseStartedAt: null,
  animationTick: 0,
  turnStartedAt: null,
  turnIterations: 0,
  turnTools: 0,
  turnErrors: 0,
  responseChars: 0,
  lastTurn: null,
  todos: [],
  interruptPending: false
}

export interface TurnSummary {
  iterations: number
  tools: number
  errors: number
  durationMs: number
  /** Total streamed assistant characters this turn. */
  chars: number
}

export const activityState = atom<ActivityState>(initialState)

// ──────────────────────────────────────────────────────────────────────────
// Token throttle
//
// Streaming providers emit a usage event per chunk (~50 ms apart in the worst
// case). Re-rendering Ink at that rate causes visible flicker for a feature
// nobody reads more than ~10 Hz anyway. We coalesce token deltas into a
// pending bucket and flush every TOKEN_FLUSH_MS.
// ──────────────────────────────────────────────────────────────────────────

const TOKEN_FLUSH_MS = 100
let pendingTokens: { input: number; output: number; cacheRead: number; cacheWrite: number } = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0
}
let flushTimer: NodeJS.Timeout | null = null
let nextTurnVerbIndex = 0

function scheduleTokenFlush(): void {
  if (flushTimer) {
    return
  }
  flushTimer = setTimeout(() => {
    flushTimer = null
    const delta = pendingTokens
    pendingTokens = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
    const current = activityState.get()
    activityState.set({
      ...current,
      tokensIn: current.tokensIn + delta.input,
      tokensOut: current.tokensOut + delta.output,
      cacheRead: current.cacheRead + delta.cacheRead,
      cacheWrite: current.cacheWrite + delta.cacheWrite
    })
  }, TOKEN_FLUSH_MS)
}

export const activityActions = {
  resetForTests(): void {
    if (flushTimer) {
      clearTimeout(flushTimer)
      flushTimer = null
    }
    pendingTokens = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
    nextTurnVerbIndex = 0
    activityState.set(initialState)
  },

  beginTurn(): void {
    if (flushTimer) {
      clearTimeout(flushTimer)
      flushTimer = null
    }
    pendingTokens = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
    const current = activityState.get()
    const now = Date.now()
    activityState.set({
      ...current,
      status: 'thinking',
      statusDetail: null,
      turnVerbIndex: nextTurnVerbIndex++,
      thinkingStartedAt: now,
      responseStartedAt: null,
      iteration: 0,
      loopState: null,
      tokensIn: 0,
      tokensOut: 0,
      cacheRead: 0,
      cacheWrite: 0,
      turnStartedAt: now,
      turnIterations: 0,
      turnTools: 0,
      turnErrors: 0,
      responseChars: 0,
      interruptPending: false
    })
  },

  markInterruptPending(): void {
    const current = activityState.get()
    if (current.interruptPending) {
      return
    }
    activityState.set({ ...current, interruptPending: true })
  },

  addResponseChars(count: number): void {
    if (count <= 0) {
      return
    }
    const current = activityState.get()
    activityState.set({
      ...current,
      responseChars: current.responseChars + count
    })
  },

  endTurn(reason: 'done' | 'cancelled' | 'error'): TurnSummary {
    const current = activityState.get()
    const status: ActivityStatus =
      reason === 'cancelled' ? 'cancelled' : reason === 'error' ? 'error' : 'idle'
    const startedAt = current.turnStartedAt ?? Date.now()
    const summary: TurnSummary = {
      iterations: current.turnIterations,
      tools: current.turnTools,
      errors: current.turnErrors,
      durationMs: Math.max(0, Date.now() - startedAt),
      chars: current.responseChars
    }
    activityState.set({
      ...current,
      status,
      statusDetail: null,
      thinkingStartedAt: null,
      responseStartedAt: null,
      turnStartedAt: null,
      // Clear the latch — the server has confirmed terminal state, so the
      // "interrupting…" hint should fall away and the activity bar can show
      // the canonical cancelled/error/idle status instead.
      interruptPending: false,
      lastTurn: summary
    })
    return summary
  },

  bumpToolCounter(isError: boolean): void {
    const current = activityState.get()
    activityState.set({
      ...current,
      turnTools: current.turnTools + 1,
      turnErrors: current.turnErrors + (isError ? 1 : 0)
    })
  },

  bumpErrorCounter(): void {
    const current = activityState.get()
    activityState.set({ ...current, turnErrors: current.turnErrors + 1 })
  },

  setStatus(status: ActivityStatus, detail: string | null = null): void {
    const current = activityState.get()
    const next: ActivityState = {
      ...current,
      status,
      statusDetail: detail
    }
    if (status === 'thinking' && current.status !== 'thinking') {
      next.thinkingStartedAt = Date.now()
    }
    if (status === 'responding' && current.responseStartedAt === null) {
      next.responseStartedAt = Date.now()
    }
    if (status === 'idle' || status === 'cancelled' || status === 'error') {
      next.thinkingStartedAt = null
      next.responseStartedAt = null
    }
    activityState.set(next)
  },

  setTodos(todos: TodoItem[]): void {
    const current = activityState.get()
    activityState.set({ ...current, todos })
  },

  setIteration(iteration: number, maxIterations?: number | null): void {
    const current = activityState.get()
    activityState.set({
      ...current,
      iteration,
      // Track the high-water-mark of iterations so the per-turn footer can
      // report a stable count even after the engine ends and resets to 0.
      turnIterations: Math.max(current.turnIterations, iteration),
      ...(maxIterations !== undefined ? { maxIterations: maxIterations } : {})
    })
  },

  setLoopState(state: string | null): void {
    const current = activityState.get()
    activityState.set({ ...current, loopState: state })
  },

  bumpAnimation(): void {
    const current = activityState.get()
    activityState.set({ ...current, animationTick: current.animationTick + 1 })
  },

  /** Throttled — call from every `usage` event without worry. */
  addUsage(input: {
    input: number
    output: number
    cacheRead?: number
    cacheWrite?: number
  }): void {
    pendingTokens.input += input.input
    pendingTokens.output += input.output
    pendingTokens.cacheRead += input.cacheRead ?? 0
    pendingTokens.cacheWrite += input.cacheWrite ?? 0
    scheduleTokenFlush()
  },

  /** Synchronous flush — primarily used by tests / `endTurn` cleanup. */
  flushUsage(): void {
    if (flushTimer) {
      clearTimeout(flushTimer)
      flushTimer = null
    }
    if (
      pendingTokens.input === 0 &&
      pendingTokens.output === 0 &&
      pendingTokens.cacheRead === 0 &&
      pendingTokens.cacheWrite === 0
    ) {
      return
    }
    const delta = pendingTokens
    pendingTokens = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
    const current = activityState.get()
    activityState.set({
      ...current,
      tokensIn: current.tokensIn + delta.input,
      tokensOut: current.tokensOut + delta.output,
      cacheRead: current.cacheRead + delta.cacheRead,
      cacheWrite: current.cacheWrite + delta.cacheWrite
    })
  }
}
