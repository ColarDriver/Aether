import { Box, Text } from 'ink'
import { useStore } from '@nanostores/react'
import { useEffect, type ReactElement } from 'react'

import { shimmer, thinkingVerbAt } from '../lib/shimmer.js'
import { theme } from '../lib/theme.js'
import { categoryFor, verbForCategory } from '../lib/toolCategory.js'
import { activityActions, activityState, type ActivityStatus } from '../store/activityStore.js'
import { sessionState } from '../store/sessionStore.js'

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
const SPINNER_FRAMES_ASCII = ['/', '-', '\\', '|']
// Slowed from 80 ms → 150 ms so the spinner reads as a calm "still working"
// indicator instead of a flickering distraction during long model calls. The
// shimmer animation reuses the same tick so we don't pay for two intervals.
const SPINNER_INTERVAL_MS = 150

// Mirrors Python `activity.TOKEN_CHAR_RATIO` / `MIN_DISPLAY_TOKENS`.
const TOKEN_CHAR_RATIO = 4
const MIN_DISPLAY_TOKENS = 3
// "thought for Ns" only appears after a real pause — sub-half-second
// thinks look broken with `thought for 0s`. Matches Python's
// `MIN_THINKING_DISPLAY_MS`.
const MIN_THINKING_DISPLAY_MS = 500

const STATIC_ICON: Record<ActivityStatus, string> = {
  idle: '◯',
  starting: '◌',
  thinking: '◐',
  responding: '◑',
  tool_use: '◒',
  cancelled: '◯',
  error: '✗'
}

const STATIC_ICON_ASCII: Record<ActivityStatus, string> = {
  idle: 'o',
  starting: '.',
  thinking: '*',
  responding: '>',
  tool_use: '@',
  cancelled: 'o',
  error: 'x'
}

const ACTIVE_STATES: ReadonlySet<ActivityStatus> = new Set([
  'thinking',
  'responding',
  'tool_use',
  'starting'
])

// Internal loop-state names from the engine that we do not want to surface
// verbatim — they read as jargon to end users (e.g. `LLM_CALL`).
const LOOP_STATE_LABELS: Record<string, string | null> = {
  LLM_CALL: null,
  TOOL_DISPATCH: null,
  TOOL_DONE: null,
  RUNNING: null,
  running: null,
  PRE_LLM: null,
  POST_LLM: null,
  COMPACTION: 'compacting',
  MAX_ITERATIONS: 'max iterations',
  CANCELLED: 'cancelled',
  INTERRUPTED: 'interrupted',
  FAILED: 'failed',
  ERROR: 'failed'
}

function loopStateLabel(state: string | null): string | null {
  if (!state) {
    return null
  }
  if (state in LOOP_STATE_LABELS) {
    const label = LOOP_STATE_LABELS[state]
    return label ?? null
  }
  // Default for unknown states: lowercase, dashes for underscores. Suppress
  // anything that still looks like raw enum jargon (all caps with underscores).
  if (/^[A-Z][A-Z0-9_]*$/.test(state)) {
    return null
  }
  return state
}

export function ActivityBar(): ReactElement {
  const activity = useStore(activityState)
  const session = useStore(sessionState)
  const ascii = !theme.isUnicodeAllowed()

  useEffect(() => {
    if (!ACTIVE_STATES.has(activity.status)) {
      return
    }
    const handle = setInterval(() => {
      activityActions.bumpAnimation()
    }, SPINNER_INTERVAL_MS)
    return () => {
      clearInterval(handle)
    }
  }, [activity.status])

  const isError = activity.status === 'error' || activity.status === 'cancelled'
  const colorName = isError ? 'error' : activity.status === 'idle' ? 'dim' : 'status'
  const colorProps = theme.colorProps(colorName)
  // Mirror Python `_interrupt_visual_pending` — once the user requested an
  // interrupt we drop the spinner/segments immediately and read as
  // "cancelling" until the gateway's cancelled/error event lands. Without
  // this latch the bar keeps spinning for the round-trip duration, which
  // reads as "the interrupt didn't register".
  if (activity.interruptPending) {
    const dim = theme.colorProps('dim')
    return (
      <Box>
        <Text {...dim} italic>
          {ascii ? '...' : '…'} interrupting
        </Text>
      </Box>
    )
  }
  const isActive = ACTIVE_STATES.has(activity.status)
  const frames = ascii ? SPINNER_FRAMES_ASCII : SPINNER_FRAMES
  const icon = isActive
    ? frames[activity.animationTick % frames.length]
    : (ascii ? STATIC_ICON_ASCII : STATIC_ICON)[activity.status]

  const verb = verbForStatus(activity)
  const elapsedMs = activity.thinkingStartedAt
    ? Math.max(0, Date.now() - activity.thinkingStartedAt)
    : null

  // Approximate token count from streamed chars when the gateway hasn't
  // emitted a usage event yet. Mirrors Python `response_chars // 4` with
  // the same MIN_DISPLAY_TOKENS floor so trivial replies don't flash
  // "↓ 1 token" for a single frame.
  const fallbackTokensOut = Math.floor(activity.responseChars / TOKEN_CHAR_RATIO)
  const tokensOutDisplay =
    activity.tokensOut > 0
      ? activity.tokensOut
      : fallbackTokensOut >= MIN_DISPLAY_TOKENS
        ? fallbackTokensOut
        : 0

  const segments: string[] = []
  // Mirrors Python `activity.py:236-241` — only the output (↓) count is
  // surfaced in the live bar; input tokens stay in `/stats` and the
  // per-turn footer. The `↓` arrow doubles as a flow-direction hint
  // ("tokens flowing out of the model"), matching Claude Code's
  // SpinnerAnimationRow convention.
  if (tokensOutDisplay) {
    segments.push(`↓ ${formatTokens(tokensOutDisplay)} tokens`)
  }
  if (session.sessionId) {
    segments.push(session.sessionId.slice(0, 8))
  }
  if (session.model) {
    segments.push(session.model)
  }
  if (activity.status === 'thinking' && activity.responseStartedAt === null) {
    segments.push('thinking')
  }
  // "thought for Ns" — appears once the response has started and the
  // pre-response wait was long enough to be meaningful. Mirrors Python
  // `thinking_status` semantics.
  if (
    activity.responseStartedAt &&
    activity.thinkingStartedAt &&
    activity.responseStartedAt > activity.thinkingStartedAt
  ) {
    const thoughtMs = activity.responseStartedAt - activity.thinkingStartedAt
    if (thoughtMs >= MIN_THINKING_DISPLAY_MS) {
      segments.push(`thought for ${Math.max(1, Math.round(thoughtMs / 1000))}s`)
    }
  }
  // Translate the engine's internal loop_state into either a friendly label
  // (compacting / max iterations / failed) or — for routine internal phases
  // like LLM_CALL — nothing at all. Showing `loop:LLM_CALL` to end users is
  // jargon: they only need to know we are working, not which engine sub-phase.
  const loopLabel = loopStateLabel(activity.loopState)
  if (loopLabel) {
    segments.push(loopLabel)
  }

  // Width-budget the suffix the way Python `activity.py:262-285` does:
  // build the verb+icon prefix first, then add elapsed/segments while
  // budget remains; drop trailing fields when over budget so a narrow
  // terminal doesn't wrap mid-row. The verb is always preserved.
  const cols = process.stdout?.columns ?? 100
  const prefix = `${icon ?? ' '} ${verb}`
  const detail = detailForStatus(activity)
  const detailSegment = detail ? ` · ${detail}` : ''
  const baseWidth = prefix.length + detailSegment.length
  const elapsedSegment =
    elapsedMs !== null && elapsedMs >= 1000 ? ` · ${formatDurationMs(elapsedMs)}` : ''
  // Build a flat list ordered by priority (most important kept first). Iter
  // and model identifiers are useful context; thinking-time and loop label
  // are nice-to-have. Order roughly matches Python's `suffix_fields`.
  const orderedSegments = [...segments]
  let runningWidth = baseWidth + elapsedSegment.length
  const kept: string[] = []
  for (const segment of orderedSegments) {
    const addition = ` · ${segment}`.length
    if (runningWidth + addition > cols - 2 && kept.length > 0) {
      break
    }
    kept.push(segment)
    runningWidth += addition
  }

  const shimmerSlices = isActive ? shimmer(verb, activity.animationTick) : null
  const shimmerColor = theme.color('text') ?? 'white'

  return (
    <Box>
      <Text {...colorProps}>{icon ?? ' '} </Text>
      {shimmerSlices ? (
        <>
          {shimmerSlices.before ? (
            <Text bold {...colorProps}>{shimmerSlices.before}</Text>
          ) : null}
          {shimmerSlices.highlight ? (
            <Text bold color={shimmerColor}>{shimmerSlices.highlight}</Text>
          ) : null}
          {shimmerSlices.after ? (
            <Text bold {...colorProps}>{shimmerSlices.after}</Text>
          ) : null}
        </>
      ) : (
        <Text bold {...colorProps}>{verb}</Text>
      )}
      {detailSegment ? <Text dimColor>{detailSegment}</Text> : null}
      {elapsedSegment ? <Text dimColor>{elapsedSegment}</Text> : null}
      {kept.length > 0 ? <Text dimColor> · {kept.join(' · ')}</Text> : null}
    </Box>
  )
}

function verbForStatus(activity: { status: ActivityStatus; statusDetail: string | null; turnVerbIndex: number }): string {
  switch (activity.status) {
    case 'thinking':
    case 'responding':
      return thinkingVerbAt(activity.turnVerbIndex)
    case 'tool_use':
      return presentVerbForTool(activity.statusDetail)
    case 'cancelled':
      return 'Cancelled'
    case 'error':
      return 'Error'
    case 'idle':
      return 'Idle'
    case 'starting':
      return 'Starting'
  }
}

function detailForStatus(activity: {
  status: ActivityStatus
  statusDetail: string | null
}): string | null {
  if (!activity.statusDetail) {
    return null
  }
  if (activity.status === 'tool_use') {
    return null
  }
  return activity.statusDetail
}

function presentVerbForTool(toolName: string | null): string {
  if (!toolName) {
    return 'Working'
  }
  const category = categoryFor(toolName)
  return verbForCategory(category, false)[0]
}

function formatTokens(value: number): string {
  if (value < 1000) {
    return String(value)
  }
  if (value < 1_000_000) {
    return `${(value / 1000).toFixed(1).replace(/\.0$/, '')}k`
  }
  return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`
}

/**
 * Mirror of Python `activity.format_duration_ms`: `12s` / `2m 14s` / `1h 03m`.
 * Activity bar reads from milliseconds, so this avoids the loss-of-precision
 * Math.floor(ms/1000) introduces when an in-flight turn crosses a minute
 * boundary mid-render.
 */
function formatDurationMs(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000))
  if (seconds < 60) {
    return `${seconds}s`
  }
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`
  }
  return `${Math.floor(seconds / 3600)}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, '0')}m`
}
