import { Box, Text } from 'ink'
import { useStore } from '@nanostores/react'
import { useEffect, type ReactElement } from 'react'

import { theme } from '../lib/theme.js'
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
  const isActive = ACTIVE_STATES.has(activity.status)
  const frames = ascii ? SPINNER_FRAMES_ASCII : SPINNER_FRAMES
  const icon = isActive
    ? frames[activity.animationTick % frames.length]
    : (ascii ? STATIC_ICON_ASCII : STATIC_ICON)[activity.status]

  // Stable verb — never rotates while a turn is in flight. Earlier we tried
  // pondering/forging/channelling/composing rotation (mirroring Python
  // `THINKING_VERBS`) but during a long-blocked LLM call the verb word
  // changing every ~3s reads as "the UI is doing something" rather than
  // "the model is taking a while", which actively misleads the user. The
  // spinner glyph + elapsed-second counter convey progress without the
  // visual churn.
  const verb = verbForStatus(activity.status)
  const elapsed = activity.thinkingStartedAt
    ? Math.max(0, Math.floor((Date.now() - activity.thinkingStartedAt) / 1000))
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
  if (activity.iteration > 0) {
    if (activity.maxIterations) {
      segments.push(`iter ${activity.iteration}/${activity.maxIterations}`)
    } else {
      segments.push(`iter ${activity.iteration}`)
    }
  }
  if (activity.tokensIn || tokensOutDisplay) {
    segments.push(
      `${formatTokens(activity.tokensIn)} in / ${formatTokens(tokensOutDisplay)} out`
    )
  }
  if (session.sessionId) {
    segments.push(session.sessionId.slice(0, 8))
  }
  if (session.model) {
    segments.push(session.model)
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

  // No outer marginTop — Python pins the activity bar flush above the
  // composer with no spacer. The bar already has enough visual weight
  // (spinner glyph + colour) to read as a separate region.
  return (
    <Box>
      <Text {...colorProps}>
        {icon ?? ' '} <Text bold>{verb}</Text>
        {activity.statusDetail ? <Text dimColor> · {activity.statusDetail}</Text> : null}
        {elapsed !== null && elapsed > 0 ? (
          <Text dimColor> · {elapsed}s</Text>
        ) : null}
        {segments.length > 0 ? <Text dimColor> · {segments.join(' · ')}</Text> : null}
      </Text>
    </Box>
  )
}

function verbForStatus(status: ActivityStatus): string {
  switch (status) {
    case 'thinking':
      return 'thinking'
    case 'responding':
      return 'responding'
    case 'tool_use':
      return 'tool use'
    case 'cancelled':
      return 'cancelled'
    case 'error':
      return 'error'
    case 'idle':
      return 'idle'
    case 'starting':
      return 'starting'
  }
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
