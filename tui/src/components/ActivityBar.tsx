import { Box, Text, useStdout, type DOMElement } from 'ink'
import { useStore } from '@nanostores/react'
import stringWidth from 'string-width'
import { useCallback, useEffect, useRef, type ReactElement } from 'react'

import { useAnimationFrame } from '../lib/useAnimationFrame.js'
import { shimmer, spinnerVerbAt } from '../lib/shimmer.js'
import {
  isDirectWriteShimmerEnabled,
  measureShimmerCell,
  startShimmerWriter
} from '../lib/shimmerWriter.js'
import { theme } from '../lib/theme.js'
import { categoryFor, verbForCategory } from '../lib/toolCategory.js'
import { activeTodoTitle, formatTodoPreviewLines } from '../lib/todos.js'
import { activityState, type ActivityStatus } from '../store/activityStore.js'
import { sessionState } from '../store/sessionStore.js'

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
const SPINNER_FRAMES_ASCII = ['/', '-', '\\', '|']
// Slowed from 80 ms → 150 ms so the spinner reads as a calm "still working"
// indicator instead of a flickering distraction during long model calls. The
// shimmer animation reuses the same tick so we don't pay for two intervals.
const SPINNER_INTERVAL_MS = 150
// `requesting` sweeps faster than the calm default to read as "actively
// waiting on the model", mirroring Claude Code's 50 ms requesting glimmer.
const REQUESTING_INTERVAL_MS = 70
// Half-period of the `tool_use` colour pulse (bright for one half, dim for
// the next) → a ~1 s flash cycle.
const TOOL_FLASH_HALF_PERIOD_MS = 500

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
  requesting: '◍',
  thinking: '◐',
  responding: '◑',
  tool_input: '◓',
  tool_use: '◒',
  cancelled: '◯',
  error: '✗'
}

const STATIC_ICON_ASCII: Record<ActivityStatus, string> = {
  idle: 'o',
  starting: '.',
  requesting: ':',
  thinking: '*',
  responding: '>',
  tool_input: '#',
  tool_use: '@',
  cancelled: 'o',
  error: 'x'
}

const ACTIVE_STATES: ReadonlySet<ActivityStatus> = new Set([
  'requesting',
  'thinking',
  'responding',
  'tool_input',
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

export function ActivityBar({ animate = true }: { animate?: boolean } = {}): ReactElement {
  const activity = useStore(activityState)
  const session = useStore(sessionState)
  const { stdout } = useStdout()
  const ascii = !theme.isUnicodeAllowed()
  const activityElementRef = useRef<DOMElement | null>(null)
  const tokenTurnStartedAtRef = useRef<number | null | undefined>(undefined)
  const displayedResponseLengthRef = useRef(0)
  const shouldAnimate =
    animate && ACTIVE_STATES.has(activity.status) && !activity.interruptPending
  // Per-mode animation, mirroring Claude Code's `useShimmerAnimation` /
  // SpinnerAnimationRow: `requesting` glimmers noticeably faster (the
  // request is in flight and we want it to read as "actively waiting"),
  // while `tool_use` pulses instead of sweeping (see flash below). Other
  // active states keep the calm 150 ms sweep.
  const intervalMs =
    activity.status === 'requesting' ? REQUESTING_INTERVAL_MS : SPINNER_INTERVAL_MS
  // `tool_use` uses a colour flash rather than the shimmer sweep, so it must
  // not take the direct-write fast path (which only knows how to sweep).
  const directWriteActive =
    shouldAnimate &&
    activity.status !== 'tool_use' &&
    isDirectWriteShimmerEnabled() &&
    theme.isColorEnabled()
  const [animationRef, animationTime] = useAnimationFrame(
    shouldAnimate && !directWriteActive ? intervalMs : null
  )
  const setActivityElement = useCallback(
    (element: DOMElement | null) => {
      activityElementRef.current = element
      animationRef(element)
    },
    [animationRef]
  )
  const animationTick = Math.floor(animationTime / intervalMs)
  // tool_use flash: toggle between the base status colour and the brighter
  // text colour on a ~1 s cycle so the running tool "pulses". A discrete
  // toggle stands in for Claude Code's sinusoidal `flashOpacity` since the
  // terminal can't interpolate opacity.
  const toolFlashBright =
    activity.status === 'tool_use' &&
    shouldAnimate &&
    Math.floor(animationTime / TOOL_FLASH_HALF_PERIOD_MS) % 2 === 0

  const isError = activity.status === 'error' || activity.status === 'cancelled'
  const colorName = isError ? 'error' : activity.status === 'idle' ? 'dim' : 'status'
  const colorProps = theme.colorProps(colorName)
  const isActive = ACTIVE_STATES.has(activity.status)
  const frames = ascii ? SPINNER_FRAMES_ASCII : SPINNER_FRAMES
  const icon = isActive && animate
    ? frames[animationTick % frames.length]
    : (ascii ? STATIC_ICON_ASCII : STATIC_ICON)[activity.status]

  const todoTitle = activeTodoTitle(activity.todos)
  const verb = todoTitle ?? verbForStatus(activity)
  const elapsedMs = activity.thinkingStartedAt
    ? Math.max(0, Date.now() - activity.thinkingStartedAt)
    : null

  // Claude Code drives the live spinner token display from response length
  // updates, including non-visible tool/control deltas. In Aether those arrive
  // as `responseChars`; provider usage is kept as a coarse fallback for
  // providers that cannot expose every streamed control-plane fragment.
  const currentResponseLength = Math.max(
    activity.responseChars,
    activity.tokensOut * TOKEN_CHAR_RATIO
  )
  if (tokenTurnStartedAtRef.current !== activity.turnStartedAt) {
    tokenTurnStartedAtRef.current = activity.turnStartedAt
    displayedResponseLengthRef.current = currentResponseLength
  } else if (!isActive || currentResponseLength <= displayedResponseLengthRef.current) {
    displayedResponseLengthRef.current = currentResponseLength
  } else {
    const gap = currentResponseLength - displayedResponseLengthRef.current
    const increment =
      gap < 70
        ? 3
        : gap < 200
          ? Math.max(8, Math.ceil(gap * 0.15))
          : 50
    displayedResponseLengthRef.current = Math.min(
      displayedResponseLengthRef.current + increment,
      currentResponseLength
    )
  }
  const estimatedTokensOut = Math.round(displayedResponseLengthRef.current / TOKEN_CHAR_RATIO)
  const tokensOutDisplay = estimatedTokensOut >= MIN_DISPLAY_TOKENS ? estimatedTokensOut : 0

  const segments: string[] = []
  // Mirrors Python `activity.py:236-241` — only the output count is surfaced
  // in the live bar; input tokens stay in `/stats` and the per-turn footer.
  // The arrow doubles as a flow-direction hint, matching Claude Code's
  // SpinnerModeGlyph: `↑` while `requesting` (the request is flowing out to
  // the model and we're waiting on it), `↓` once tokens are streaming back.
  if (tokensOutDisplay) {
    const tokenArrow = activity.status === 'requesting' ? '↑' : '↓'
    segments.push(`${tokenArrow} ${formatTokens(tokensOutDisplay)} tokens`)
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

  // tool_use pulses (flash) rather than sweeps, so skip the shimmer slices
  // for it and let the flash colour below carry the animation instead.
  const useShimmer = isActive && animate && activity.status !== 'tool_use'
  const shimmerSlices = useShimmer ? shimmer(verb, animationTick) : null
  const shimmerColor = theme.color('text') ?? 'white'
  const verbColorProps = toolFlashBright ? theme.colorProps('text') : colorProps

  const todoLines = formatTodoPreviewLines(activity.todos, {
    ascii,
    width: Math.max(20, cols - 2)
  })

  useEffect(() => {
    if (!directWriteActive) {
      return
    }
    const stream = stdout as NodeJS.WriteStream | undefined
    const baseColor = theme.color(colorName)
    const highlightColor = theme.color('text')
    if (!stream || !baseColor || !highlightColor) {
      return
    }
    const cell = measureShimmerCell(
      activityElementRef.current,
      stream,
      stringWidth(`${icon ?? ' '} `)
    )
    if (!cell) {
      return
    }
    const writer = startShimmerWriter({
      stdout: stream,
      row: cell.row,
      col: cell.col,
      label: verb,
      baseColor,
      highlightColor,
      intervalMs
    })
    return () => {
      writer?.stop()
    }
  }, [colorName, cols, directWriteActive, icon, intervalMs, stdout, verb])

  // Mirror Python `_interrupt_visual_pending` — once the user requested an
  // interrupt we drop the spinner/segments immediately and read as
  // "cancelling" until the gateway's cancelled/error event lands. Without
  // this latch the bar keeps spinning for the round-trip duration, which
  // reads as "the interrupt didn't register".
  if (activity.interruptPending) {
    return <></>
  }

  return (
    <Box flexDirection="column" ref={setActivityElement}>
      <Box>
        <Text {...colorProps}>{icon ?? ' '} </Text>
        {shimmerSlices && !directWriteActive ? (
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
          <Text bold {...verbColorProps}>{verb}</Text>
        )}
        {detailSegment ? <Text dimColor>{detailSegment}</Text> : null}
        {elapsedSegment ? <Text dimColor>{elapsedSegment}</Text> : null}
        {kept.length > 0 ? <Text dimColor> · {kept.join(' · ')}</Text> : null}
      </Box>
      {todoLines.length > 0 ? (
        <Box flexDirection="column" marginLeft={2}>
          {todoLines.map((line, idx) => (
            <Text key={idx} dimColor={idx !== 0}>
              {line}
            </Text>
          ))}
        </Box>
      ) : null}
    </Box>
  )
}

function verbForStatus(activity: { status: ActivityStatus; statusDetail: string | null; turnVerbIndex: number }): string {
  switch (activity.status) {
    case 'requesting':
    case 'thinking':
    case 'responding':
      return spinnerVerbAt(activity.turnVerbIndex)
    case 'tool_input':
      return 'Preparing'
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
