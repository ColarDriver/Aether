import { Box, Text, useInput } from 'ink'
import { useStore } from '@nanostores/react'
import stringWidth from 'string-width'
import { useMemo, useRef, useState, type ReactElement, type ReactNode } from 'react'

import { applyCompletion, buildCompleterState } from '../lib/slashCompleter.js'
import { isOnlyMouseTrackingInput, stripMouseTrackingSequences } from '../lib/terminalMouse.js'
import { theme } from '../lib/theme.js'
import { activityState } from '../store/activityStore.js'
import { chatActions } from '../store/chatStore.js'
import { composerActions, composerState } from '../store/composerStore.js'
import { focusActions, focusOwner } from '../store/focusStore.js'
import { overlayStack } from '../store/overlayStore.js'
import { sessionState } from '../store/sessionStore.js'

const DOUBLE_ESC_MS = 800
const DOUBLE_CTRL_C_MS = 2000

export interface ComposerProps {
  disabled?: boolean
  busy?: boolean
  onSubmit(text: string): void
  onCancel(): void
}

export function Composer(props: ComposerProps): ReactElement {
  const { disabled, busy, onSubmit, onCancel } = props
  const state = useStore(composerState)
  const overlays = useStore(overlayStack)
  const owner = useStore(focusOwner)
  const session = useStore(sessionState)
  const hasOverlay = overlays.length > 0
  const isActive = !hasOverlay && owner === 'composer'

  const escTimerRef = useRef<number>(0)
  const ctrlCTimerRef = useRef<number>(0)
  const [hint, setHint] = useState<string | null>(null)
  const [completerCursor, setCompleterCursor] = useState(0)
  // Anchor the slash popup to the leading token at the moment cycling started,
  // not to the current draft. Without this, Tab / Up / Down stamp the draft
  // with the highlighted completion, which then re-narrows `matches` to that
  // single command and the user gets stuck on the first cycle. Mirrors
  // prompt_toolkit's "menu mode" Python uses: the buffer text updates while
  // the candidate list stays pinned to the original prefix.
  const [completerAnchor, setCompleterAnchor] = useState<string | null>(null)

  const completer = useMemo(() => {
    const source = completerAnchor ?? state.draft
    return buildCompleterState(source, session.catalog)
  }, [completerAnchor, state.draft, session.catalog])

  function flashHint(message: string): void {
    setHint(message)
    setTimeout(() => setHint((current) => (current === message ? null : current)), 1500)
  }

  useInput(
    (input, key) => {
      if (isOnlyMouseTrackingInput(input)) {
        return
      }
      const textInput = stripMouseTrackingSequences(input)

      if (disabled) {
        if (key.ctrl && textInput === 'c') {
          onCancel()
        }
        return
      }

      // Helpers used by both ESC (close popup) and the editing branches
      // below. Cycling keys (Tab / Up / Down) call cycleCompleter and
      // leave the anchor in place; every other key that modifies the
      // draft calls resetCompleterAnchor before its action so the popup
      // returns to live mode.
      const cycleCompleter = (delta: number): void => {
        if (!(completer.active && completer.matches.length > 0)) {
          return
        }
        // First cycle pins the popup to the current token so subsequent
        // cycles see the same candidate list even after the draft is
        // stamped with the highlighted command. Mirrors prompt_toolkit's
        // "menu mode" Python uses.
        if (completerAnchor === null) {
          setCompleterAnchor(state.draft)
        }
        const nextCursor =
          (completerCursor + delta + completer.matches.length) % completer.matches.length
        setCompleterCursor(nextCursor)
        const candidate = completer.matches[nextCursor]
        if (candidate) {
          composerActions.setDraft(applyCompletion(state.draft, candidate))
        }
      }
      const resetCompleterAnchor = (): void => {
        if (completerAnchor !== null) {
          setCompleterAnchor(null)
        }
      }
      const completerActive = completer.active && completer.matches.length > 0

      // ── Cancel / exit chord ────────────────────────────────────────
      if (key.ctrl && textInput === 'c') {
        const now = Date.now()
        const lastCtrlC = ctrlCTimerRef.current
        if (busy) {
          onCancel()
          return
        }
        if (state.draft) {
          resetCompleterAnchor()
          composerActions.clear()
          return
        }
        if (now - lastCtrlC < DOUBLE_CTRL_C_MS) {
          onCancel()
          return
        }
        ctrlCTimerRef.current = now
        flashHint('press Ctrl-C again to exit')
        return
      }
      if (key.ctrl && textInput === 'd') {
        if (!state.draft) {
          onCancel()
          return
        }
        // Ctrl-D in mid-draft deletes char forward (readline convention).
        resetCompleterAnchor()
        composerActions.deleteForward()
        return
      }

      // ── Readline-style line/word ops ──────────────────────────────
      if (key.ctrl && textInput === 'a') {
        composerActions.moveLineStart()
        return
      }
      if (key.ctrl && textInput === 'e') {
        composerActions.moveLineEnd()
        return
      }
      if (key.ctrl && textInput === 'u') {
        resetCompleterAnchor()
        composerActions.killToLineStart()
        return
      }
      if (key.ctrl && textInput === 'k') {
        resetCompleterAnchor()
        composerActions.killToLineEnd()
        return
      }
      if (key.ctrl && textInput === 'w') {
        resetCompleterAnchor()
        composerActions.deleteWordBackward()
        return
      }
      if (key.ctrl && textInput === 'b') {
        composerActions.moveLeft()
        return
      }
      if (key.ctrl && textInput === 'f') {
        composerActions.moveRight()
        return
      }

      // ── ESC chain ────────────────────────────────────────────────
      if (key.escape) {
        const now = Date.now()
        const lastEsc = escTimerRef.current
        if (busy) {
          onCancel()
          return
        }
        // When the slash popup is anchored (user cycled), ESC closes the
        // popup back to its live state without clearing the draft. Same
        // behaviour as prompt_toolkit's completion-menu Python uses, so a
        // user who cycled past the intended command can press ESC and
        // keep the cycled draft for editing.
        if (completerAnchor !== null) {
          resetCompleterAnchor()
          return
        }
        if (state.draft) {
          composerActions.clear()
          return
        }
        if (state.queued.length > 0) {
          composerActions.popQueued()
          return
        }
        if (now - lastEsc < DOUBLE_ESC_MS) {
          chatActions.reset()
          escTimerRef.current = 0
          flashHint('history cleared')
          return
        }
        escTimerRef.current = now
        flashHint('press ESC again to clear history')
        return
      }

      if (key.tab) {
        if (completerActive) {
          cycleCompleter(key.shift ? -1 : 1)
          return
        }
        focusActions.set('transcript')
        return
      }

      // ── Cursor navigation ────────────────────────────────────────
      if (key.leftArrow) {
        if (key.meta || key.ctrl) {
          composerActions.moveWordLeft()
        } else {
          composerActions.moveLeft()
        }
        return
      }
      if (key.rightArrow) {
        if (key.meta || key.ctrl) {
          composerActions.moveWordRight()
        } else {
          composerActions.moveRight()
        }
        return
      }
      if (key.upArrow) {
        if (completerActive) {
          cycleCompleter(-1)
          return
        }
        // In multiline mode, navigate within the buffer; otherwise walk history.
        if (state.multiline) {
          if (composerActions.moveLineUp()) {
            return
          }
        }
        composerActions.previousHistory()
        return
      }
      if (key.downArrow) {
        if (completerActive) {
          cycleCompleter(1)
          return
        }
        if (state.multiline) {
          if (composerActions.moveLineDown()) {
            return
          }
        }
        composerActions.nextHistory()
        return
      }
      // Ink reports Home/End sometimes as `pageUp`/`pageDown` on certain
      // terminal emulators; map both to line-start/line-end as the
      // conservative interpretation.
      if (key.pageUp) {
        composerActions.moveLineStart()
        return
      }
      if (key.pageDown) {
        composerActions.moveLineEnd()
        return
      }

      // ── Editing ──────────────────────────────────────────────────
      // Ink parses `\x08` (Ctrl-H/BS) as `key.backspace` and `\x7f` (DEL) as
      // `key.delete`, but on Linux/macOS terminals the Backspace key sends
      // `\x7f` — so we must treat both flags as backspace. Forward-delete is
      // available via Ctrl-D (readline) above. Ink's own source carries a
      // TODO about merging these in a future major version
      // (node_modules/ink/build/parse-keypress.js:431).
      if (key.backspace || key.delete) {
        resetCompleterAnchor()
        composerActions.backspace()
        return
      }
      if (key.return) {
        resetCompleterAnchor()
        if (state.multiline || key.meta) {
          composerActions.newline()
        } else {
          submit(state.draft, busy === true, onSubmit)
        }
        return
      }
      if (textInput) {
        resetCompleterAnchor()
        composerActions.insert(textInput)
        setCompleterCursor(0)
      }
    },
    { isActive }
  )

  const borderColor = theme.color('border') ?? 'gray'

  return (
    <Box flexDirection="column" width="100%">
      {completer.active && completer.matches.length > 0 ? (
        <SlashPopup
          matches={completer.matches}
          cursor={completerCursor}
          fullCount={completer.matches.length}
        />
      ) : null}
      <Box width="100%" borderStyle="single" borderColor={borderColor} paddingX={1}>
        <Box flexDirection="column" flexGrow={1}>
          <ComposerLines draft={state.draft} cursor={state.cursor} active={isActive} />
        </Box>
      </Box>
      <ComposerFooter
        hint={hint}
        queued={state.queued.length}
        busy={busy === true}
      />
    </Box>
  )
}

/**
 * Vertical popup rendered above the composer when the draft starts with `/`.
 * Mirrors Python `aether/cli/app.py` `completion-menu.*` styling exactly:
 *
 *   completion-menu.completion          → bg #1E293B, fg dim
 *   completion-menu.completion.current  → bg AETHER_PRIMARY, fg white bold
 *   completion-menu.meta                → bg #1E293B, fg dim italic
 *   completion-menu.meta.current        → bg AETHER_PRIMARY, fg white italic
 *
 * No border, no marker, no in-popup key hints — the composer's main footer
 * already advertises the keys. The selection is communicated entirely by
 * background colour, same as prompt_toolkit's default completion menu.
 */
function SlashPopup({
  matches,
  cursor
}: {
  matches: ReadonlyArray<{ name: string; description: string }>
  cursor: number
  fullCount: number
}): ReactElement {
  const VISIBLE = 10
  // Keep the focused row inside the visible window.
  const start = Math.max(0, Math.min(cursor - Math.floor(VISIBLE / 2), Math.max(0, matches.length - VISIBLE)))
  const slice = matches.slice(start, start + VISIBLE)
  const popupWidth = Math.max(24, composerPopupWidth())
  const nameWidth = Math.max(...slice.map((cmd) => stringWidth(cmd.name)), 8)
  const descWidth = Math.max(...slice.map((cmd) => stringWidth(cmd.description)), 1)
  const dimFg = theme.color('dim')
  const brandBg = theme.color('brand')
  const popupBg = theme.color('popup_bg')

  return (
    <Box flexDirection="column" width="100%">
      {slice.map((cmd, idx) => {
        const realIdx = start + idx
        const focused = realIdx === cursor
        const namePadded = padDisplayWidth(cmd.name, nameWidth)
        const descPadded = padDisplayWidth(cmd.description, descWidth)
        const filler = ' '.repeat(
          Math.max(
            0,
            popupWidth -
              stringWidth(` ${namePadded}  ${descPadded} `)
          )
        )
        if (focused) {
          // `completion-menu.completion.current` → bg=primary, fg=white bold
          // `completion-menu.meta.current`       → bg=primary, fg=white italic
          const bgProps = brandBg ? { backgroundColor: brandBg } : {}
          return (
            <Box key={cmd.name} width="100%">
              <Text {...bgProps} color="white">{' '}</Text>
              <Text {...bgProps} color="white" bold>{namePadded}</Text>
              <Text {...bgProps} color="white">{'  '}</Text>
              <Text {...bgProps} color="white" italic>{descPadded}</Text>
              <Text {...bgProps} color="white">{' ' + filler}</Text>
            </Box>
          )
        }
        // `completion-menu.completion` → bg=#1E293B, fg=dim
        // `completion-menu.meta`       → bg=#1E293B, fg=dim italic
        const bgProps = popupBg ? { backgroundColor: popupBg } : {}
        const fgProps = dimFg ? { color: dimFg } : {}
        return (
          <Box key={cmd.name} width="100%">
            <Text {...bgProps} {...fgProps}>{' '}</Text>
            <Text {...bgProps} {...fgProps}>{namePadded}</Text>
            <Text {...bgProps} {...fgProps}>{'  '}</Text>
            <Text {...bgProps} {...fgProps} italic>{descPadded}</Text>
            <Text {...bgProps} {...fgProps}>{' ' + filler}</Text>
          </Box>
        )
      })}
    </Box>
  )
}

function composerPopupWidth(): number {
  const cols = process.stdout?.columns
  if (!Number.isFinite(cols) || !cols) {
    return 78
  }
  // App root uses `paddingX={1}`, so the composer spans roughly cols - 2.
  return Math.max(24, cols - 2)
}

function padDisplayWidth(value: string, width: number): string {
  const current = stringWidth(value)
  if (current >= width) {
    return value
  }
  return value + ' '.repeat(width - current)
}

/**
 * Render the draft line-by-line and inject an inverse-block cursor at the
 * correct character position. We deliberately render the cursor inline
 * rather than relying on terminal-level cursor positioning so the chrome
 * stays correct even when Ink has multiple lines being repainted.
 */
function ComposerLines({
  draft,
  cursor,
  active
}: {
  draft: string
  cursor: number
  active: boolean
}): ReactElement {
  const lines = draft.split('\n')
  const promptProps = theme.colorProps('brand')

  // Compute (lineIndex, columnIndex) for the cursor.
  let remaining = cursor
  let cursorLine = 0
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? ''
    if (remaining <= line.length) {
      cursorLine = i
      break
    }
    remaining -= line.length + 1 // +1 for the consumed '\n'
    cursorLine = i + 1
  }
  const cursorCol = remaining

  return (
    <>
      {lines.map((line, idx) => {
        const prefix = idx === 0 ? '› ' : '  '
        const isCursorLine = active && idx === cursorLine
        return (
          <Text key={idx}>
            <Text {...promptProps}>{prefix}</Text>
            {isCursorLine ? renderCursorLine(line, cursorCol) : <Text>{line || ' '}</Text>}
          </Text>
        )
      })}
    </>
  )
}

function renderCursorLine(line: string, col: number): ReactNode {
  const before = line.slice(0, col)
  const at = line[col] ?? ' '
  const after = line.slice(col + 1)
  return (
    <>
      {before ? <Text>{before}</Text> : null}
      <Text inverse>{at}</Text>
      {after ? <Text>{after}</Text> : null}
    </>
  )
}

// Mirrors Python `app.py:1165-1238` keymap fragments — kept visible at all
// times so users always know which keys map to which actions. State-conditional
// segments (queued count, "running…", "interrupting…", flash hint) layer on
// top of the static keymap rather than replacing it.
function ComposerFooter({
  hint,
  queued,
  busy
}: {
  hint: string | null
  queued: number
  busy: boolean
}): ReactElement {
  const activity = useStore(activityState)
  const dim = theme.colorProps('dim')
  const accent = theme.colorProps('accent')
  const escLabel = activity.interruptPending
    ? ' interrupted'
    : busy
      ? ' interrupt'
      : queued > 0
        ? ' pop queued'
        : ' clear'
  return (
    <Box paddingX={1} flexDirection="column">
      <Box>
        <Text {...dim}>
          <Text {...accent}>Enter</Text> send
          {'  '}
          <Text {...dim}>·</Text>{' '}
          <Text {...accent}>Shift+Enter</Text> newline
          {'  '}
          <Text {...dim}>·</Text>{' '}
          <Text {...accent}>ESC</Text>{escLabel}
          {'  '}
          <Text {...dim}>·</Text>{' '}
          <Text {...accent}>/help</Text> commands
          {'  '}
          <Text {...dim}>·</Text>{' '}
          <Text {...accent}>Ctrl-D</Text> exit
        </Text>
      </Box>
      {queued > 0 || busy || hint || activity.interruptPending ? (
        <Box>
          {queued > 0 ? (
            <Text bold color="magenta">
              ▸ queued ({queued}){' '}
            </Text>
          ) : null}
          {/* Interrupt-pending takes precedence over the plain busy hint — */}
          {/* mirrors Python `app.py:1222` priority order. */}
          {activity.interruptPending ? (
            <Text {...dim} italic>interrupting… </Text>
          ) : busy ? (
            <Text {...dim} italic>running… </Text>
          ) : null}
          {hint ? <Text {...dim}>· {hint}</Text> : null}
        </Box>
      ) : null}
    </Box>
  )
}

function submit(text: string, busy: boolean, onSubmit: (text: string) => void): void {
  const trimmed = text.trim()
  if (!trimmed) {
    return
  }
  if (busy) {
    composerActions.enqueue(trimmed)
    return
  }
  composerActions.commit(trimmed)
  onSubmit(trimmed)
}
