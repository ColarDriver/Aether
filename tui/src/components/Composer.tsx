import { Box, Text, useInput } from 'ink'
import { useStore } from '@nanostores/react'
import { useMemo, useRef, useState, type ReactElement, type ReactNode } from 'react'

import { applyCompletion, buildCompleterState } from '../lib/slashCompleter.js'
import { theme } from '../lib/theme.js'
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

  const completer = useMemo(
    () => buildCompleterState(state.draft, session.catalog),
    [state.draft, session.catalog]
  )

  function flashHint(message: string): void {
    setHint(message)
    setTimeout(() => setHint((current) => (current === message ? null : current)), 1500)
  }

  useInput(
    (input, key) => {
      if (disabled) {
        if (key.ctrl && input === 'c') {
          onCancel()
        }
        return
      }

      // ── Cancel / exit chord ────────────────────────────────────────
      if (key.ctrl && input === 'c') {
        const now = Date.now()
        const lastCtrlC = ctrlCTimerRef.current
        if (state.draft) {
          composerActions.clear()
          return
        }
        if (busy) {
          onCancel()
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
      if (key.ctrl && input === 'd') {
        if (!state.draft) {
          onCancel()
          return
        }
        // Ctrl-D in mid-draft deletes char forward (readline convention).
        composerActions.deleteForward()
        return
      }

      // ── Readline-style line/word ops ──────────────────────────────
      if (key.ctrl && input === 'a') {
        composerActions.moveLineStart()
        return
      }
      if (key.ctrl && input === 'e') {
        composerActions.moveLineEnd()
        return
      }
      if (key.ctrl && input === 'u') {
        composerActions.killToLineStart()
        return
      }
      if (key.ctrl && input === 'k') {
        composerActions.killToLineEnd()
        return
      }
      if (key.ctrl && input === 'w') {
        composerActions.deleteWordBackward()
        return
      }
      if (key.ctrl && input === 'b') {
        composerActions.moveLeft()
        return
      }
      if (key.ctrl && input === 'f') {
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

      // ── Tab: slash completion or focus hand-off ──────────────────
      if (key.tab) {
        if (completer.active && completer.matches.length > 0) {
          const direction = key.shift ? -1 : 1
          const nextCursor =
            (completerCursor + direction + completer.matches.length) % completer.matches.length
          setCompleterCursor(nextCursor)
          const candidate = completer.matches[nextCursor]
          if (candidate) {
            composerActions.setDraft(applyCompletion(state.draft, candidate))
          }
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
        composerActions.backspace()
        return
      }
      if (key.return) {
        if (state.multiline || key.meta) {
          composerActions.newline()
        } else {
          submit(state.draft, busy === true, onSubmit)
        }
        return
      }
      if (input) {
        composerActions.insert(input)
        setCompleterCursor(0)
      }
    },
    { isActive }
  )

  const borderColor = hasOverlay || owner !== 'composer' ? 'gray' : disabled ? 'gray' : 'cyan'

  return (
    <Box flexDirection="column">
      {completer.active && completer.matches.length > 0 ? (
        <SlashPopup
          matches={completer.matches}
          cursor={completerCursor}
          fullCount={completer.matches.length}
        />
      ) : null}
      <Box borderStyle="single" borderColor={borderColor} paddingX={1}>
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
 * Mirrors the Python `SlashCompleter` floating menu (claude-code style) —
 * each row shows `/cmd` plus its description, with the focused row
 * highlighted. Tab cycles, Enter executes the highlighted command (handled
 * upstream in the composer's input handler).
 */
function SlashPopup({
  matches,
  cursor,
  fullCount
}: {
  matches: ReadonlyArray<{ name: string; description: string }>
  cursor: number
  fullCount: number
}): ReactElement {
  const VISIBLE = 10
  // Keep the focused row inside the visible window.
  const start = Math.max(0, Math.min(cursor - Math.floor(VISIBLE / 2), Math.max(0, matches.length - VISIBLE)))
  const slice = matches.slice(start, start + VISIBLE)
  const nameWidth = Math.max(...slice.map((cmd) => cmd.name.length), 8)
  const accent = theme.colorProps('accent')
  const dim = theme.colorProps('dim')
  const border = theme.color('border')
  return (
    <Box
      flexDirection="column"
      borderStyle="single"
      {...(border ? { borderColor: border } : {})}
      paddingX={1}
    >
      {slice.map((cmd, idx) => {
        const realIdx = start + idx
        const focused = realIdx === cursor
        const namePadded = cmd.name.padEnd(nameWidth)
        return focused ? (
          <Box key={cmd.name}>
            <Text {...accent}>{'▸ '}</Text>
            <Text {...accent} bold>
              {namePadded}
            </Text>
            <Text> </Text>
            <Text>{cmd.description}</Text>
          </Box>
        ) : (
          <Box key={cmd.name}>
            <Text>{'  '}</Text>
            <Text {...accent}>{namePadded}</Text>
            <Text> </Text>
            <Text {...dim}>{cmd.description}</Text>
          </Box>
        )
      })}
      {fullCount > VISIBLE ? (
        <Box>
          <Text {...dim}>
            ↹ {cursor + 1}/{fullCount} · Tab next · Shift+Tab prev · Enter run · ESC close
          </Text>
        </Box>
      ) : (
        <Box>
          <Text {...dim}>↹ Tab next · Enter run · ESC close</Text>
        </Box>
      )}
    </Box>
  )
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
  const accentProps = theme.colorProps('accent')

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
            <Text {...accentProps}>{prefix}</Text>
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

function ComposerFooter({
  hint,
  queued,
  busy
}: {
  hint: string | null
  queued: number
  busy: boolean
}): ReactElement | null {
  if (!hint && queued === 0 && !busy) {
    return null
  }
  return (
    <Box paddingX={1}>
      {queued > 0 ? <Text color="magenta">▸ queued ({queued}) </Text> : null}
      {busy ? <Text dimColor>· turn running · ESC cancel </Text> : null}
      {hint ? <Text dimColor>· {hint}</Text> : null}
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
