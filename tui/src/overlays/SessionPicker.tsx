import { Box, Text, useInput } from 'ink'
import { useEffect, useMemo, useState, type ReactElement } from 'react'

import type { JsonObject, SessionInfo, TranscriptMessage } from '../gatewayTypes.js'
import { overlayActions, type OverlayState } from '../store/overlayStore.js'

export interface SessionPickerPayload {
  sessions: SessionInfo[]
  /** Resolves the selected session into a transcript so the parent can render it. */
  resolveResume: (sessionId: string) => Promise<{ info: SessionInfo; messages: TranscriptMessage[] }>
  /** Hook the parent uses to apply the resume result (push messages, set session). */
  onResume: (info: SessionInfo, messages: TranscriptMessage[]) => void
  /** Optional pre-selected id; defaults to the most recent. */
  initialSessionId?: string
}

const PAGE_SIZE = 12

export function SessionPicker({
  overlay
}: {
  overlay: OverlayState<SessionPickerPayload>
}): ReactElement {
  const { sessions, resolveResume, onResume, initialSessionId } = overlay.payload
  const initialIndex = useMemo(() => {
    if (!initialSessionId) {
      return 0
    }
    const idx = sessions.findIndex((session) => session.session_id === initialSessionId)
    return idx === -1 ? 0 : idx
  }, [sessions, initialSessionId])

  const [cursor, setCursor] = useState(initialIndex)
  const [pageStart, setPageStart] = useState(0)
  const [resuming, setResuming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Keep the cursor inside the visible window.
  useEffect(() => {
    if (cursor < pageStart) {
      setPageStart(cursor)
      return
    }
    if (cursor >= pageStart + PAGE_SIZE) {
      setPageStart(cursor - PAGE_SIZE + 1)
    }
  }, [cursor, pageStart])

  useInput(
    (input, key) => {
      if (resuming) {
        return
      }
      if (key.escape) {
        overlayActions.dismiss(overlay.id, 'cancel')
        return
      }
      if (sessions.length === 0) {
        return
      }
      if (key.upArrow) {
        setCursor((value) => (value <= 0 ? 0 : value - 1))
        return
      }
      if (key.downArrow) {
        setCursor((value) => Math.min(sessions.length - 1, value + 1))
        return
      }
      if (key.pageUp) {
        setCursor((value) => Math.max(0, value - PAGE_SIZE))
        return
      }
      if (key.pageDown) {
        setCursor((value) => Math.min(sessions.length - 1, value + PAGE_SIZE))
        return
      }
      if (input === 'g') {
        setCursor(0)
        return
      }
      if (input === 'G') {
        setCursor(sessions.length - 1)
        return
      }
      if (key.return) {
        const session = sessions[cursor]
        if (!session) {
          return
        }
        setResuming(true)
        resolveResume(session.session_id)
          .then(({ info, messages }) => {
            onResume(info, messages)
            overlayActions.dismiss(overlay.id, 'commit', { sessionId: info.session_id })
          })
          .catch((err: unknown) => {
            const message = err instanceof Error ? err.message : String(err)
            setError(message)
            setResuming(false)
          })
      }
    },
    { isActive: true }
  )

  if (sessions.length === 0) {
    return (
      <Box flexDirection="column" borderStyle="round" borderColor="cyan" paddingX={1}>
        <Text bold color="cyan">
          Resume session
        </Text>
        <Box marginTop={1}>
          <Text>No saved sessions found.</Text>
        </Box>
        <Box marginTop={1}>
          <Text dimColor>ESC close</Text>
        </Box>
      </Box>
    )
  }

  const visible = sessions.slice(pageStart, pageStart + PAGE_SIZE)
  const hasAbove = pageStart > 0
  const hasBelow = pageStart + PAGE_SIZE < sessions.length

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="cyan" paddingX={1}>
      <Box>
        <Text bold color="cyan">
          Resume session
        </Text>
        <Text dimColor>
          {' '}
          · {sessions.length} entr{sessions.length === 1 ? 'y' : 'ies'}
        </Text>
      </Box>

      {hasAbove ? (
        <Box marginTop={1}>
          <Text dimColor>↑ {pageStart} more above</Text>
        </Box>
      ) : null}

      <Box flexDirection="column" marginTop={1}>
        {visible.map((session, idx) => {
          const realIdx = pageStart + idx
          const focused = realIdx === cursor
          return (
            <Text key={session.session_id}>
              {focused ? <Text color="cyan">▸ </Text> : <Text>  </Text>}
              <Text bold>{session.session_id.slice(0, 8)}</Text>
              <Text dimColor> · {formatRelativeTime(session.updated_at)}</Text>
              {session.provider || session.model ? (
                <Text dimColor>
                  {' '}
                  · {session.provider}
                  {session.model ? `/${session.model}` : ''}
                </Text>
              ) : null}
              {session.summary ? (
                <Text>
                  {' · '}
                  {truncate(session.summary, 60)}
                </Text>
              ) : null}
            </Text>
          )
        })}
      </Box>

      {hasBelow ? (
        <Box>
          <Text dimColor>↓ {sessions.length - pageStart - PAGE_SIZE} more below</Text>
        </Box>
      ) : null}

      <Box marginTop={1} flexDirection="column">
        {resuming ? <Text color="yellow">resuming…</Text> : null}
        {error ? <Text color="red">error: {error}</Text> : null}
        <Text dimColor>↑/↓ navigate · PgUp/PgDn page · Enter resume · ESC cancel</Text>
      </Box>
    </Box>
  )
}

function formatRelativeTime(epochSeconds: number | undefined): string {
  if (!epochSeconds) {
    return 'unknown'
  }
  const epochMs = epochSeconds < 1e12 ? epochSeconds * 1000 : epochSeconds
  const diff = Date.now() - epochMs
  if (diff < 0) {
    return new Date(epochMs).toISOString().slice(0, 16).replace('T', ' ')
  }
  if (diff < 60_000) {
    return `${Math.floor(diff / 1000)}s ago`
  }
  if (diff < 3_600_000) {
    return `${Math.floor(diff / 60_000)}m ago`
  }
  if (diff < 86_400_000) {
    return `${Math.floor(diff / 3_600_000)}h ago`
  }
  if (diff < 604_800_000) {
    return `${Math.floor(diff / 86_400_000)}d ago`
  }
  return new Date(epochMs).toISOString().slice(0, 10)
}

function truncate(value: string, max: number): string {
  if (!value) {
    return ''
  }
  if (value.length <= max) {
    return value
  }
  return `${value.slice(0, max - 1)}…`
}

// Re-export so the picker can be referenced as a generic OverlayState payload.
export type { JsonObject }
