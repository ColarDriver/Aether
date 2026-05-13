import type { SessionInfo } from '../../gatewayTypes.js'
import type { SessionCurrentResult, SessionListResult, SlashCommand } from '../dispatcher.js'

export const sessionCommand: SlashCommand = {
  name: '/session',
  category: 'remote',
  async execute(_args, ctx) {
    const result = await ctx.client.request<SessionCurrentResult>('session.current')
    if (!result.session_id || !result.info) {
      return { kind: 'note', level: 'warn', text: 'No active session.' }
    }
    return {
      kind: 'session',
      info: result.info,
      note: formatCurrent(result.info)
    }
  }
}

export const sessionsCommand: SlashCommand = {
  name: '/sessions',
  category: 'remote',
  async execute(args, ctx) {
    const limit = parseLimit(args[0])
    const result = await ctx.client.request<SessionListResult>('session.list', { limit })
    if (result.sessions.length === 0) {
      return { kind: 'note', level: 'info', text: 'No saved sessions.' }
    }
    const current = ctx.getSession().sessionId
    return {
      kind: 'note',
      level: 'info',
      text: renderSessionsTable(result.sessions, current)
    }
  }
}

function parseLimit(raw: string | undefined): number {
  if (!raw) {
    return 10
  }
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 10
}

/**
 * Mirrors Python `_cmd_sessions` (aether/cli/commands.py:318-352): a
 * column-aligned table with marker / id / when / turns / model / preview.
 * Rendered as plain monospace text since the chat transcript can't host a
 * Rich table directly; the layout is faithful otherwise.
 */
function renderSessionsTable(sessions: SessionInfo[], currentId: string | null): string {
  const header = ['', 'id', 'when', 'turns', 'model', 'preview']
  const rows: string[][] = [header]
  for (const session of sessions) {
    const marker = session.session_id === currentId ? '✓' : ' '
    const id = session.session_id.slice(0, 8)
    const when = formatRelativeTime(session.updated_at)
    const turns = String(session.message_count ?? 0)
    const model = `${session.provider}/${session.model}`.replace(/^\/|\/$/g, '')
    const preview = truncate((session.summary ?? '(no messages yet)').replace(/\n/g, ' '), 60)
    rows.push([marker, id, when, turns, model, preview])
  }

  // Compute column widths from data (header + rows). Last column (preview)
  // is allowed to overflow — it already wraps naturally in the terminal.
  const widths = header.map((_, col) =>
    col === header.length - 1
      ? 0
      : Math.max(...rows.map((row) => (row[col] ?? '').length))
  )

  const lines = rows.map((row, rowIdx) => {
    const cells = row.map((value, col) =>
      col === header.length - 1 ? value : (value ?? '').padEnd(widths[col] ?? 0)
    )
    if (rowIdx === 0) {
      return cells.join('  ')
    }
    return cells.join('  ')
  })

  return `Saved sessions · ${sessions.length}\n${lines.join('\n')}\nUse /resume to switch into one of these sessions.`
}

function formatCurrent(info: SessionInfo): string {
  return `${info.session_id.slice(0, 8)}  ${info.provider}/${info.model}${
    info.summary ? ' · ' + info.summary : ''
  }`
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
  if (value.length <= max) {
    return value
  }
  return value.slice(0, Math.max(0, max - 1)) + '…'
}
