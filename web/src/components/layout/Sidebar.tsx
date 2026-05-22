import { MessageSquare, Plus, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { navItems } from '../../navItems'
import type { ConsoleView, SessionInfo } from '../../api/types'
import { Button } from '../shared/Button'

type Props = {
  sessions: SessionInfo[]
  activeSessionId: string | null
  activeView: ConsoleView
  onSelectSession: (sessionId: string) => void
  onSelectView: (view: ConsoleView) => void
  onNewSession: () => void
}

export function Sidebar({ sessions, activeSessionId, activeView, onSelectSession, onSelectView, onNewSession }: Props) {
  const [query, setQuery] = useState('')
  const filteredSessions = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return sessions
    return sessions.filter((session) =>
      [session.session_id, session.summary, session.provider, session.model]
        .some((value) => String(value ?? '').toLowerCase().includes(needle)),
    )
  }, [query, sessions])
  const groupedSessions = useMemo(() => groupSessionsByRecency(filteredSessions), [filteredSessions])

  return (
    <>
      <nav className="app-rail" aria-label="Console sections">
        <div className="brand-mark">A</div>
        <div className="app-rail-section">
          {railPrimaryItems.map((item) => renderRailButton(item, activeView, onSelectView))}
        </div>
        <div className="app-rail-section app-rail-section-quiet">
          {railSecondaryItems.map((item) => renderRailButton(item, activeView, onSelectView))}
        </div>
        <div className="app-rail-spacer" />
        <div className="app-rail-section">
          {railBottomItems.map((item) => renderRailButton(item, activeView, onSelectView))}
        </div>
      </nav>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">A</div>
          <div>
            <strong>Aether</strong>
            <span>{sessions.length + ' sessions'}</span>
          </div>
        </div>
        <div className="sidebar-primary-actions">
          <button type="button" className="new-chat-button" onClick={onNewSession}>
            <Plus size={15} />
            <span>New session</span>
          </button>
        </div>
        <div className="sidebar-section-header">
          <span>Sessions</span>
          <Button title="New session" aria-label="New session quick action" onClick={onNewSession}>
            <Plus size={15} />
          </Button>
        </div>
        <label className="session-search">
          <Search size={14} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search sessions" />
        </label>
        <div className="session-list">
          {sessions.length === 0 ? <div className="muted pad">No sessions yet</div> : null}
          {sessions.length > 0 && filteredSessions.length === 0 ? <div className="muted pad">No matching sessions</div> : null}
          {groupedSessions.map((group) => (
            <section className="session-group" key={group.id} aria-label={group.label}>
              <div className="session-group-header">
                <span>{group.label}</span>
                <small>{group.sessions.length}</small>
              </div>
              {group.sessions.map((session) => (
                <button
                  type="button"
                  key={session.session_id}
                  className={session.session_id === activeSessionId ? 'session-item session-item-active' : 'session-item'}
                  onClick={() => onSelectSession(session.session_id)}
                  aria-current={session.session_id === activeSessionId ? 'page' : undefined}
                >
                  <span className="session-row-leading" aria-hidden="true">
                    <MessageSquare size={13} />
                  </span>
                  <span className="session-row-body">
                    <span className="session-title-row">
                      <span className="session-item-title">{session.summary || session.session_id.slice(0, 8)}</span>
                      <time>{formatSessionTime(session.updated_at)}</time>
                    </span>
                    <span className="session-item-meta">
                      <small>{session.model}</small>
                      {session.mode === 'plan' ? <em>plan</em> : null}
                      {session.message_count > 0 ? <small>{session.message_count} msgs</small> : null}
                    </span>
                  </span>
                </button>
              ))}
            </section>
          ))}
        </div>
      </aside>
    </>
  )
}

const railPrimaryIds = new Set<ConsoleView>(['chat', 'sessions', 'models', 'tools', 'workspace'])
const railBottomIds = new Set<ConsoleView>(['settings'])
const railPrimaryItems = navItems.filter((item) => railPrimaryIds.has(item.id))
const railSecondaryItems = navItems.filter((item) => !railPrimaryIds.has(item.id) && !railBottomIds.has(item.id))
const railBottomItems = navItems.filter((item) => railBottomIds.has(item.id))

function renderRailButton(
  item: (typeof navItems)[number],
  activeView: ConsoleView,
  onSelectView: (view: ConsoleView) => void,
) {
  const Icon = item.icon
  return (
    <button
      key={item.id}
      className={activeView === item.id ? 'nav-item nav-item-active' : 'nav-item'}
      type="button"
      title={item.label}
      aria-label={item.label}
      onClick={() => onSelectView(item.id)}
    >
      <Icon size={16} />
    </button>
  )
}

type SessionGroup = {
  id: string
  label: string
  sessions: SessionInfo[]
}

function groupSessionsByRecency(sessions: SessionInfo[]): SessionGroup[] {
  const groups: SessionGroup[] = [
    { id: 'today', label: 'Today', sessions: [] },
    { id: 'week', label: 'Last 7 days', sessions: [] },
    { id: 'older', label: 'Older', sessions: [] },
  ]
  const now = Date.now()
  const dayMs = 86_400_000
  for (const session of sessions) {
    const updatedAt = timestampMs(session.updated_at)
    const age = Math.max(0, now - updatedAt)
    if (age < dayMs) groups[0]!.sessions.push(session)
    else if (age < dayMs * 7) groups[1]!.sessions.push(session)
    else groups[2]!.sessions.push(session)
  }
  return groups.filter((group) => group.sessions.length > 0)
}

function formatSessionTime(value: number): string {
  const updatedAt = timestampMs(value)
  const diff = Math.max(0, Date.now() - updatedAt)
  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour
  if (diff < minute) return 'now'
  if (diff < hour) return Math.floor(diff / minute) + 'm'
  if (diff < day) return Math.floor(diff / hour) + 'h'
  if (diff < day * 7) return Math.floor(diff / day) + 'd'
  return new Date(updatedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function timestampMs(value: number): number {
  return value < 1_000_000_000_000 ? value * 1000 : value
}
