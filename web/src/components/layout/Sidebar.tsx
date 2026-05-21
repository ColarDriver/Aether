import { Plus, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { navItems } from '../../App'
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

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">A</div>
        <div>
          <strong>Aether</strong>
          <span>Web Console</span>
        </div>
      </div>
      <nav className="nav-list" aria-label="Console sections">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.id}
              className={activeView === item.id ? 'nav-item nav-item-active' : 'nav-item'}
              type="button"
              onClick={() => onSelectView(item.id)}
            >
              <Icon size={16} />
              <span>{item.label}</span>
            </button>
          )
        })}
      </nav>
      <div className="sidebar-section-header">
        <span>Sessions</span>
        <Button title="New session" aria-label="New session" onClick={onNewSession}>
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
        {filteredSessions.map((session) => (
          <button
            type="button"
            key={session.session_id}
            className={session.session_id === activeSessionId ? 'session-item session-item-active' : 'session-item'}
            onClick={() => onSelectSession(session.session_id)}
          >
            <span>{session.summary || session.session_id.slice(0, 8)}</span>
            <small>{session.model}</small>
          </button>
        ))}
      </div>
    </aside>
  )
}
