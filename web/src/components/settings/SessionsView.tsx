import { RefreshCw, Search, Trash2 } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { api } from "../../api/client"
import type { SessionInfo, TranscriptMessage } from "../../api/types"
import { useAppStore } from "../../stores/appStore"
import { transcriptToChatState } from "../../stores/chatStore"
import { useSessionStore } from "../../stores/sessionStore"
import { useToastStore } from "../../stores/toastStore"
import { MessageList } from "../chat/MessageList"
import { Spinner } from "../shared/Spinner"

type SessionDetail = {
  session_id: string
  info: SessionInfo
  messages: TranscriptMessage[]
}

export function SessionsView() {
  const { sessions, activeSessionId, setActiveSession, loadSessions } = useSessionStore()
  const setActiveView = useAppStore((state) => state.setActiveView)
  const notify = useToastStore((state) => state.notify)
  const [query, setQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SessionInfo[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(activeSessionId ?? sessions[0]?.session_id ?? null)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [searching, setSearching] = useState(false)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (sessions.length === 0) void loadSessions()
  }, [loadSessions, sessions.length])

  useEffect(() => {
    const needle = query.trim()
    if (!needle) {
      setSearchResults(null)
      return
    }
    let cancelled = false
    setSearching(true)
    api.searchSessions(needle)
      .then(({ sessions: results }) => {
        if (!cancelled) setSearchResults(results)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => {
        if (!cancelled) setSearching(false)
      })
    return () => {
      cancelled = true
    }
  }, [notify, query])

  const displayedSessions = searchResults ?? sessions

  useEffect(() => {
    if (!selectedId && displayedSessions[0]) setSelectedId(displayedSessions[0].session_id)
  }, [displayedSessions, selectedId])

  useEffect(() => {
    if (!selectedId) {
      setDetail(null)
      return
    }
    let cancelled = false
    setLoadingDetail(true)
    setError(null)
    api.sessionDetail(selectedId)
      .then((result) => {
        if (!cancelled) setDetail(result)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        setDetail(null)
        notify(message, "error")
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false)
      })
    return () => {
      cancelled = true
    }
  }, [notify, selectedId])

  const messages = useMemo(() => {
    if (!detail) return []
    return transcriptToChatState(detail.session_id, detail.messages).messages
  }, [detail])

  const resumeSelected = () => {
    if (!selectedId) return
    setSaving(true)
    api.resumeSession(selectedId)
      .then((result) => {
        setActiveSession(result.session_id)
        setActiveView("chat")
        notify("Resumed " + result.session_id.slice(0, 8), "success")
        void loadSessions()
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => setSaving(false))
  }

  const deleteSelected = () => {
    if (!selectedId) return
    const id = selectedId
    setSaving(true)
    api.deleteSession(id)
      .then(() => {
        if (activeSessionId === id) setActiveSession(null)
        setSelectedId(null)
        setDetail(null)
        notify("Deleted " + id.slice(0, 8), "success")
        void loadSessions()
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => setSaving(false))
  }

  return (
    <div className="settings-panel sessions-panel">
      <header className="panel-header">
        <div>
          <h2>Sessions</h2>
          <p>Search, inspect, resume, and delete local conversation records.</p>
        </div>
        <button type="button" onClick={() => void loadSessions()} disabled={searching || loadingDetail}>
          <RefreshCw size={14} /> Refresh
        </button>
      </header>
      {error ? <div className="notice notice-error">{error}</div> : null}
      <div className="session-manager">
        <aside className="session-manager-list">
          <label className="session-search session-search-wide">
            <Search size={14} />
            <input
              aria-label="Search session records"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search summaries, ids, providers, or models"
            />
          </label>
          {searching ? <Spinner label="Searching sessions" /> : null}
          <div className="session-result-list">
            {displayedSessions.length === 0 ? <div className="muted pad">No sessions found</div> : null}
            {displayedSessions.map((session) => (
              <button
                type="button"
                key={session.session_id}
                className={session.session_id === selectedId ? "session-result session-result-active" : "session-result"}
                onClick={() => setSelectedId(session.session_id)}
              >
                <span>{session.summary || session.session_id.slice(0, 8)}</span>
                <small>{session.provider} / {session.model}</small>
              </button>
            ))}
          </div>
        </aside>
        <section className="session-detail-panel">
          {loadingDetail ? <Spinner label="Loading session" /> : null}
          {!loadingDetail && !detail ? <div className="empty-chat">Select a session to inspect.</div> : null}
          {detail ? (
            <>
              <div className="session-detail-header">
                <div>
                  <h3>{detail.info.summary || detail.session_id}</h3>
                  <p>{detail.info.provider} / {detail.info.model}</p>
                </div>
                <div className="session-detail-actions">
                  <button type="button" onClick={resumeSelected} disabled={saving}>
                    Resume session
                  </button>
                  <button type="button" onClick={deleteSelected} disabled={saving} className="danger-action" aria-label="Delete session">
                    <Trash2 size={14} /> Delete
                  </button>
                </div>
              </div>
              <div className="info-grid compact-grid session-metadata">
                <div className="info-row"><span>ID</span><strong>{detail.session_id}</strong></div>
                <div className="info-row"><span>Messages</span><strong>{detail.info.message_count}</strong></div>
                <div className="info-row"><span>Created</span><strong>{formatTimestamp(detail.info.created_at)}</strong></div>
                <div className="info-row"><span>Updated</span><strong>{formatTimestamp(detail.info.updated_at)}</strong></div>
              </div>
              <div className="transcript-preview">
                <MessageList messages={messages} />
              </div>
            </>
          ) : null}
        </section>
      </div>
    </div>
  )
}

function formatTimestamp(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "-"
  const millis = value > 10_000_000_000 ? value : value * 1000
  return new Date(millis).toLocaleString()
}
