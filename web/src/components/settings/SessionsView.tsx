import { Download, FileUp, Pencil, RefreshCw, Search, Trash2 } from "lucide-react"
import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react"
import { api } from "../../api/client"
import type { SessionInfo, TranscriptMessage } from "../../api/types"
import { normalizeTranscript } from "../../chat-rendering"
import { useAppStore } from "../../stores/appStore"
import { useChatStore } from "../../stores/chatStore"
import { useSessionStore } from "../../stores/sessionStore"
import { useTaskStore } from "../../stores/taskStore"
import { useToastStore } from "../../stores/toastStore"
import { ChatTimeline } from "../chat/ChatTimeline"
import { ConfirmDialog } from "../shared/ConfirmDialog"
import { Spinner } from "../shared/Spinner"

type SessionDetail = {
  session_id: string
  info: SessionInfo
  messages: TranscriptMessage[]
}

type PendingDeleteSession = {
  sessionId: string
  title: string
}

export function SessionsView() {
  const { sessions, activeSessionId, setActiveSession, deleteSession, importSession, loadSessions, renameSession } = useSessionStore()
  const setActiveView = useAppStore((state) => state.setActiveView)
  const notify = useToastStore((state) => state.notify)
  const clearChatSession = useChatStore((state) => state.clearSession)
  const clearSessionTasks = useTaskStore((state) => state.clearSessionTasks)
  const importFileRef = useRef<HTMLInputElement | null>(null)
  const [query, setQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SessionInfo[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(activeSessionId ?? sessions[0]?.session_id ?? null)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [searching, setSearching] = useState(false)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<PendingDeleteSession | null>(null)
  const [renameOpen, setRenameOpen] = useState(false)
  const [renameDraft, setRenameDraft] = useState("")

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

  const transcriptBlocks = useMemo(() => {
    if (!detail) return []
    return normalizeTranscript(detail.session_id, detail.messages)
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

  const beginRenameSelected = () => {
    if (!selectedId) return
    setRenameDraft(selectedId)
    setRenameOpen(true)
  }

  const confirmRenameSelected = () => {
    if (!selectedId || saving) return
    const oldId = selectedId
    const nextId = renameDraft.trim()
    if (!nextId || nextId === oldId) {
      setRenameOpen(false)
      return
    }
    setSaving(true)
    renameSession(oldId, nextId)
      .then((info) => {
        clearChatSession(oldId)
        clearSessionTasks(oldId)
        setSelectedId(info.session_id)
        setDetail(null)
        setSearchResults(null)
        setRenameOpen(false)
        notify("Renamed session to " + info.session_id.slice(0, 8), "success")
        void loadSessions()
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => setSaving(false))
  }

  const exportSelected = () => {
    if (!selectedId || saving) return
    setSaving(true)
    api.exportSession(selectedId)
      .then((result) => {
        downloadSessionJson(result.session_id, result.data)
        notify("Exported " + result.session_id.slice(0, 8), "success")
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => setSaving(false))
  }

  const importFromFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ""
    if (!file || saving) return
    setSaving(true)
    file.text()
      .then((text) => JSON.parse(text) as Record<string, unknown>)
      .then((data) => importSession({ data }))
      .then((result) => {
        setSearchResults(null)
        setSelectedId(result.info.session_id)
        setDetail({
          session_id: result.info.session_id,
          info: result.info,
          messages: result.messages,
        })
        notify("Imported " + result.info.session_id.slice(0, 8), "success")
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
    setDeleteTarget({ sessionId: selectedId, title: detail?.info.summary || selectedId.slice(0, 8) })
  }

  const confirmDeleteSelected = () => {
    if (!deleteTarget) return
    const id = deleteTarget.sessionId
    setDeleteTarget(null)
    setSaving(true)
    deleteSession(id)
      .then(() => {
        clearChatSession(id)
        clearSessionTasks(id)
        setSearchResults((results) => results ? results.filter((session) => session.session_id !== id) : results)
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
        <button type="button" onClick={() => importFileRef.current?.click()} disabled={saving}>
          <FileUp size={14} /> Import JSON
        </button>
        <input
          ref={importFileRef}
          type="file"
          accept="application/json,.json"
          className="sr-only"
          aria-label="Import session JSON"
          onChange={importFromFile}
        />
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
                  <button type="button" onClick={beginRenameSelected} disabled={saving}>
                    <Pencil size={14} /> Rename ID
                  </button>
                  <button type="button" onClick={exportSelected} disabled={saving}>
                    <Download size={14} /> Export JSON
                  </button>
                  <button type="button" onClick={deleteSelected} disabled={saving} className="danger-action" aria-label="Delete session">
                    <Trash2 size={14} /> Delete
                  </button>
                </div>
              </div>
              {renameOpen ? (
                <form
                  className="session-rename-form"
                  onSubmit={(event) => {
                    event.preventDefault()
                    confirmRenameSelected()
                  }}
                >
                  <label>
                    <span>New session ID</span>
                    <input
                      aria-label="New session ID"
                      value={renameDraft}
                      onChange={(event) => setRenameDraft(event.target.value)}
                      disabled={saving}
                    />
                  </label>
                  <div className="session-rename-actions">
                    <button type="button" onClick={() => setRenameOpen(false)} disabled={saving}>Cancel</button>
                    <button type="submit" disabled={saving || !renameDraft.trim()}>Rename</button>
                  </div>
                </form>
              ) : null}
              <div className="info-grid compact-grid session-metadata">
                <div className="info-row"><span>ID</span><strong>{detail.session_id}</strong></div>
                <div className="info-row"><span>Messages</span><strong>{detail.info.message_count}</strong></div>
                <div className="info-row"><span>Created</span><strong>{formatTimestamp(detail.info.created_at)}</strong></div>
                <div className="info-row"><span>Updated</span><strong>{formatTimestamp(detail.info.updated_at)}</strong></div>
              </div>
              <div className="transcript-preview">
                <ChatTimeline blocks={transcriptBlocks} />
              </div>
            </>
          ) : null}
        </section>
      </div>
      {deleteTarget ? (
        <ConfirmDialog
          title="Delete session"
          description={'Delete session "' + deleteTarget.title + '"? This removes its conversation context.'}
          confirmLabel="Delete"
          cancelLabel="Cancel"
          onCancel={() => setDeleteTarget(null)}
          onConfirm={confirmDeleteSelected}
        />
      ) : null}
    </div>
  )
}

function formatTimestamp(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "-"
  const millis = value > 10_000_000_000 ? value : value * 1000
  return new Date(millis).toLocaleString()
}

function downloadSessionJson(sessionId: string, data: Record<string, unknown>) {
  const text = JSON.stringify(data, null, 2)
  const filename = sessionId + ".aether-session.json"
  if (typeof URL.createObjectURL !== "function") {
    return
  }
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }))
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  anchor.rel = "noopener"
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}
