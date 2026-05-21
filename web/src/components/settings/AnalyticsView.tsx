import { RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../../api/client'
import type { AnalyticsReport, AnalyticsSessionEntry } from '../../api/types'
import { useToastStore } from '../../stores/toastStore'
import { Spinner } from '../shared/Spinner'

const PERIODS = [7, 30, 90]

export function AnalyticsView() {
  const [days, setDays] = useState(30)
  const [report, setReport] = useState<AnalyticsReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const notify = useToastStore((state) => state.notify)

  const load = () => {
    setLoading(true)
    setError(null)
    api.analytics({ days, limit: 20 })
      .then(setReport)
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, 'error')
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [days])

  const peakTokens = useMemo(() => {
    return Math.max(...(report?.daily ?? []).map((entry) => entry.usage.total_tokens), 1)
  }, [report])

  return (
    <div className="settings-panel analytics-panel">
      <header className="panel-header">
        <div>
          <h2>Analytics</h2>
          <p>Local session and token usage over the selected window.</p>
        </div>
        <div className="analytics-actions">
          <select aria-label="Analytics period" value={days} onChange={(event) => setDays(Number(event.target.value))}>
            {PERIODS.map((period) => <option key={period} value={period}>{period}d</option>)}
          </select>
          <button type="button" onClick={load} disabled={loading}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </header>
      {loading ? <Spinner label="Loading analytics" /> : null}
      {error ? <div className="notice notice-error">{error}</div> : null}
      {report ? (
        <>
          <div className="analytics-summary-grid">
            <MetricCard label="Sessions" value={String(report.summary.session_count)} />
            <MetricCard label="Messages" value={String(report.summary.message_count)} />
            <MetricCard label="Tool calls" value={String(report.summary.tool_call_count)} />
            <MetricCard label="Tokens" value={formatTokens(report.summary.usage.total_tokens)} />
          </div>
          <section className="catalog-card analytics-card">
            <div className="catalog-card-header">
              <strong>Daily token usage</strong>
              <span>{report.daily.length} days with activity</span>
            </div>
            <div className="analytics-chart" aria-label="Daily token usage chart">
              {report.daily.length === 0 ? <div className="muted pad">No token usage recorded</div> : null}
              {report.daily.map((entry) => (
                <div className="analytics-bar-wrap" key={entry.day} title={entry.day + ': ' + entry.usage.total_tokens + ' tokens'}>
                  <div className="analytics-bar" style={{ height: Math.max(4, Math.round((entry.usage.total_tokens / peakTokens) * 150)) }} />
                  <span>{formatDay(entry.day)}</span>
                </div>
              ))}
            </div>
          </section>
          <section className="catalog-card analytics-card">
            <div className="catalog-card-header"><strong>Models</strong><span>{report.models.length} model groups</span></div>
            <div className="analytics-table-wrap">
              <table className="analytics-table">
                <thead><tr><th>Model</th><th>Sessions</th><th>Messages</th><th>Tokens</th></tr></thead>
                <tbody>
                  {report.models.map((entry) => (
                    <tr key={entry.provider + ':' + entry.model}>
                      <td>{entry.provider} / {entry.model}</td>
                      <td>{entry.sessions}</td>
                      <td>{entry.messages}</td>
                      <td>{formatTokens(entry.usage.total_tokens)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="catalog-card analytics-card">
            <div className="catalog-card-header"><strong>Top sessions</strong><span>{report.top_sessions.length} rows</span></div>
            <div className="analytics-session-list">
              {report.top_sessions.length === 0 ? <div className="muted pad">No sessions in this window</div> : null}
              {report.top_sessions.map((session) => <SessionUsageRow key={session.session_id} session={session} />)}
            </div>
          </section>
        </>
      ) : null}
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="analytics-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function SessionUsageRow({ session }: { session: AnalyticsSessionEntry }) {
  return (
    <div className="analytics-session-row">
      <div>
        <strong>{session.summary || session.session_id.slice(0, 8)}</strong>
        <p>{session.provider} / {session.model}</p>
      </div>
      <span>{session.messages} messages</span>
      <span>{formatTokens(session.usage.total_tokens)} tokens</span>
    </div>
  )
}

function formatTokens(value: number): string {
  if (value >= 1_000_000) return (value / 1_000_000).toFixed(1) + 'M'
  if (value >= 1_000) return (value / 1_000).toFixed(1) + 'K'
  return String(value)
}

function formatDay(value: string): string {
  const date = new Date(value + 'T00:00:00Z')
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
