import { Activity, BarChart3, Brain, CircleAlert, Server, Sparkles, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type {
  AnalyticsReport,
  HealthStatus,
  ProviderRuntimeStatus,
  SkillSummary,
  StatusResponse,
} from '../../api/types'

export type ComposerInspectorKind = 'status' | 'context' | 'cost' | 'skills' | 'mcp'

type Props = {
  kind: ComposerInspectorKind
  sessionId?: string | null
  sessionSummary?: string | null
  messageCount?: number | null
  provider?: string | null
  model?: string | null
  mode?: string | null
  inputTokens?: number | null
  outputTokens?: number | null
  onClose: () => void
}

type LoadState<T> =
  | { state: 'idle' }
  | { state: 'loading' }
  | { state: 'ready'; data: T }
  | { state: 'error'; message: string }

type StatusPanelData = {
  status: StatusResponse
  health: HealthStatus
  provider: ProviderRuntimeStatus | null
}

type ContextPanelData = {
  contextWindow: number | null
  discovery: string | null
}

const TITLES: Record<ComposerInspectorKind, { title: string; subtitle: string }> = {
  status: { title: 'Status', subtitle: 'Runtime, provider, and session state' },
  context: { title: 'Context', subtitle: 'Current turn tokens and model window' },
  cost: { title: 'Cost', subtitle: 'Local usage over the last 30 days' },
  skills: { title: 'Skills', subtitle: 'Available local skill cards' },
  mcp: { title: 'MCP', subtitle: 'Integration boundary for this web console' },
}

export function ComposerInspectorPanel({
  kind,
  sessionId,
  sessionSummary,
  messageCount,
  provider,
  model,
  mode,
  inputTokens,
  outputTokens,
  onClose,
}: Props) {
  const label = TITLES[kind]

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <section className="composer-inspector-panel" aria-label={label.title + ' inspector'}>
      <header className="composer-inspector-header">
        <span className="composer-inspector-icon" aria-hidden="true">{iconForKind(kind)}</span>
        <span>
          <strong>{label.title}</strong>
          <small>{label.subtitle}</small>
        </span>
        <button type="button" className="composer-inspector-close" aria-label="Close inspector" onClick={onClose}>
          <X size={14} />
        </button>
      </header>
      <div className="composer-inspector-body">
        {kind === 'status' ? (
          <StatusInspector sessionId={sessionId} sessionSummary={sessionSummary} mode={mode} provider={provider} model={model} messageCount={messageCount} />
        ) : null}
        {kind === 'context' ? (
          <ContextInspector
            provider={provider}
            model={model}
            mode={mode}
            messageCount={messageCount}
            inputTokens={inputTokens}
            outputTokens={outputTokens}
          />
        ) : null}
        {kind === 'cost' ? <CostInspector /> : null}
        {kind === 'skills' ? <SkillsInspector /> : null}
        {kind === 'mcp' ? <McpInspector /> : null}
      </div>
    </section>
  )
}

function StatusInspector({
  sessionId,
  sessionSummary,
  mode,
  provider,
  model,
  messageCount,
}: {
  sessionId?: string | null
  sessionSummary?: string | null
  mode?: string | null
  provider?: string | null
  model?: string | null
  messageCount?: number | null
}) {
  const [result, setResult] = useState<LoadState<StatusPanelData>>({ state: 'loading' })

  useEffect(() => {
    let cancelled = false
    Promise.all([
      api.status(),
      api.health(),
      api.currentProvider().catch(() => null),
    ])
      .then(([status, health, currentProvider]) => {
        if (!cancelled) setResult({ state: 'ready', data: { status, health, provider: currentProvider } })
      })
      .catch((error) => {
        if (!cancelled) setResult({ state: 'error', message: errorMessage(error) })
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (result.state === 'loading' || result.state === 'idle') return <PanelLoading label="Loading runtime status" />
  if (result.state === 'error') return <PanelError message={result.message} />

  const runtime = result.data.health.runtime
  const currentProvider = result.data.provider
  return (
    <>
      <div className="composer-inspector-grid">
        <Metric label="Runtime" value={result.data.status.ok ? 'online' : 'degraded'} detail={result.data.status.name + ' ' + result.data.status.version} />
        <Metric label="Health" value={result.data.health.status} detail={runtime.python_version + ' / ' + runtime.platform} />
        <Metric label="Provider" value={provider || currentProvider?.provider_name || 'unknown'} detail={model || currentProvider?.model || 'No model selected'} />
        <Metric label="Session" value={sessionSummary || shortId(sessionId) || 'none'} detail={(messageCount ?? 0).toLocaleString() + ' messages / ' + (mode || 'agent') + ' mode'} />
      </div>
      <Rows
        rows={result.data.health.services.map((service) => ({
          label: service.name,
          value: service.status,
          detail: service.detail || (service.available ? 'available' : 'unavailable'),
        }))}
        empty="No service diagnostics reported."
      />
    </>
  )
}

function ContextInspector({
  provider,
  model,
  mode,
  messageCount,
  inputTokens,
  outputTokens,
}: {
  provider?: string | null
  model?: string | null
  mode?: string | null
  messageCount?: number | null
  inputTokens?: number | null
  outputTokens?: number | null
}) {
  const [result, setResult] = useState<LoadState<ContextPanelData>>({ state: 'loading' })
  const activeTokens = Math.max(0, (inputTokens ?? 0) + (outputTokens ?? 0))

  useEffect(() => {
    let cancelled = false
    if (!provider || !model) {
      setResult({ state: 'ready', data: { contextWindow: null, discovery: 'No active provider/model selected.' } })
      return () => {
        cancelled = true
      }
    }
    api.providerModels(provider)
      .then((result) => {
        const matched = result.models.find((item) => item.id === model)
        if (!cancelled) {
          setResult({
            state: 'ready',
            data: {
              contextWindow: matched?.context_window ?? null,
              discovery: result.discovery.reason || result.discovery.source || result.discovery.kind || null,
            },
          })
        }
      })
      .catch((error) => {
        if (!cancelled) setResult({ state: 'error', message: errorMessage(error) })
      })
    return () => {
      cancelled = true
    }
  }, [model, provider])

  const contextWindow = result.state === 'ready' ? result.data.contextWindow : null
  const percent = contextWindow && contextWindow > 0 ? Math.min(100, Math.round((activeTokens / contextWindow) * 100)) : null

  return (
    <>
      <div className="composer-inspector-grid">
        <Metric label="Active tokens" value={formatNumber(activeTokens)} detail={formatNumber(inputTokens ?? 0) + ' in / ' + formatNumber(outputTokens ?? 0) + ' out'} />
        <Metric label="Context window" value={contextWindow ? formatNumber(contextWindow) : 'unknown'} detail={percent === null ? 'Provider did not report a window' : percent + '% of reported window'} />
        <Metric label="Transcript" value={formatNumber(messageCount ?? 0)} detail="messages in current session metadata" />
        <Metric label="Mode" value={mode || 'agent'} detail={provider && model ? provider + ' / ' + model : 'No model selected'} />
      </div>
      {result.state === 'loading' || result.state === 'idle' ? <PanelLoading label="Loading model context metadata" /> : null}
      {result.state === 'error' ? <PanelError message={result.message} /> : null}
      {result.state === 'ready' ? (
        <p className="composer-inspector-note">
          Context usage is estimated from the active run token stream and session metadata. It is not a full reconstructed prompt budget yet.
          {result.data.discovery ? ' Discovery: ' + result.data.discovery : ''}
        </p>
      ) : null}
    </>
  )
}

function CostInspector() {
  const [result, setResult] = useState<LoadState<AnalyticsReport>>({ state: 'loading' })

  useEffect(() => {
    let cancelled = false
    api.analytics({ days: 30, limit: 6 })
      .then((data) => {
        if (!cancelled) setResult({ state: 'ready', data })
      })
      .catch((error) => {
        if (!cancelled) setResult({ state: 'error', message: errorMessage(error) })
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (result.state === 'loading' || result.state === 'idle') return <PanelLoading label="Loading usage analytics" />
  if (result.state === 'error') return <PanelError message={result.message} />

  const summary = result.data.summary
  return (
    <>
      <div className="composer-inspector-grid">
        <Metric label="Messages" value={formatNumber(summary.message_count)} detail={formatNumber(summary.assistant_message_count) + ' assistant messages'} />
        <Metric label="Tool calls" value={formatNumber(summary.tool_call_count)} detail="recorded in transcripts" />
        <Metric label="Tokens" value={formatNumber(summary.usage.total_tokens)} detail={formatNumber(summary.usage.input_tokens) + ' in / ' + formatNumber(summary.usage.output_tokens) + ' out'} />
        <Metric label="Sessions" value={formatNumber(summary.session_count)} detail={'last ' + result.data.days + ' days'} />
      </div>
      <Rows
        rows={result.data.models.slice(0, 6).map((entry) => ({
          label: entry.model,
          value: formatNumber(entry.usage.total_tokens) + ' tokens',
          detail: entry.provider + ' / ' + formatNumber(entry.messages) + ' messages',
        }))}
        empty="No model usage recorded yet."
      />
    </>
  )
}

function SkillsInspector() {
  const [result, setResult] = useState<LoadState<SkillSummary[]>>({ state: 'loading' })

  useEffect(() => {
    let cancelled = false
    api.skills()
      .then((data) => {
        if (!cancelled) setResult({ state: 'ready', data: data.skills })
      })
      .catch((error) => {
        if (!cancelled) setResult({ state: 'error', message: errorMessage(error) })
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (result.state === 'loading' || result.state === 'idle') return <PanelLoading label="Loading skills" />
  if (result.state === 'error') return <PanelError message={result.message} />

  return (
    <div className="composer-inspector-list">
      {result.data.slice(0, 8).map((skill) => (
        <article key={skill.name} className="composer-inspector-list-item">
          <strong>{skill.name}</strong>
          <p>{skill.description || skill.when_to_use || 'No description provided.'}</p>
          <small>{skill.source.path || skill.source.source}</small>
        </article>
      ))}
      {result.data.length === 0 ? <p className="composer-inspector-note">No skills are installed for this workspace.</p> : null}
    </div>
  )
}

function McpInspector() {
  return (
    <div className="composer-inspector-empty">
      <CircleAlert size={18} />
      <div>
        <strong>MCP management is not wired into Aether web yet.</strong>
        <p>
          This panel is intentionally explicit instead of inventing server data. A complete MCP view needs backend routes for configured
          servers, connection state, tools, resources, and credential status.
        </p>
      </div>
    </div>
  )
}

function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="composer-inspector-metric">
      <small>{label}</small>
      <strong>{value}</strong>
      {detail ? <span>{detail}</span> : null}
    </div>
  )
}

function Rows({ rows, empty }: { rows: Array<{ label: string; value: string; detail?: string | null }>; empty: string }) {
  if (rows.length === 0) return <p className="composer-inspector-note">{empty}</p>
  return (
    <div className="composer-inspector-rows">
      {rows.map((row) => (
        <div key={row.label + row.value} className="composer-inspector-row">
          <span>
            <strong>{row.label}</strong>
            {row.detail ? <small>{row.detail}</small> : null}
          </span>
          <em>{row.value}</em>
        </div>
      ))}
    </div>
  )
}

function PanelLoading({ label }: { label: string }) {
  return <p className="composer-inspector-note">{label}...</p>
}

function PanelError({ message }: { message: string }) {
  return <p className="composer-inspector-error">{message}</p>
}

function iconForKind(kind: ComposerInspectorKind) {
  if (kind === 'status') return <Activity size={16} />
  if (kind === 'context') return <Brain size={16} />
  if (kind === 'cost') return <BarChart3 size={16} />
  if (kind === 'skills') return <Sparkles size={16} />
  return <Server size={16} />
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function shortId(sessionId?: string | null) {
  return sessionId ? sessionId.slice(0, 8) : null
}

function formatNumber(value: number) {
  return value.toLocaleString()
}
