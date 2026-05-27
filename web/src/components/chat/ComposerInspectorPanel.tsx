import { Activity, BarChart3, Brain, RefreshCw, Server, Sparkles, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { TokenUsage } from '../../chat-rendering'
import { tokenUsageBreakdown, tokenUsageFromRecord, tokenUsageTotal } from '../../chat-rendering'
import type {
  AnalyticsReport,
  ContextStatus,
  HealthStatus,
  McpConfigList,
  McpConfiguredServer,
  McpResourceList,
  McpResourceReadResult,
  McpResourceSummary,
  McpStatus,
  ProviderPreflightStatus,
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
  tokens?: TokenUsage | null
  runMetadata?: Record<string, unknown> | null
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
  preflight: ProviderPreflightStatus | null
}

type ContextPanelData = {
  contextWindow: number | null
  discovery: string | null
  status: ContextStatus | null
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
  tokens,
  runMetadata,
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
            sessionId={sessionId}
            provider={provider}
            model={model}
            mode={mode}
            messageCount={messageCount}
            inputTokens={inputTokens}
            outputTokens={outputTokens}
            tokens={tokens}
            runMetadata={runMetadata}
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
      api.providerPreflight({
        provider: provider || undefined,
        model: model || undefined,
      }).catch(() => null),
    ])
      .then(([status, health, currentProvider, preflight]) => {
        if (!cancelled) setResult({ state: 'ready', data: { status, health, provider: currentProvider, preflight } })
      })
      .catch((error) => {
        if (!cancelled) setResult({ state: 'error', message: errorMessage(error) })
      })
    return () => {
      cancelled = true
    }
  }, [model, provider])

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
        <Metric label="Preflight" value={preflightStatus(result.data.preflight)} detail={preflightDetail(result.data.preflight)} />
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
  sessionId,
  provider,
  model,
  mode,
  messageCount,
  inputTokens,
  outputTokens,
  tokens,
  runMetadata,
}: {
  sessionId?: string | null
  provider?: string | null
  model?: string | null
  mode?: string | null
  messageCount?: number | null
  inputTokens?: number | null
  outputTokens?: number | null
  tokens?: TokenUsage | null
  runMetadata?: Record<string, unknown> | null
}) {
  const [result, setResult] = useState<LoadState<ContextPanelData>>({ state: 'loading' })
  const [focus, setFocus] = useState('')
  const [compressing, setCompressing] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const metadataUsage = tokenUsageFromRecord(recordFromUnknown(runMetadata?.usage))
  const fallbackTokens = Math.max(0, (inputTokens ?? 0) + (outputTokens ?? 0))
  const activeTokens = tokenUsageTotal(tokens) || tokenUsageTotal(metadataUsage) || fallbackTokens
  const activeBreakdown = tokenUsageBreakdown(tokens ?? metadataUsage)
  const activeDetail = activeBreakdown.length > 0
    ? activeBreakdown.join(' / ')
    : formatNumber(inputTokens ?? 0) + ' in / ' + formatNumber(outputTokens ?? 0) + ' out'

  const loadContext = useCallback(() => {
    let cancelled = false
    setResult({ state: 'loading' })
    setActionError(null)

    const modelRequest = provider && model
      ? api.providerModels(provider).then((modelResult) => {
        const matched = modelResult.models.find((item) => item.id === model)
        return {
          contextWindow: matched?.context_window ?? null,
          discovery: modelResult.discovery.reason || modelResult.discovery.source || modelResult.discovery.kind || null,
        }
      })
      : Promise.resolve({ contextWindow: null, discovery: provider || model ? 'No matching provider/model selected.' : 'No active provider/model selected.' })

    const statusRequest = sessionId
      ? api.contextStatus(sessionId).catch(() => null)
      : Promise.resolve(null)

    Promise.all([modelRequest, statusRequest])
      .then(([modelData, status]) => {
        if (!cancelled) {
          setResult({
            state: 'ready',
            data: {
          contextWindow: modelData.contextWindow ?? status?.context_window ?? null,
              discovery: modelData.discovery,
              status,
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
  }, [model, provider, sessionId])

  useEffect(() => loadContext(), [loadContext])

  const compressContext = () => {
    if (!sessionId || compressing) return
    setCompressing(true)
    setActionError(null)
    api.compressContext(sessionId, { focus: focus.trim() || null, force: true })
      .then((status) => {
        setResult((current) => {
          if (current.state !== 'ready') {
            return { state: 'ready', data: { contextWindow: null, discovery: null, status } }
          }
          return { state: 'ready', data: { ...current.data, status } }
        })
      })
      .catch((error) => setActionError(errorMessage(error)))
      .finally(() => setCompressing(false))
  }

  const contextStatus = result.state === 'ready' ? result.data.status : null
  const contextWindow = result.state === 'ready' ? result.data.contextWindow : null
  const serviceTokens = contextStatus?.token_estimate ?? 0
  const promptTokens = contextStatus?.prompt_tokens ?? serviceTokens
  const displayTokens = activeTokens || promptTokens
  const percent = contextWindow && contextWindow > 0 ? Math.min(100, Math.round((displayTokens / contextWindow) * 100)) : null
  const metadataRows = contextRowsFromRunMetadata(runMetadata)
  const serviceRows = contextRowsFromStatus(contextStatus)
  const compressionRows = mergeRows(serviceRows, metadataRows)

  return (
    <>
      <div className="composer-inspector-grid">
        <Metric label="Active tokens" value={formatNumber(displayTokens)} detail={activeTokens ? activeDetail : 'estimated from session transcript'} />
        <Metric label="Context window" value={contextWindow ? formatNumber(contextWindow) : 'unknown'} detail={percent === null ? 'Provider did not report a window' : percent + '% of reported window'} />
        <Metric label="Transcript" value={formatNumber(contextStatus?.message_count ?? messageCount ?? 0)} detail="messages in current session" />
        <Metric label="Pressure" value={contextStatus?.pressure_level ?? 'unknown'} detail={contextStatus?.next_action ? 'next action: ' + contextStatus.next_action : (mode || 'agent') + ' mode'} />
      </div>
      <div className="composer-inspector-actions" aria-label="Context controls">
        <label>
          <span>Focus</span>
          <input
            value={focus}
            onChange={(event) => setFocus(event.target.value)}
            placeholder="optional compression focus"
            disabled={!sessionId || compressing}
          />
        </label>
        <button type="button" onClick={loadContext} disabled={compressing}>
          <RefreshCw size={13} />
          <span>Refresh</span>
        </button>
        <button type="button" className="composer-inspector-primary" onClick={compressContext} disabled={!sessionId || compressing}>
          <Brain size={13} />
          <span>{compressing ? 'Compressing' : 'Compress context'}</span>
        </button>
      </div>
      {result.state === 'loading' || result.state === 'idle' ? <PanelLoading label="Loading model context metadata" /> : null}
      {result.state === 'error' ? <PanelError message={result.message} /> : null}
      {actionError ? <PanelError message={actionError} /> : null}
      <Rows
        rows={compressionRows}
        empty="No context compression has been recorded for this session."
      />
      {result.state === 'ready' ? (
        <p className="composer-inspector-note">
          Active usage comes from the live token stream, latest run metadata, or session token estimate. Context window comes from the provider catalog.
          {result.data.discovery ? ' Discovery: ' + result.data.discovery : ''}
        </p>
      ) : null}
    </>
  )
}

function contextRowsFromStatus(status: ContextStatus | null): Array<{ label: string; value: string; detail?: string | null }> {
  if (!status) return []
  const rows: Array<{ label: string; value: string; detail?: string | null }> = []
  rows.push({
    label: 'Context service',
    value: status.context_engine || 'default',
    detail: status.status ? status.status.replace(/_/g, ' ') : 'ready',
  })
  rows.push({
    label: 'Session estimate',
    value: formatNumber(status.token_estimate),
    detail: formatNumber(status.message_count) + ' messages',
  })
  if (status.breakdown?.length) {
    for (const row of status.breakdown) {
      rows.push({
        label: row.label,
        value: formatNumber(row.tokens),
        detail: row.detail ?? null,
      })
    }
  }
  rows.push({
    label: 'Compressions',
    value: formatNumber(status.compression_count ?? 0),
    detail: status.status ? status.status.replace(/_/g, ' ') : null,
  })
  const last = recordFromUnknown(status.last_compression)
  if (Object.keys(last).length > 0) {
    const sourceTokens = numberValue(last.source_tokens)
    const resultTokens = numberValue(last.result_tokens)
    if (sourceTokens !== null || resultTokens !== null) {
      const saved = sourceTokens !== null && resultTokens !== null ? Math.max(0, sourceTokens - resultTokens) : null
      rows.push({
        label: 'Manual compression tokens',
        value: sourceTokens !== null && resultTokens !== null ? formatNumber(sourceTokens) + ' -> ' + formatNumber(resultTokens) : formatNumber(sourceTokens ?? resultTokens ?? 0),
        detail: saved !== null ? formatNumber(saved) + ' tokens freed' : null,
      })
    }
    const sourceMessages = numberValue(last.source_message_count)
    const resultMessages = numberValue(last.result_message_count)
    if (sourceMessages !== null || resultMessages !== null) {
      rows.push({
        label: 'Manual compression messages',
        value: sourceMessages !== null && resultMessages !== null ? formatNumber(sourceMessages) + ' -> ' + formatNumber(resultMessages) : formatNumber(sourceMessages ?? resultMessages ?? 0),
        detail: stringValue(last.reason) || stringValue(last.error) || null,
      })
    }
  }
  if (status.error) {
    rows.push({ label: 'Compression error', value: status.error, detail: null })
  }
  return rows
}

function mergeRows(...groups: Array<Array<{ label: string; value: string; detail?: string | null }>>): Array<{ label: string; value: string; detail?: string | null }> {
  const seen = new Set<string>()
  const rows: Array<{ label: string; value: string; detail?: string | null }> = []
  for (const group of groups) {
    for (const row of group) {
      const key = row.label + '\n' + row.value
      if (seen.has(key)) continue
      seen.add(key)
      rows.push(row)
    }
  }
  return rows
}

function contextRowsFromRunMetadata(metadata: Record<string, unknown> | null | undefined): Array<{ label: string; value: string; detail?: string | null }> {
  const contextEngine = recordFromUnknown(metadata?.context_engine)
  const compression = recordFromUnknown(contextEngine.compression)
  const engine = recordFromUnknown(compression.engine)
  const turn = recordFromUnknown(metadata?.turn)
  const lastResult = recordFromUnknown(turn.compaction_last_result)
  const compaction = recordFromUnknown(metadata?.compaction)
  const lineage = recordFromUnknown(metadata?.compression_lineage)

  const rows: Array<{ label: string; value: string; detail?: string | null }> = []
  const engineName = stringValue(contextEngine.name)
  const compressionStatus = stringValue(compression.status) || stringValue(lastResult.status)
  const trigger = stringValue(compression.trigger_reason) || stringValue(lastResult.trigger_reason) || stringValue(contextEngine.last_trigger_reason)
  if (engineName || compressionStatus || trigger) {
    rows.push({
      label: 'Context engine',
      value: engineName || 'default',
      detail: [compressionStatus ? compressionStatus.replace(/_/g, ' ') : null, trigger ? 'trigger ' + trigger.replace(/_/g, ' ') : null].filter(Boolean).join(' / '),
    })
  }

  const sourceTokens = numberValue(compression.source_tokens) ?? numberValue(engine.tokens_before) ?? numberValue(lastResult.tokens_before)
  const resultTokens = numberValue(compression.result_tokens) ?? numberValue(engine.tokens_after) ?? numberValue(lastResult.tokens_after)
  if (sourceTokens !== null || resultTokens !== null) {
    const saved = sourceTokens !== null && resultTokens !== null ? Math.max(0, sourceTokens - resultTokens) : null
    rows.push({
      label: 'Compression tokens',
      value: sourceTokens !== null && resultTokens !== null ? formatNumber(sourceTokens) + ' -> ' + formatNumber(resultTokens) : formatNumber(sourceTokens ?? resultTokens ?? 0),
      detail: saved !== null ? formatNumber(saved) + ' tokens freed' : null,
    })
  }

  const sourceMessages = numberValue(compression.source_message_count)
  const resultMessages = numberValue(compression.result_message_count)
  if (sourceMessages !== null || resultMessages !== null) {
    rows.push({
      label: 'Compression messages',
      value: sourceMessages !== null && resultMessages !== null ? formatNumber(sourceMessages) + ' -> ' + formatNumber(resultMessages) : formatNumber(sourceMessages ?? resultMessages ?? 0),
      detail: stringValue(compression.reason) || stringValue(compression.error) || null,
    })
  }

  const tiers = arrayOfStrings(engine.tiers_run).length > 0 ? arrayOfStrings(engine.tiers_run) : arrayOfStrings(lastResult.tiers_run)
  const counters = compactionCounters(compaction)
  if (tiers.length > 0 || counters.length > 0) {
    rows.push({
      label: 'Compaction tiers',
      value: tiers.length > 0 ? tiers.join(', ') : counters.join(', '),
      detail: tiers.length > 0 && counters.length > 0 ? counters.join(', ') : null,
    })
  }

  const generation = numberValue(lineage.generation)
  if (generation !== null) {
    rows.push({
      label: 'Compression lineage',
      value: 'generation ' + formatNumber(generation),
      detail: stringValue(lineage.trigger_reason) || null,
    })
  }

  return rows
}

function compactionCounters(compaction: Record<string, unknown>): string[] {
  return [
    counterLabel(compaction.tier1_spilled_count, 'spill'),
    counterLabel(compaction.tier2_snipped_count, 'snip'),
    counterLabel(compaction.tier3_cleared_count, 'micro'),
    counterLabel(compaction.tier4_collapse_segments, 'collapse'),
    counterLabel(compaction.tier5_summaries_generated, 'summary'),
  ].filter((item): item is string => Boolean(item))
}

function counterLabel(value: unknown, label: string): string | null {
  const number = numberValue(value)
  return number && number > 0 ? formatNumber(number) + ' ' + label : null
}

function recordFromUnknown(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string' && item.length > 0) : []
}

function stringValue(value: unknown): string {
  return typeof value === 'string' && value.trim() ? value.trim() : ''
}

function numberValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
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

type McpPanelData = {
  status: McpStatus
  config: McpConfigList
  resources: McpResourceList
}

function McpInspector() {
  const [result, setResult] = useState<LoadState<McpPanelData>>({ state: 'loading' })
  const [resourceRead, setResourceRead] = useState<LoadState<McpResourceReadResult> | null>(null)
  const [saving, setSaving] = useState(false)
  const [managementMessage, setManagementMessage] = useState<string | null>(null)
  const [managementError, setManagementError] = useState<string | null>(null)
  const [serverForm, setServerForm] = useState({
    name: '',
    kind: 'stdio',
    command: '',
    args: '',
    url: '',
    transport: 'http',
    env: '',
    headers: '',
    enabled: true,
  })

  const loadMcp = useCallback(() => {
    let cancelled = false
    setResult((current) => current.state === 'ready' ? current : { state: 'loading' })
    Promise.all([api.mcpStatus(), api.mcpConfig(), api.mcpResources()])
      .then(([status, config, resources]) => {
        if (!cancelled) setResult({ state: 'ready', data: { status, config, resources } })
      })
      .catch((error) => {
        if (!cancelled) setResult({ state: 'error', message: errorMessage(error) })
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => loadMcp(), [loadMcp])

  if (result.state === 'loading' || result.state === 'idle') return <PanelLoading label="Loading MCP integration status" />
  if (result.state === 'error') return <PanelError message={result.message} />

  const readResource = (resource: McpResourceSummary) => {
    setResourceRead({ state: 'loading' })
    api.mcpResourceRead(resource.server, resource.uri)
      .then((next) => setResourceRead({ state: 'ready', data: next }))
      .catch((error) => setResourceRead({ state: 'error', message: errorMessage(error) }))
  }

  const data = result.data.status
  const resources = result.data.resources
  const importedTools = data.imported_tools ?? []
  const servers = data.servers ?? []
  const resourceItems = resources.resources ?? []
  return (
    <>
      <div className="composer-inspector-grid">
        <Metric label="Status" value={data.status.replace(/_/g, ' ')} detail={data.enabled ? 'MCP tools are exposed' : 'No MCP servers configured'} />
        <Metric label="Servers" value={formatNumber(servers.length)} detail="discovered from tool namespaces" />
        <Metric label="Configured" value={formatNumber(result.data.config.servers.length)} detail={result.data.config.exists ? result.data.config.config_path : 'no config file'} />
        <Metric label="Tools" value={formatNumber(importedTools.length)} detail="mcp__server__tool entries" />
        <Metric label="Resources" value={formatNumber(resourceItems.length)} detail={resources.status.replace(/_/g, ' ')} />
      </div>
      <p className="composer-inspector-note">{data.message}</p>
      <p className="composer-inspector-note">{resources.message}</p>
      <div className="composer-inspector-actions">
        <button
          type="button"
          className="composer-inspector-inline-action"
          onClick={() => {
            setManagementMessage(null)
            setManagementError(null)
            api.refreshMcp()
              .then((status) => {
                setManagementMessage(status.message)
                loadMcp()
              })
              .catch((error) => setManagementError(errorMessage(error)))
          }}
        >
          Refresh MCP runtime
        </button>
      </div>
      <Rows
        rows={servers.map((server) => ({
          label: server.name,
          value: server.status,
          detail: formatNumber(server.tools_count) + ' tools / ' + formatNumber(server.resources_count) + ' resources / credential ' + server.credential_status,
        }))}
        empty="No MCP servers are configured for this runtime."
      />
      <McpConfigManager
        config={result.data.config}
        form={serverForm}
        setForm={setServerForm}
        saving={saving}
        message={managementMessage}
        error={managementError}
        onDelete={(server) => {
          setSaving(true)
          setManagementMessage(null)
          setManagementError(null)
          api.deleteMcpServer(server.name)
            .then((next) => {
              setManagementMessage(next.message)
              loadMcp()
            })
            .catch((error) => setManagementError(errorMessage(error)))
            .finally(() => setSaving(false))
        }}
        onSave={() => {
          setSaving(true)
          setManagementMessage(null)
          setManagementError(null)
          const isRemote = serverForm.kind === 'remote'
          api.upsertMcpServer({
            name: serverForm.name,
            enabled: serverForm.enabled,
            ...(isRemote ? {
              url: serverForm.url,
              transport: serverForm.transport,
              headers: parseKeyValueLines(serverForm.headers),
            } : {
              command: serverForm.command,
              args: parseListLines(serverForm.args),
              env: parseKeyValueLines(serverForm.env),
              transport: 'stdio',
            }),
          })
            .then((next) => {
              setManagementMessage(next.message)
              setServerForm({
                name: '',
                kind: serverForm.kind,
                command: '',
                args: '',
                url: '',
                transport: serverForm.transport,
                env: '',
                headers: '',
                enabled: true,
              })
              loadMcp()
            })
            .catch((error) => setManagementError(errorMessage(error)))
            .finally(() => setSaving(false))
        }}
      />
      {resourceItems.length > 0 ? (
        <div className="composer-inspector-list" aria-label="MCP resources">
          {resourceItems.slice(0, 8).map((resource) => (
            <article key={resource.server + resource.uri} className="composer-inspector-list-item">
              <strong>{resource.name || resource.uri}</strong>
              <p>{resource.description || resource.mime_type || 'No description provided.'}</p>
              <small>{resource.server} / {resource.uri}</small>
              <button type="button" className="composer-inspector-inline-action" onClick={() => readResource(resource)}>
                Read resource
              </button>
            </article>
          ))}
        </div>
      ) : null}
      {resourceRead ? <McpResourceReadPanel result={resourceRead} /> : null}
      {importedTools.length > 0 ? (
        <div className="composer-inspector-list" aria-label="MCP imported tools">
          {importedTools.slice(0, 8).map((tool) => (
            <article key={tool.name} className="composer-inspector-list-item">
              <strong>{tool.local_name}</strong>
              <p>{tool.description || 'No description provided.'}</p>
              <small>{tool.server} / {tool.name}</small>
            </article>
          ))}
        </div>
      ) : null}
    </>
  )
}

type McpServerForm = {
  name: string
  kind: string
  command: string
  args: string
  url: string
  transport: string
  env: string
  headers: string
  enabled: boolean
}

function McpConfigManager({
  config,
  form,
  setForm,
  saving,
  message,
  error,
  onSave,
  onDelete,
}: {
  config: McpConfigList
  form: McpServerForm
  setForm: (form: McpServerForm) => void
  saving: boolean
  message: string | null
  error: string | null
  onSave: () => void
  onDelete: (server: McpConfiguredServer) => void
}) {
  const isRemote = form.kind === 'remote'
  return (
    <section className="composer-inspector-manager" aria-label="MCP server management">
      <header>
        <strong>MCP servers</strong>
        <small>{config.config_path}</small>
      </header>
      {config.servers.length > 0 ? (
        <div className="composer-inspector-list">
          {config.servers.map((server) => (
            <article key={server.name} className="composer-inspector-list-item">
              <strong>{server.name}</strong>
              <p>{server.command || server.url || server.transport}</p>
              <small>
                {server.enabled ? 'enabled' : 'disabled'} / {server.transport}
                {server.env_keys.length > 0 ? ' / env ' + server.env_keys.join(', ') : ''}
                {server.header_keys.length > 0 ? ' / headers ' + server.header_keys.join(', ') : ''}
              </small>
              <button type="button" className="composer-inspector-inline-action" disabled={saving} onClick={() => onDelete(server)}>
                Delete server
              </button>
            </article>
          ))}
        </div>
      ) : (
        <p className="composer-inspector-note">No MCP servers are saved in the managed config file.</p>
      )}
      <div className="composer-inspector-form">
        <label>
          <span>Name</span>
          <input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="filesystem" />
        </label>
        <label>
          <span>Type</span>
          <select value={form.kind} onChange={(event) => setForm({ ...form, kind: event.target.value })}>
            <option value="stdio">stdio command</option>
            <option value="remote">remote HTTP/SSE</option>
          </select>
        </label>
        {isRemote ? (
          <>
            <label>
              <span>URL</span>
              <input value={form.url} onChange={(event) => setForm({ ...form, url: event.target.value })} placeholder="https://example.com/mcp" />
            </label>
            <label>
              <span>Transport</span>
              <select value={form.transport} onChange={(event) => setForm({ ...form, transport: event.target.value })}>
                <option value="http">streamable HTTP</option>
                <option value="sse">SSE</option>
              </select>
            </label>
            <label className="composer-inspector-form-wide">
              <span>Headers</span>
              <textarea value={form.headers} onChange={(event) => setForm({ ...form, headers: event.target.value })} placeholder="Authorization=Bearer ${MCP_TOKEN}" />
            </label>
          </>
        ) : (
          <>
            <label>
              <span>Command</span>
              <input value={form.command} onChange={(event) => setForm({ ...form, command: event.target.value })} placeholder="node" />
            </label>
            <label className="composer-inspector-form-wide">
              <span>Args</span>
              <textarea value={form.args} onChange={(event) => setForm({ ...form, args: event.target.value })} placeholder={'server.js\n--root\n${WORKSPACE_ROOT}'} />
            </label>
            <label className="composer-inspector-form-wide">
              <span>Env</span>
              <textarea value={form.env} onChange={(event) => setForm({ ...form, env: event.target.value })} placeholder="TOKEN=${MCP_TOKEN}" />
            </label>
          </>
        )}
        <label className="composer-inspector-check">
          <input type="checkbox" checked={form.enabled} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} />
          <span>Enabled</span>
        </label>
        <button type="button" className="composer-inspector-primary" disabled={saving} onClick={onSave}>
          {saving ? 'Saving server' : 'Save MCP server'}
        </button>
      </div>
      {message ? <p className="composer-inspector-note">{message}</p> : null}
      {error ? <PanelError message={error} /> : null}
    </section>
  )
}

function McpResourceReadPanel({ result }: { result: LoadState<McpResourceReadResult> }) {
  if (result.state === 'loading' || result.state === 'idle') return <PanelLoading label="Reading MCP resource" />
  if (result.state === 'error') return <PanelError message={result.message} />

  const data = result.data
  return (
    <section className="composer-inspector-resource-read" aria-label="MCP resource content">
      <header>
        <strong>{data.name || data.uri}</strong>
        <small>{data.server} / {data.mime_type || data.status}</small>
      </header>
      <p className="composer-inspector-note">{data.message}</p>
      {data.contents.length === 0 ? <p className="composer-inspector-note">No resource content was returned.</p> : null}
      {data.contents.map((content, index) => (
        <article key={(content.uri || data.uri) + '-' + index}>
          <small>{content.mime_type || content.type}</small>
          {content.text ? (
            <pre className="composer-inspector-code">{content.text}</pre>
          ) : content.blob ? (
            <pre className="composer-inspector-code">{content.blob}</pre>
          ) : (
            <p className="composer-inspector-note">Binary or empty content.</p>
          )}
        </article>
      ))}
    </section>
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

function preflightStatus(preflight: ProviderPreflightStatus | null): string {
  if (!preflight) return 'not checked'
  if (preflight.status === 'ready') return 'ready'
  if (preflight.status === 'warning') return 'warning'
  if (preflight.status === 'error') return 'blocked'
  return preflight.status
}

function preflightDetail(preflight: ProviderPreflightStatus | null): string {
  if (!preflight) return 'No provider preflight was available.'
  if (preflight.issues.length > 0) return preflight.issues[0] || 'Check provider configuration.'
  if (preflight.chat_completions_url) return preflight.chat_completions_url
  if (preflight.credential?.configured) return 'credential configured'
  return 'credential and endpoint status checked'
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

function parseListLines(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
}

function parseKeyValueLines(value: string): Record<string, string> {
  const result: Record<string, string> = {}
  for (const line of value.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const separator = trimmed.indexOf('=')
    if (separator <= 0) continue
    const key = trimmed.slice(0, separator).trim()
    const item = trimmed.slice(separator + 1).trim()
    if (key) result[key] = item
  }
  return result
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
