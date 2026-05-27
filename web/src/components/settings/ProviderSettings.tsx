import { AlertTriangle, CheckCircle2, RefreshCw, Wifi } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useProviderStore } from '../../stores/providerStore'
import { useSessionStore } from '../../stores/sessionStore'

export function ProviderSettings() {
  const providers = useProviderStore((state) => state.providers)
  const current = useProviderStore((state) => state.current)
  const preflight = useProviderStore((state) => state.preflight)
  const modelsByProvider = useProviderStore((state) => state.modelsByProvider)
  const discoveryByProvider = useProviderStore((state) => state.discoveryByProvider)
  const loadProviders = useProviderStore((state) => state.loadProviders)
  const loadModels = useProviderStore((state) => state.loadModels)
  const loadPreflight = useProviderStore((state) => state.loadPreflight)
  const selectModel = useProviderStore((state) => state.selectModel)
  const activeSessionId = useSessionStore((state) => state.activeSessionId)
  const activeSession = useSessionStore((state) => state.sessions.find((session) => session.session_id === state.activeSessionId) ?? null)
  const updateSession = useSessionStore((state) => state.updateSession)
  const [testingProvider, setTestingProvider] = useState(false)
  const [selectionMessage, setSelectionMessage] = useState<string | null>(null)
  const [selectionError, setSelectionError] = useState<string | null>(null)

  useEffect(() => {
    if (!providers.length) void loadProviders()
  }, [loadProviders, providers.length])

  const activeProvider = current?.provider_name || providers[0]?.name || 'openai'
  const models = modelsByProvider[activeProvider] ?? []
  const discovery = discoveryByProvider[activeProvider]
  const preflightParams = {
    provider: activeSession?.provider || current?.provider_name || activeProvider,
    model: activeSession?.model || current?.model || null,
    baseUrl: activeSession?.base_url || current?.base_url || null,
  }

  useEffect(() => {
    if (activeProvider) void loadModels(activeProvider)
  }, [activeProvider, loadModels])

  useEffect(() => {
    if (preflightParams.provider) void Promise.resolve(loadPreflight(preflightParams)).catch(() => undefined)
  }, [loadPreflight, preflightParams.baseUrl, preflightParams.model, preflightParams.provider])

  const testProvider = async () => {
    if (!activeProvider) return
    setTestingProvider(true)
    try {
      await Promise.all([
        Promise.resolve(loadModels(activeProvider, { force: true })),
        Promise.resolve(loadPreflight(preflightParams)),
      ])
    } finally {
      setTestingProvider(false)
    }
  }

  const selectProviderModel = async (provider: string, model: string) => {
    setSelectionMessage(null)
    setSelectionError(null)
    try {
      const result = await selectModel(provider, model)
      if (activeSessionId) {
        const updated = await updateSession(activeSessionId, {
          provider: result.provider,
          model: result.model,
          base_url: result.base_url ?? null,
        })
        setSelectionMessage('Updated current session `' + updated.session_id.slice(0, 8) + '` to `' + updated.provider + '/' + updated.model + '`.')
      } else {
        setSelectionMessage('Saved default model `' + result.provider + '/' + result.model + '` for new sessions.')
      }
    } catch (error) {
      setSelectionError(error instanceof Error ? error.message : String(error))
    }
  }

  return (
    <div className="settings-panel">
      <header className="panel-header">
        <div>
          <h2>Provider and model</h2>
          <p>Current runtime selection and local credential readiness.</p>
        </div>
        <div className="panel-header-actions">
          <button type="button" onClick={testProvider} disabled={testingProvider}>
            <Wifi size={15} /> {testingProvider ? 'Testing' : 'Test provider'}
          </button>
          <button type="button" onClick={() => void loadProviders()}>
            <RefreshCw size={15} /> Refresh
          </button>
        </div>
      </header>
      <div className="info-grid compact-grid">
        <div className="info-row"><span>Provider</span><strong>{current?.provider_name || 'unconfigured'}</strong></div>
        <div className="info-row"><span>Family</span><strong>{current?.family || '-'}</strong></div>
        <div className="info-row"><span>Model</span><strong>{current?.model || '-'}</strong></div>
        <div className="info-row"><span>Credential</span><strong>{current?.credential?.configured ? current.credential.name || 'configured' : 'missing'}</strong></div>
        <div className="info-row"><span>Base URL</span><strong>{current?.base_url || 'default'}</strong></div>
        <div className="info-row"><span>Discovery</span><strong>{formatDiscovery(discovery)}</strong></div>
        <div className="info-row"><span>Source</span><strong>{current?.source || '-'}</strong></div>
      </div>
      <section className="catalog-card provider-runtime-card" aria-label="Provider runtime status">
        <div className="catalog-card-header">
          <strong><CheckCircle2 size={15} /> Runtime readiness</strong>
          <span>{discovery?.kind === 'live' ? 'live model discovery' : 'static fallback'}</span>
        </div>
        <div className="provider-runtime-grid">
          <div><span>API key env</span><strong>{current?.api_key_env_names?.join(', ') || '-'}</strong></div>
          <div><span>Credential state</span><strong>{current?.credential?.configured ? 'configured' : 'missing'}</strong></div>
          <div><span>Discovery detail</span><strong>{discoveryDetail(discovery)}</strong></div>
          <div><span>Current session</span><strong>{activeSession ? activeSession.provider + '/' + activeSession.model : 'none selected'}</strong></div>
          <div><span>Preflight</span><strong>{preflightLabel(preflight)}</strong></div>
          <div><span>Chat endpoint</span><strong>{preflight?.chat_completions_url || 'provider default'}</strong></div>
          <div><span>Models endpoint</span><strong>{preflight?.models_url || discovery?.url || 'not checked'}</strong></div>
        </div>
        {selectionMessage ? <div className="notice notice-success">{selectionMessage}</div> : null}
        {selectionError ? <div className="notice notice-error">{selectionError}</div> : null}
        {discovery?.warning ? <div className="notice notice-warning">{discovery.warning}</div> : null}
        {discovery?.error ? <div className="notice notice-error">{discovery.error}</div> : null}
        {preflight?.issues?.map((issue) => (
          <div className="notice notice-warning" key={issue}>
            <AlertTriangle size={14} />
            <span>{issue}</span>
          </div>
        ))}
        {preflight?.suggestions?.map((suggestion) => (
          <div className="notice" key={suggestion}>{suggestion}</div>
        ))}
      </section>
      <div className="catalog-list two-column-list">
        {providers.map((provider) => (
          <section className="catalog-card" key={provider.name}>
            <div className="catalog-card-header">
              <strong>{provider.display_name}</strong>
              <span>{provider.name}</span>
            </div>
            <p>{provider.requires_api_key ? 'API key required' : 'No API key required'}</p>
            <div className="model-list">
              {(modelsByProvider[provider.name] ?? []).slice(0, 8).map((model) => (
                <button
                  type="button"
                  key={model.id}
                  className={current?.model === model.id && current.provider_name === provider.name ? 'model-chip active-model' : 'model-chip'}
                  onClick={() => void selectProviderModel(provider.name, model.id)}
                >
                  {model.display_name || model.id}
                </button>
              ))}
              {provider.name === activeProvider && models.length === 0 ? <span className="muted">No model list loaded yet</span> : null}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}


function formatDiscovery(discovery: ReturnType<typeof useProviderStore.getState>['discoveryByProvider'][string] | undefined): string {
  if (!discovery) return 'not checked'
  if (discovery.kind === 'live') return discovery.count ? 'live · ' + discovery.count + ' models' : 'live'
  return discovery.reason ? 'static · ' + discovery.reason : 'static'
}

function discoveryDetail(discovery: ReturnType<typeof useProviderStore.getState>['discoveryByProvider'][string] | undefined): string {
  if (!discovery) return 'Model list has not been checked yet.'
  if (discovery.kind === 'live') return discovery.url || discovery.source || 'provider responded'
  return discovery.error || discovery.reason || 'using bundled model catalog'
}

function preflightLabel(preflight: ReturnType<typeof useProviderStore.getState>['preflight']): string {
  if (!preflight) return 'not checked'
  if (preflight.status === 'ready') return 'ready'
  if (preflight.status === 'warning') return 'warning'
  if (preflight.status === 'error') return 'blocked'
  return preflight.status
}
