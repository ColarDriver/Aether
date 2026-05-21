import { RefreshCw } from 'lucide-react'
import { useEffect } from 'react'
import { useProviderStore } from '../../stores/providerStore'

export function ProviderSettings() {
  const providers = useProviderStore((state) => state.providers)
  const current = useProviderStore((state) => state.current)
  const modelsByProvider = useProviderStore((state) => state.modelsByProvider)
  const loadProviders = useProviderStore((state) => state.loadProviders)
  const loadModels = useProviderStore((state) => state.loadModels)
  const selectModel = useProviderStore((state) => state.selectModel)

  useEffect(() => {
    if (!providers.length) void loadProviders()
  }, [loadProviders, providers.length])

  const activeProvider = current?.provider_name || providers[0]?.name || 'openai'
  const models = modelsByProvider[activeProvider] ?? []

  useEffect(() => {
    if (activeProvider) void loadModels(activeProvider)
  }, [activeProvider, loadModels])

  return (
    <div className="settings-panel">
      <header className="panel-header">
        <div>
          <h2>Provider and model</h2>
          <p>Current runtime selection and local credential readiness.</p>
        </div>
        <button type="button" onClick={() => void loadProviders()}>
          <RefreshCw size={15} /> Refresh
        </button>
      </header>
      <div className="info-grid compact-grid">
        <div className="info-row"><span>Provider</span><strong>{current?.provider_name || 'unconfigured'}</strong></div>
        <div className="info-row"><span>Family</span><strong>{current?.family || '-'}</strong></div>
        <div className="info-row"><span>Model</span><strong>{current?.model || '-'}</strong></div>
        <div className="info-row"><span>Credential</span><strong>{current?.credential?.configured ? 'configured' : 'missing'}</strong></div>
      </div>
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
                  onClick={() => void selectModel(provider.name, model.id)}
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
