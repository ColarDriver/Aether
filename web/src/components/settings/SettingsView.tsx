import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { ConfigPaths, EffectiveConfig } from '../../api/types'
import { Spinner } from '../shared/Spinner'

export function SettingsView() {
  const [config, setConfig] = useState<EffectiveConfig | null>(null)
  const [paths, setPaths] = useState<ConfigPaths | null>(null)
  const [prefs, setPrefs] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    Promise.all([api.config(), api.configPaths(), api.prefs()])
      .then(([configResult, pathResult, prefsResult]) => {
        setConfig(configResult)
        setPaths(pathResult)
        setPrefs(prefsResult)
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="settings-panel">
      <header className="panel-header"><div><h2>Settings</h2><p>Read-only public configuration summary.</p></div></header>
      {loading ? <Spinner label="Loading settings" /> : null}
      <div className="info-grid compact-grid">
        <div className="info-row"><span>AETHER_HOME</span><strong>{paths?.aether_home || '-'}</strong></div>
        <div className="info-row"><span>Sessions</span><strong>{paths?.sessions_dir || '-'}</strong></div>
        <div className="info-row"><span>Prefs</span><strong>{paths?.prefs_file || '-'}</strong></div>
        <div className="info-row"><span>Prefs version</span><strong>{String(prefs?.version ?? '-')}</strong></div>
      </div>
      <section className="catalog-card">
        <div className="catalog-card-header"><strong>Effective config</strong><span>{Object.keys(config?.values ?? {}).length} fields</span></div>
        <pre className="json-preview">{JSON.stringify(config?.values ?? {}, null, 2)}</pre>
      </section>
    </div>
  )
}
