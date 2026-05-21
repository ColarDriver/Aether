import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { ToolGroup } from '../../api/types'
import { Spinner } from '../shared/Spinner'

export function ToolsView() {
  const [groups, setGroups] = useState<ToolGroup[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    api.toolGroups()
      .then((result) => setGroups(result.groups))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="settings-panel">
      <header className="panel-header">
        <div><h2>Tools</h2><p>Built-in tools grouped by runtime category.</p></div>
      </header>
      {loading ? <Spinner label="Loading tools" /> : null}
      {error ? <div className="notice notice-error">{error}</div> : null}
      <div className="catalog-list">
        {groups.map((group) => (
          <section className="catalog-card" key={group.name}>
            <div className="catalog-card-header"><strong>{group.name}</strong><span>{group.tools.length} tools</span></div>
            <div className="tool-grid">
              {group.tools.map((tool) => (
                <div className="tool-row" key={tool.name}>
                  <strong>{tool.name}</strong>
                  <span>{tool.description}</span>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
