import { Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../../api/client'
import type { ToolGroup, ToolSummary } from '../../api/types'
import { Spinner } from '../shared/Spinner'

type ToolWithGroup = ToolSummary & { group: string }

export function ToolsView() {
  const [groups, setGroups] = useState<ToolGroup[]>([])
  const [activeToolName, setActiveToolName] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    api.toolGroups()
      .then((result) => {
        setGroups(result.groups)
        const firstTool = result.groups[0]?.tools[0]
        if (firstTool) setActiveToolName(firstTool.name)
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [])

  const tools = useMemo(() => flattenGroups(groups), [groups])
  const filteredGroups = useMemo(() => filterGroups(groups, query), [groups, query])
  const filteredTools = useMemo(() => flattenGroups(filteredGroups), [filteredGroups])
  const availableTools = query.trim() ? filteredTools : tools
  const activeTool = availableTools.find((tool) => tool.name === activeToolName) ?? availableTools[0] ?? null

  return (
    <div className="settings-panel tools-panel">
      <header className="panel-header">
        <div><h2>Tools</h2><p>Inspect built-in tools, required parameters, and runtime categories.</p></div>
      </header>
      {loading ? <Spinner label="Loading tools" /> : null}
      {error ? <div className="notice notice-error">{error}</div> : null}

      <div className="tools-layout">
        <aside className="tools-index" aria-label="Tool catalog">
          <label className="tools-search">
            <Search size={14} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search tools" />
          </label>
          <div className="tools-group-list">
            {filteredGroups.map((group) => (
              <section key={group.name}>
                <div className="tools-group-heading"><strong>{group.name}</strong><span>{group.tools.length}</span></div>
                {group.tools.map((tool) => (
                  <button
                    type="button"
                    key={tool.name}
                    className={activeTool?.name === tool.name ? 'active' : ''}
                    onClick={() => setActiveToolName(tool.name)}
                  >
                    <strong>{tool.name}</strong>
                    <span>{tool.description || 'No description'}</span>
                  </button>
                ))}
              </section>
            ))}
            {filteredGroups.length === 0 && !loading ? <div className="empty-chat">No tools matched.</div> : null}
          </div>
        </aside>

        <section className="tool-detail" aria-label="Tool details">
          {activeTool ? (
            <>
              <div className="tool-detail-header">
                <div>
                  <span>{activeTool.group}</span>
                  <h3>{activeTool.name}</h3>
                  <p>{activeTool.description || 'No description'}</p>
                </div>
                <span className={activeTool.enabled ? 'tool-status tool-status-enabled' : 'tool-status'}>
                  {activeTool.enabled ? 'enabled' : 'disabled'}
                </span>
              </div>

              <div className="tool-detail-section">
                <strong>Required</strong>
                <div className="tool-required-list">
                  {activeTool.required.length > 0 ? activeTool.required.map((item) => <span key={item}>{item}</span>) : <em>None</em>}
                </div>
              </div>

              <div className="tool-detail-section">
                <strong>Parameters</strong>
                <ParameterTable tool={activeTool} />
              </div>

              <div className="tool-detail-section">
                <strong>Schema</strong>
                <pre className="tool-schema">{JSON.stringify(activeTool.parameters, null, 2)}</pre>
              </div>
            </>
          ) : !loading ? (
            <div className="empty-chat">Select a tool.</div>
          ) : null}
        </section>
      </div>
    </div>
  )
}

function flattenGroups(groups: ToolGroup[]): ToolWithGroup[] {
  return groups.flatMap((group) => group.tools.map((tool) => ({ ...tool, group: group.name })))
}

function filterGroups(groups: ToolGroup[], query: string): ToolGroup[] {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return groups
  return groups
    .map((group) => ({
      ...group,
      tools: group.tools.filter((tool) => (
        group.name.toLowerCase().includes(normalized)
        || tool.name.toLowerCase().includes(normalized)
        || tool.description.toLowerCase().includes(normalized)
      )),
    }))
    .filter((group) => group.tools.length > 0)
}

function ParameterTable({ tool }: { tool: ToolSummary }) {
  const properties = getProperties(tool.parameters)
  if (properties.length === 0) return <div className="empty-chat">No structured parameters.</div>
  const required = new Set(tool.required)
  return (
    <div className="tool-param-table-wrap">
      <table className="tool-param-table">
        <thead><tr><th>Name</th><th>Type</th><th>Required</th><th>Description</th></tr></thead>
        <tbody>
          {properties.map(([name, schema]) => (
            <tr key={name}>
              <td><code>{name}</code></td>
              <td>{schemaType(schema)}</td>
              <td>{required.has(name) ? 'yes' : 'no'}</td>
              <td>{schemaDescription(schema)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function getProperties(parameters: Record<string, unknown>): Array<[string, Record<string, unknown>]> {
  const properties = parameters.properties
  if (!properties || typeof properties !== 'object' || Array.isArray(properties)) return []
  return Object.entries(properties as Record<string, unknown>).map(([name, schema]) => [
    name,
    schema && typeof schema === 'object' && !Array.isArray(schema) ? schema as Record<string, unknown> : {},
  ])
}

function schemaType(schema: Record<string, unknown>): string {
  const value = schema.type
  if (Array.isArray(value)) return value.join(' | ')
  if (typeof value === 'string') return value
  if (schema.enum) return 'enum'
  return 'unknown'
}

function schemaDescription(schema: Record<string, unknown>): string {
  const value = schema.description
  return typeof value === 'string' && value.trim() ? value : '-'
}
