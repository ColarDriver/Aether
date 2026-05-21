import { Eye, EyeOff, KeyRound, Save, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../../api/client'
import type { EnvVarSummary } from '../../api/types'
import { Spinner } from '../shared/Spinner'

export function EnvironmentView() {
  const [envPath, setEnvPath] = useState('')
  const [variables, setVariables] = useState<EnvVarSummary[]>([])
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [revealed, setRevealed] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = () => {
    setLoading(true)
    setError(null)
    api.env()
      .then((catalog) => {
        setEnvPath(catalog.env_path)
        setVariables(catalog.variables)
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [])

  const grouped = useMemo(() => {
    const map = new Map<string, EnvVarSummary[]>()
    for (const variable of variables) {
      const group = variable.category || 'other'
      map.set(group, [...(map.get(group) ?? []), variable])
    }
    return [...map.entries()].sort(([a], [b]) => a.localeCompare(b))
  }, [variables])

  const save = (key: string) => {
    setSaving(key)
    setError(null)
    api.setEnvVar({ key, value: edits[key] ?? '' })
      .then(() => {
        setEdits((state) => withoutKey(state, key))
        setRevealed((state) => withoutKey(state, key))
        load()
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setSaving(null))
  }

  const clear = (key: string) => {
    setSaving(key)
    setError(null)
    api.deleteEnvVar(key)
      .then(() => {
        setEdits((state) => withoutKey(state, key))
        setRevealed((state) => withoutKey(state, key))
        load()
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setSaving(null))
  }

  const reveal = (key: string) => {
    if (revealed[key] !== undefined) {
      setRevealed((state) => withoutKey(state, key))
      return
    }
    setSaving(key)
    setError(null)
    api.revealEnvVar(key)
      .then((result) => setRevealed((state) => ({ ...state, [key]: result.value })))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setSaving(null))
  }

  return (
    <div className="settings-panel">
      <header className="panel-header">
        <div>
          <h2>Environment</h2>
          <p>Manage provider and tool keys stored in the local .env file.</p>
        </div>
        <span className="muted">{envPath || '.env'}</span>
      </header>
      {loading ? <Spinner label="Loading environment" /> : null}
      {error ? <div className="notice notice-error">{error}</div> : null}
      {grouped.map(([category, entries]) => (
        <section className="catalog-card env-group" key={category}>
          <div className="catalog-card-header">
            <strong>{category}</strong>
            <span>{entries.filter((entry) => entry.is_set).length} configured</span>
          </div>
          <div className="env-list">
            {entries.map((entry) => (
              <EnvRow
                key={entry.key}
                entry={entry}
                editValue={edits[entry.key]}
                revealedValue={revealed[entry.key]}
                saving={saving === entry.key}
                onEdit={(value) => setEdits((state) => ({ ...state, [entry.key]: value }))}
                onCancel={() => setEdits((state) => withoutKey(state, entry.key))}
                onSave={() => save(entry.key)}
                onDelete={() => clear(entry.key)}
                onReveal={() => reveal(entry.key)}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}

type RowProps = {
  entry: EnvVarSummary
  editValue: string | undefined
  revealedValue: string | undefined
  saving: boolean
  onEdit: (value: string) => void
  onCancel: () => void
  onSave: () => void
  onDelete: () => void
  onReveal: () => void
}

function EnvRow({ entry, editValue, revealedValue, saving, onEdit, onCancel, onSave, onDelete, onReveal }: RowProps) {
  const editing = editValue !== undefined
  const visibleValue = revealedValue ?? entry.redacted_value ?? ''
  return (
    <div className={'env-row' + (entry.is_set ? ' env-row-set' : '')}>
      <div className="env-row-main">
        <div className="env-key-line">
          <KeyRound size={14} />
          <strong>{entry.key}</strong>
          <span className={'env-source env-source-' + entry.source}>{entry.source}</span>
          {entry.advanced ? <span className="env-source">advanced</span> : null}
        </div>
        {entry.description ? <p>{entry.description}</p> : null}
      </div>
      <div className="env-value-panel">
        {editing ? (
          <input
            aria-label={'Value for ' + entry.key}
            value={editValue}
            onChange={(event) => onEdit(event.target.value)}
            placeholder={entry.is_secret ? 'Secret value' : 'Value'}
            type={entry.is_secret ? 'password' : 'text'}
          />
        ) : (
          <code>{entry.is_set ? visibleValue : 'not set'}</code>
        )}
      </div>
      <div className="env-actions">
        {editing ? (
          <>
            <button type="button" onClick={onSave} disabled={saving} title="Save">
              <Save size={14} /> Save
            </button>
            <button type="button" onClick={onCancel} disabled={saving}>Cancel</button>
          </>
        ) : (
          <>
            {entry.is_set ? (
              <button type="button" onClick={onReveal} disabled={saving} title={revealedValue ? 'Hide value' : 'Reveal value'}>
                {revealedValue ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            ) : null}
            <button type="button" onClick={() => onEdit('')} disabled={saving}>{entry.is_set ? 'Replace' : 'Set'}</button>
            {entry.is_set && entry.source === 'file' ? (
              <button type="button" onClick={onDelete} disabled={saving} title="Delete" className="danger-action">
                <Trash2 size={14} />
              </button>
            ) : null}
          </>
        )}
      </div>
    </div>
  )
}

function withoutKey<T>(record: Record<string, T>, key: string): Record<string, T> {
  const next = { ...record }
  delete next[key]
  return next
}
