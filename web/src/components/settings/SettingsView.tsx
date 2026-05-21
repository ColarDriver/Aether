import { Plus, Save, Trash2 } from "lucide-react"
import { FormEvent, useEffect, useMemo, useState } from "react"
import { api } from "../../api/client"
import type { ConfigPaths, EffectiveConfig } from "../../api/types"
import { useToastStore } from "../../stores/toastStore"
import { Spinner } from "../shared/Spinner"

export function SettingsView() {
  const [config, setConfig] = useState<EffectiveConfig | null>(null)
  const [paths, setPaths] = useState<ConfigPaths | null>(null)
  const [prefs, setPrefs] = useState<Record<string, unknown> | null>(null)
  const [prefEdits, setPrefEdits] = useState<Record<string, string>>({})
  const [newPrefKey, setNewPrefKey] = useState("")
  const [newPrefValue, setNewPrefValue] = useState("")
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const notify = useToastStore((state) => state.notify)

  const load = () => {
    setLoading(true)
    setError(null)
    Promise.all([api.config(), api.configPaths(), api.prefs()])
      .then(([configResult, pathResult, prefsResult]) => {
        setConfig(configResult)
        setPaths(pathResult)
        setPrefs(prefsResult)
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [])

  const prefEntries = useMemo(() => {
    return Object.entries(prefs ?? {}).sort(([left], [right]) => left.localeCompare(right))
  }, [prefs])

  const startEdit = (key: string, value: unknown) => {
    setPrefEdits((state) => ({ ...state, [key]: formatPrefValue(value) }))
  }

  const cancelEdit = (key: string) => {
    setPrefEdits((state) => withoutKey(state, key))
  }

  const savePref = (key: string) => {
    const raw = prefEdits[key] ?? ""
    setSaving(key)
    setError(null)
    api.setPref({ key, value: parsePrefInput(raw) })
      .then(() => {
        setPrefEdits((state) => withoutKey(state, key))
        notify("Saved " + key, "success")
        load()
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => setSaving(null))
  }

  const deletePref = (key: string) => {
    setSaving(key)
    setError(null)
    api.deletePref(key)
      .then(() => {
        setPrefEdits((state) => withoutKey(state, key))
        notify("Deleted " + key, "success")
        load()
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => setSaving(null))
  }

  const addPref = (event: FormEvent) => {
    event.preventDefault()
    const key = newPrefKey.trim()
    if (!key) return
    setSaving(key)
    setError(null)
    api.setPref({ key, value: parsePrefInput(newPrefValue) })
      .then(() => {
        setNewPrefKey("")
        setNewPrefValue("")
        notify("Saved " + key, "success")
        load()
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        notify(message, "error")
      })
      .finally(() => setSaving(null))
  }

  return (
    <div className="settings-panel">
      <header className="panel-header">
        <div>
          <h2>Settings</h2>
          <p>Public configuration summary and local preference overrides.</p>
        </div>
      </header>
      {loading ? <Spinner label="Loading settings" /> : null}
      {error ? <div className="notice notice-error">{error}</div> : null}
      <div className="info-grid compact-grid">
        <div className="info-row"><span>AETHER_HOME</span><strong>{paths?.aether_home || "-"}</strong></div>
        <div className="info-row"><span>Sessions</span><strong>{paths?.sessions_dir || "-"}</strong></div>
        <div className="info-row"><span>Prefs</span><strong>{paths?.prefs_file || "-"}</strong></div>
        <div className="info-row"><span>Prefs version</span><strong>{String(prefs?.version ?? "-")}</strong></div>
      </div>
      <section className="catalog-card prefs-card">
        <div className="catalog-card-header"><strong>Preferences</strong><span>{prefEntries.length} fields</span></div>
        <div className="pref-list">
          {prefEntries.map(([key, value]) => (
            <PrefRow
              key={key}
              prefKey={key}
              value={value}
              editValue={prefEdits[key]}
              saving={saving === key}
              readOnly={key === "version"}
              onEdit={() => startEdit(key, value)}
              onChange={(next) => setPrefEdits((state) => ({ ...state, [key]: next }))}
              onCancel={() => cancelEdit(key)}
              onSave={() => savePref(key)}
              onDelete={() => deletePref(key)}
            />
          ))}
        </div>
        <form className="pref-add-form" onSubmit={addPref}>
          <input
            aria-label="New preference key"
            placeholder="ui.example"
            value={newPrefKey}
            onChange={(event) => setNewPrefKey(event.target.value)}
          />
          <input
            aria-label="New preference value"
            placeholder="value or JSON"
            value={newPrefValue}
            onChange={(event) => setNewPrefValue(event.target.value)}
          />
          <button type="submit" disabled={saving !== null || !newPrefKey.trim()}>
            <Plus size={14} /> Add
          </button>
        </form>
      </section>
      <section className="catalog-card">
        <div className="catalog-card-header"><strong>Effective config</strong><span>{Object.keys(config?.values ?? {}).length} fields</span></div>
        <pre className="json-preview">{JSON.stringify(config?.values ?? {}, null, 2)}</pre>
      </section>
    </div>
  )
}

type PrefRowProps = {
  prefKey: string
  value: unknown
  editValue: string | undefined
  saving: boolean
  readOnly: boolean
  onEdit: () => void
  onChange: (value: string) => void
  onCancel: () => void
  onSave: () => void
  onDelete: () => void
}

function PrefRow({ prefKey, value, editValue, saving, readOnly, onEdit, onChange, onCancel, onSave, onDelete }: PrefRowProps) {
  const editing = editValue !== undefined
  return (
    <div className="pref-row">
      <div className="pref-row-main">
        <strong>{prefKey}</strong>
        {readOnly ? <span>read only</span> : null}
      </div>
      <div className="pref-value-panel">
        {editing ? (
          <textarea
            aria-label={"Preference value for " + prefKey}
            value={editValue}
            onChange={(event) => onChange(event.target.value)}
          />
        ) : (
          <code>{formatPrefValue(value)}</code>
        )}
      </div>
      <div className="pref-actions">
        {editing ? (
          <>
            <button type="button" onClick={onSave} disabled={saving} title="Save preference">
              <Save size={14} /> Save
            </button>
            <button type="button" onClick={onCancel} disabled={saving}>Cancel</button>
          </>
        ) : (
          <>
            <button type="button" onClick={onEdit} disabled={saving || readOnly} aria-label={"Edit " + prefKey}>Edit</button>
            <button type="button" onClick={onDelete} disabled={saving || readOnly} className="danger-action" title="Delete preference" aria-label={"Delete " + prefKey}>
              <Trash2 size={14} />
            </button>
          </>
        )}
      </div>
    </div>
  )
}

function formatPrefValue(value: unknown): string {
  if (typeof value === "string") return value
  return JSON.stringify(value, null, 2)
}

function parsePrefInput(raw: string): unknown {
  const trimmed = raw.trim()
  if (!trimmed) return ""
  try {
    return JSON.parse(trimmed)
  } catch {
    return raw
  }
}

function withoutKey<T>(record: Record<string, T>, key: string): Record<string, T> {
  const next = { ...record }
  delete next[key]
  return next
}
