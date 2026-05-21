import { RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { LogFileSummary } from '../../api/types'
import { Spinner } from '../shared/Spinner'

const LEVELS = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR'] as const
const LINE_COUNTS = [50, 100, 200, 500] as const

export function LogsView() {
  const [files, setFiles] = useState<LogFileSummary[]>([])
  const [file, setFile] = useState('gateway')
  const [level, setLevel] = useState<(typeof LEVELS)[number]>('ALL')
  const [lineCount, setLineCount] = useState<(typeof LINE_COUNTS)[number]>(100)
  const [search, setSearch] = useState('')
  const [lines, setLines] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      api.logFiles(),
      api.logs({ file, lines: lineCount, level, search: search.trim() || undefined }),
    ])
      .then(([filesResult, logsResult]) => {
        setFiles(filesResult.files)
        setLines(logsResult.lines)
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [file, lineCount, level, search])

  useEffect(() => {
    refresh()
  }, [refresh])

  return (
    <div className="settings-panel">
      <header className="panel-header">
        <div>
          <h2>Logs</h2>
          <p>Tail and filter local runtime logs from AETHER_HOME.</p>
        </div>
        <button type="button" onClick={refresh} disabled={loading}>
          <RefreshCw size={15} /> Refresh
        </button>
      </header>

      <div className="log-controls" role="toolbar" aria-label="Log filters">
        <label>
          <span>File</span>
          <select value={file} onChange={(event) => setFile(event.target.value)}>
            {files.length === 0 ? <option value="gateway">gateway</option> : null}
            {files.map((item) => (
              <option value={item.key} key={item.key}>{item.key}{item.exists ? '' : ' (missing)'}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Level</span>
          <select value={level} onChange={(event) => setLevel(event.target.value as (typeof LEVELS)[number])}>
            {LEVELS.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </label>
        <label>
          <span>Lines</span>
          <select value={lineCount} onChange={(event) => setLineCount(Number(event.target.value) as (typeof LINE_COUNTS)[number])}>
            {LINE_COUNTS.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </label>
        <label className="log-search-field">
          <span>Search</span>
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter text" />
        </label>
      </div>

      {loading ? <Spinner label="Loading logs" /> : null}
      {error ? <div className="notice notice-error">{error}</div> : null}

      <section className="log-viewer" aria-label="Log output">
        {lines.length === 0 && !loading ? <div className="empty-chat">No log lines matched.</div> : null}
        {lines.map((line, index) => (
          <div className={'log-line log-line-' + classifyLine(line)} key={index}>{line}</div>
        ))}
      </section>
    </div>
  )
}

function classifyLine(line: string): 'error' | 'warning' | 'debug' | 'info' {
  const upper = line.toUpperCase()
  if (upper.includes('ERROR') || upper.includes('CRITICAL') || upper.includes('FATAL')) return 'error'
  if (upper.includes('WARNING') || upper.includes('WARN')) return 'warning'
  if (upper.includes('DEBUG')) return 'debug'
  return 'info'
}
