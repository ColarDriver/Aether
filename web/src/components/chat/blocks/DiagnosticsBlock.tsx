import { AlertCircle, AlertTriangle, Info, Lightbulb } from 'lucide-react'
import type { DiagnosticEntry, DiagnosticsBlock as DiagnosticsChatBlock } from '../../../chat-rendering'

type Props = {
  block: DiagnosticsChatBlock
}

export function DiagnosticsBlock({ block }: Props) {
  const count = block.files.reduce((total, file) => total + file.diagnostics.length, 0)
  return (
    <article className="chat-block diagnostics-block" aria-label="Diagnostics">
      <header className="diagnostics-block-header">
        <span className="diagnostics-block-icon" aria-hidden="true"><AlertCircle size={16} /></span>
        <span>
          <strong>Diagnostics</strong>
          <small>{count.toLocaleString()} issue{count === 1 ? '' : 's'} after recent edits</small>
        </span>
      </header>
      <div className="diagnostics-file-list">
        {block.files.map((file) => (
          <section className="diagnostics-file" key={file.path}>
            <header>
              <strong>{file.path}</strong>
              <em>{file.diagnostics.length.toLocaleString()}</em>
            </header>
            <div className="diagnostics-row-list">
              {file.diagnostics.map((diagnostic, index) => (
                <DiagnosticRow diagnostic={diagnostic} key={index} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </article>
  )
}

function DiagnosticRow({ diagnostic }: { diagnostic: DiagnosticEntry }) {
  const Icon = iconForSeverity(diagnostic.severity)
  return (
    <div className={'diagnostics-row diagnostics-row-' + severityTone(diagnostic.severity)}>
      <span className="diagnostics-severity" title={diagnostic.severity}>
        <Icon size={13} aria-hidden="true" />
        <strong>{diagnostic.severity}</strong>
      </span>
      <code>{diagnostic.line}:{diagnostic.column}</code>
      <span className="diagnostics-message">
        <strong>{diagnostic.source}{diagnostic.code ? ' [' + diagnostic.code + ']' : ''}</strong>
        <span>{diagnostic.message}</span>
      </span>
    </div>
  )
}

function iconForSeverity(severity: string) {
  const normalized = severity.toLowerCase()
  if (normalized === 'error') return AlertCircle
  if (normalized === 'warning') return AlertTriangle
  if (normalized === 'hint') return Lightbulb
  return Info
}

function severityTone(severity: string): string {
  const normalized = severity.toLowerCase()
  if (normalized === 'error') return 'error'
  if (normalized === 'warning') return 'warning'
  if (normalized === 'hint') return 'hint'
  return 'info'
}
