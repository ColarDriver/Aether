import { CheckCircle2, Terminal, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { CopyButton } from '../../shared/CopyButton'

type Props = {
  command?: string | null
  output: string
  isError?: boolean
  exitCode?: number | null
  durationMs?: number | null
  maxLines?: number
}

export function TerminalChrome({
  command,
  output,
  isError = false,
  exitCode = null,
  durationMs = null,
  maxLines = 80,
}: Props) {
  const [expanded, setExpanded] = useState(false)
  const { visibleOutput, truncated, hiddenLineCount } = useMemo(() => trimOutput(output, expanded, maxLines), [expanded, maxLines, output])
  const status = isError || (typeof exitCode === 'number' && exitCode !== 0) ? 'failed' : 'completed'
  const StatusIcon = status === 'failed' ? XCircle : CheckCircle2

  return (
    <section className={'terminal-chrome terminal-chrome-' + status} aria-label="Terminal output">
      <header className="terminal-chrome-bar">
        <span className="terminal-chrome-title">
          <Terminal size={14} />
          <strong>{status === 'failed' ? 'Command failed' : 'Command output'}</strong>
        </span>
        <span className="terminal-chrome-status">
          <StatusIcon size={13} />
          {typeof exitCode === 'number' ? 'exit ' + exitCode : status}
        </span>
        {typeof durationMs === 'number' ? <span className="terminal-chrome-meta">{formatDuration(durationMs)}</span> : null}
        <CopyButton text={output} label="Copy terminal output" className="terminal-chrome-copy" />
      </header>
      {command ? (
        <div className="terminal-chrome-command">
          <span>$</span>
          <code>{command}</code>
        </div>
      ) : null}
      <pre className="terminal-chrome-output"><code>{visibleOutput || '(no output)'}</code></pre>
      {truncated ? (
        <button type="button" className="terminal-chrome-expand" onClick={() => setExpanded(true)}>
          Show {hiddenLineCount.toLocaleString()} more lines
        </button>
      ) : null}
    </section>
  )
}

function trimOutput(output: string, expanded: boolean, maxLines: number) {
  const lines = output.split('\n')
  const truncated = !expanded && lines.length > maxLines
  if (!truncated) return { visibleOutput: output, truncated: false, hiddenLineCount: 0 }
  return {
    visibleOutput: lines.slice(0, maxLines).join('\n'),
    truncated: true,
    hiddenLineCount: lines.length - maxLines,
  }
}

function formatDuration(durationMs: number) {
  if (durationMs < 1000) return Math.round(durationMs) + 'ms'
  return (durationMs / 1000).toFixed(durationMs < 10_000 ? 1 : 0) + 's'
}
