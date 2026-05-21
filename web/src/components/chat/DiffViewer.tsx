type Props = {
  diff: string
}

export function DiffViewer({ diff }: Props) {
  const lines = diff.split('\n')
  return (
    <div className="diff-viewer" role="table" aria-label="Code diff">
      {lines.map((line, index) => {
        const kind = line.startsWith('+') && !line.startsWith('+++')
          ? 'add'
          : line.startsWith('-') && !line.startsWith('---')
            ? 'remove'
            : 'context'
        return (
          <div className={'diff-line diff-line-' + kind} role="row" key={index + '-' + line}>
            <span className="diff-marker">{markerFor(kind, line)}</span>
            <code>{line}</code>
          </div>
        )
      })}
    </div>
  )
}

function markerFor(kind: string, line: string) {
  if (kind === 'add') return '+'
  if (kind === 'remove') return '-'
  return line.startsWith('@') ? '@' : ' '
}
