type Props = {
  diff: string
}

type DiffLineKind = 'add' | 'remove' | 'context' | 'hunk' | 'meta'

type ParsedDiffLine = {
  kind: DiffLineKind
  marker: string
  content: string
  oldLine: number | null
  newLine: number | null
}

export function DiffViewer({ diff }: Props) {
  const lines = parseUnifiedDiff(diff)
  return (
    <div className="diff-viewer" role="table" aria-label="Code diff">
      {lines.map((line, index) => {
        return (
          <div className={'diff-line diff-line-' + line.kind} role="row" key={index + '-' + line.content}>
            <span className="diff-line-number">{displayLineNumber(line) ?? ''}</span>
            <span className="diff-marker">{line.marker}</span>
            <code>{line.content}</code>
          </div>
        )
      })}
    </div>
  )
}


function displayLineNumber(line: ParsedDiffLine): number | null {
  if (line.kind === 'remove') return line.oldLine
  if (line.kind === 'add' || line.kind === 'context') return line.newLine
  return null
}

export function parseUnifiedDiff(diff: string): ParsedDiffLine[] {
  const parsed: ParsedDiffLine[] = []
  let oldLine: number | null = null
  let newLine: number | null = null

  for (const rawLine of diff.replace(/\r\n/g, '\n').split('\n')) {
    const hunk = parseHunkHeader(rawLine)
    if (hunk) {
      oldLine = hunk.oldStart
      newLine = hunk.newStart
      parsed.push({ kind: 'hunk', marker: '@', content: rawLine, oldLine: null, newLine: null })
      continue
    }

    if (rawLine.startsWith('+++') || rawLine.startsWith('---') || rawLine.startsWith('diff --git')) {
      parsed.push({ kind: 'meta', marker: ' ', content: rawLine, oldLine: null, newLine: null })
      continue
    }

    if (rawLine.startsWith('+')) {
      parsed.push({
        kind: 'add',
        marker: '+',
        content: rawLine.slice(1),
        oldLine: null,
        newLine,
      })
      if (newLine != null) newLine += 1
      continue
    }

    if (rawLine.startsWith('-')) {
      parsed.push({
        kind: 'remove',
        marker: '-',
        content: rawLine.slice(1),
        oldLine,
        newLine: null,
      })
      if (oldLine != null) oldLine += 1
      continue
    }

    parsed.push({
      kind: 'context',
      marker: ' ',
      content: rawLine.startsWith(' ') ? rawLine.slice(1) : rawLine,
      oldLine,
      newLine,
    })
    if (oldLine != null) oldLine += 1
    if (newLine != null) newLine += 1
  }

  return parsed
}

function parseHunkHeader(line: string): { oldStart: number; newStart: number } | null {
  const match = line.match(/^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@/)
  if (!match) return null
  return {
    oldStart: Number(match[1]),
    newStart: Number(match[2]),
  }
}
