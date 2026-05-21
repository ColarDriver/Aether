import type { ReactNode } from 'react'

type Props = {
  code: string
  language?: string | null
  title?: string
  wrap?: boolean
}

export function CodeBlock({ code, language = '', title, wrap = false }: Props) {
  const label = title || language || ''
  return (
    <div className="code-block">
      {label ? <div className="code-block-header">{label}</div> : null}
      <pre className={wrap ? 'code-block-body code-block-wrap' : 'code-block-body'}>
        <code>{highlightCode(code, language ?? '')}</code>
      </pre>
    </div>
  )
}

export function highlightCode(code: string, language: string): ReactNode {
  if (!['json', 'jsonc', 'js', 'javascript', 'ts', 'typescript'].includes(language.toLowerCase())) {
    return code
  }
  const pattern = /("(?:\\.|[^"\\])*"|\btrue\b|\bfalse\b|\bnull\b|-?\b\d+(?:\.\d+)?\b)/g
  const parts: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = pattern.exec(code)) !== null) {
    if (match.index > lastIndex) parts.push(code.slice(lastIndex, match.index))
    const value = match[0]
    const tokenClass = value.startsWith('"')
      ? 'syntax-string'
      : value === 'true' || value === 'false'
        ? 'syntax-boolean'
        : value === 'null'
          ? 'syntax-null'
          : 'syntax-number'
    parts.push(<span className={tokenClass} key={parts.length}>{value}</span>)
    lastIndex = match.index + value.length
  }
  if (lastIndex < code.length) parts.push(code.slice(lastIndex))
  return parts
}
