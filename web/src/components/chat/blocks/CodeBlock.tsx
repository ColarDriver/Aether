import type { ReactNode } from 'react'
import { CopyButton } from '../../shared/CopyButton'

type Props = {
  code: string
  language?: string | null
  title?: string
  wrap?: boolean
}

export function CodeBlock({ code, language = '', title, wrap = false }: Props) {
  const label = title || language || 'code'
  return (
    <div className="code-block">
      <div className="code-block-header">
        <span>{label}</span>
        <CopyButton text={code} label={'Copy ' + label} className="code-block-copy" />
      </div>
      <pre className={wrap ? 'code-block-body code-block-wrap' : 'code-block-body'}>
        <code>{highlightCode(code, language ?? '')}</code>
      </pre>
    </div>
  )
}

export function highlightCode(code: string, language: string): ReactNode {
  const normalized = language.toLowerCase()
  const keywords = keywordsForLanguage(normalized)
  if (!keywords && !['json', 'jsonc'].includes(normalized)) {
    return code
  }
  const keywordPattern = keywords ? '|\\b(?:' + keywords.join('|') + ')\\b' : ''
  const pattern = new RegExp(
    '(#.*$|\\/\\/.*$|"(?:\\\\.|[^"\\\\])*"|\\\'(?:\\\\.|[^\\\'\\\\])*\\\'' +
    '|`(?:\\\\.|[^`\\\\])*`' +
    keywordPattern +
    '|\\btrue\\b|\\bfalse\\b|\\bnull\\b|\\bNone\\b|\\bTrue\\b|\\bFalse\\b|-?\\b\\d+(?:\\.\\d+)?\\b)',
    'gm',
  )
  const parts: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = pattern.exec(code)) !== null) {
    if (match.index > lastIndex) parts.push(code.slice(lastIndex, match.index))
    const value = match[0]
    const tokenClass = value.startsWith('#') || value.startsWith('//')
      ? 'syntax-comment'
      : value.startsWith('"') || value.startsWith("'") || value.startsWith('`')
      ? 'syntax-string'
      : isKeyword(value, keywords)
        ? 'syntax-keyword'
      : value === 'true' || value === 'false' || value === 'True' || value === 'False'
        ? 'syntax-boolean'
        : value === 'null' || value === 'None'
          ? 'syntax-null'
          : 'syntax-number'
    parts.push(<span className={tokenClass} key={parts.length}>{value}</span>)
    lastIndex = match.index + value.length
  }
  if (lastIndex < code.length) parts.push(code.slice(lastIndex))
  return parts
}

function keywordsForLanguage(language: string): string[] | null {
  if (['js', 'javascript', 'ts', 'typescript', 'tsx', 'jsx'].includes(language)) {
    return [
      'async', 'await', 'break', 'case', 'catch', 'class', 'const', 'continue',
      'default', 'else', 'export', 'extends', 'finally', 'for', 'from', 'function',
      'if', 'import', 'let', 'new', 'return', 'switch', 'throw', 'try', 'type',
      'var', 'while',
    ]
  }
  if (['py', 'python'].includes(language)) {
    return [
      'and', 'as', 'async', 'await', 'class', 'def', 'elif', 'else', 'except',
      'finally', 'for', 'from', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or',
      'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
    ]
  }
  if (['bash', 'sh', 'shell', 'zsh'].includes(language)) {
    return ['case', 'do', 'done', 'elif', 'else', 'esac', 'fi', 'for', 'function', 'if', 'in', 'then', 'while']
  }
  return null
}

function isKeyword(value: string, keywords: string[] | null): boolean {
  return Boolean(keywords?.includes(value))
}
