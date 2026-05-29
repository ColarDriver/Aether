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
  const keywordPattern = keywords ? '\\b(?:' + keywords.join('|') + ')\\b' : ''
  const tokenPattern = [
    '#.*$',
    '\\/\\/.*$',
    '"(?:\\\\.|[^"\\\\])*"',
    "'(?:\\.|[^'\\])*'",
    '`(?:\\\\.|[^`\\\\])*`',
    keywordPattern,
    '\\btrue\\b|\\bfalse\\b|\\bnull\\b|\\bNone\\b|\\bTrue\\b|\\bFalse\\b',
    '-?\\b\\d+(?:\\.\\d+)?\\b',
    '\\b[A-Za-z_$][\\w$]*\\b',
    '[{}()[\\]<>.=:+\\-*/%,;!?|&]+',
  ].filter(Boolean).join('|')
  const pattern = new RegExp(tokenPattern, 'gm')
  const parts: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = pattern.exec(code)) !== null) {
    if (match.index > lastIndex) parts.push(code.slice(lastIndex, match.index))
    const value = match[0]
    const tokenClass = classifyToken(value, code, match.index, keywords)
    parts.push(<span className={tokenClass} key={parts.length}>{value}</span>)
    lastIndex = match.index + value.length
  }
  if (lastIndex < code.length) parts.push(code.slice(lastIndex))
  return parts
}

function classifyToken(value: string, source: string, index: number, keywords: string[] | null): string {
  if (value.startsWith('#') || value.startsWith('//')) return 'syntax-comment'
  if (value.startsWith('"') || value.startsWith("'") || value.startsWith('`')) return 'syntax-string'
  if (isKeyword(value, keywords)) return 'syntax-keyword'
  if (value === 'true' || value === 'false' || value === 'True' || value === 'False') return 'syntax-boolean'
  if (value === 'null' || value === 'None') return 'syntax-null'
  if (/^-?\d/.test(value)) return 'syntax-number'
  if (/^[{}()[\]<>.=:+\-*/%,;!?|&]+$/.test(value)) return 'syntax-operator'
  if (isFunctionIdentifier(value, source, index)) return 'syntax-function'
  if (isTypeIdentifier(value, source, index)) return 'syntax-type'
  return 'syntax-variable'
}

function isFunctionIdentifier(value: string, source: string, index: number): boolean {
  if (!/^[$A-Z_a-z][$\w]*$/.test(value)) return false
  const after = source.slice(index + value.length)
  return /^\s*(?:<[^>]+>)?\s*\(/.test(after)
}

function isTypeIdentifier(value: string, source: string, index: number): boolean {
  if (!/^[$A-Z_a-z][$\w]*$/.test(value)) return false
  if (/^[A-Z]/.test(value)) return true
  const before = source.slice(0, index).trimEnd()
  return /[:<]\s*$/.test(before)
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
