import { Bot, ExternalLink, FileText, Globe, Search } from 'lucide-react'
import type { ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { CodeBlock } from './CodeBlock'

type Props = {
  block: ToolResult
  toolArguments?: Record<string, unknown>
}

type PreviewProps = {
  block: ToolResult
  toolArguments: Record<string, unknown>
}

type SearchMatch = {
  path?: string | null
  line?: number | null
  text: string
}

type WebResult = {
  title: string
  url: string | null
  snippet: string | null
}

export function canPreviewToolResult(block: ToolResult): boolean {
  const toolName = (block.toolName || '').toLowerCase()
  if (isReadTool(toolName) || isTaskTool(toolName)) return true
  if (isSearchTool(toolName)) return parseSearchMatches(block.content).length > 0
  if (isWebTool(toolName)) return parseWebResults(block.content).length > 0
  return false
}

export function ToolResultPreview({ block, toolArguments = {} }: Props) {
  const toolName = (block.toolName || '').toLowerCase()
  if (isReadTool(toolName)) return <ReadFilePreview block={block} toolArguments={toolArguments} />
  if (isSearchTool(toolName)) return <SearchPreview content={block.content} />
  if (isWebTool(toolName)) return <WebPreview content={block.content} toolName={block.toolName || 'web'} />
  if (isTaskTool(toolName)) return <TaskPreview block={block} toolArguments={toolArguments} />
  return null
}

function ReadFilePreview({ block, toolArguments }: PreviewProps) {
  const path = stringValue(toolArguments.path) || stringValue(toolArguments.file_path) || stringValue(toolArguments.filePath) || stringValue(block.metadata.path)
  const language = languageFromPath(path) || stringValue(block.metadata.language) || stringValue(block.metadata.lang) || 'text'
  return (
    <section className="tool-preview tool-preview-file" aria-label="File preview">
      <header>
        <span><FileText size={14} /><strong>{path || 'File content'}</strong></span>
        <em>{language}</em>
      </header>
      <CodeBlock code={block.content} language={language} wrap />
    </section>
  )
}

function SearchPreview({ content }: { content: string }) {
  const matches = parseSearchMatches(content)
  if (matches.length === 0) return null
  return (
    <section className="tool-preview tool-preview-search" aria-label="Search results">
      <header>
        <span><Search size={14} /><strong>Search results</strong></span>
        <em>{matches.length.toLocaleString()} matches</em>
      </header>
      <ol className="tool-preview-list">
        {matches.slice(0, 30).map((match, index) => (
          <li key={index}>
            <span className="tool-preview-location">
              {match.path ? <strong>{match.path}</strong> : null}
              {typeof match.line === 'number' ? <em>:{match.line}</em> : null}
            </span>
            <code>{match.text}</code>
          </li>
        ))}
      </ol>
      {matches.length > 30 ? <p className="tool-preview-note">Showing first 30 matches.</p> : null}
    </section>
  )
}

function WebPreview({ content, toolName }: { content: string; toolName: string }) {
  const results = parseWebResults(content)
  if (results.length === 0) return null
  return (
    <section className="tool-preview tool-preview-web" aria-label="Web results">
      <header>
        <span><Globe size={14} /><strong>{toolName}</strong></span>
        <em>{results.length.toLocaleString()} results</em>
      </header>
      <ol className="tool-preview-web-list">
        {results.slice(0, 8).map((result, index) => (
          <li key={index}>
            {result.url ? (
              <a href={result.url} target="_blank" rel="noreferrer">
                <strong>{result.title}</strong>
                <ExternalLink size={12} />
              </a>
            ) : (
              <strong>{result.title}</strong>
            )}
            {result.snippet ? <p>{result.snippet}</p> : null}
            {result.url ? <small>{result.url}</small> : null}
          </li>
        ))}
      </ol>
    </section>
  )
}

function TaskPreview({ block, toolArguments }: PreviewProps) {
  const prompt = stringValue(toolArguments.prompt) || stringValue(toolArguments.description) || stringValue(toolArguments.task) || stringValue(block.metadata.prompt)
  const model = stringValue(toolArguments.model) || stringValue(block.metadata.model)
  const status = block.isError ? 'failed' : stringValue(block.metadata.status) || 'completed'
  return (
    <section className={'tool-preview tool-preview-task tool-preview-task-' + status} aria-label="Subagent result">
      <header>
        <span><Bot size={14} /><strong>{prompt || 'Subagent task'}</strong></span>
        <em>{status}</em>
      </header>
      {model ? <p className="tool-preview-note">Model: {model}</p> : null}
      {block.content ? <pre className="tool-preview-task-output"><code>{block.content}</code></pre> : null}
    </section>
  )
}

function parseSearchMatches(content: string): SearchMatch[] {
  return content
    .split('\n')
    .map((line) => line.trimEnd())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^(.+?):(\d+)(?::\d+)?:\s?(.*)$/)
      if (match) return { path: match[1], line: Number(match[2]), text: match[3] || '' }
      return { text: line }
    })
    .filter((match) => match.text.trim().length > 0)
}

function parseWebResults(content: string): WebResult[] {
  const parsed = parseJson(content)
  const candidates = Array.isArray(parsed)
    ? parsed
    : isRecord(parsed)
      ? arrayValue(parsed.results) || arrayValue(parsed.items) || arrayValue(parsed.search_results) || arrayValue(parsed.data) || []
      : []
  const results: WebResult[] = []
  for (const item of candidates) {
    if (!isRecord(item)) continue
    const title = stringValue(item.title) || stringValue(item.name) || stringValue(item.url)
    if (!title) continue
    results.push({
      title,
      url: stringValue(item.url) || stringValue(item.link),
      snippet: stringValue(item.snippet) || stringValue(item.description) || stringValue(item.content),
    })
  }
  return results
}

function parseJson(content: string): unknown {
  try {
    return JSON.parse(content)
  } catch {
    return null
  }
}

function isReadTool(toolName: string) {
  return ['read', 'read_file', 'view_file', 'open_file'].includes(toolName)
}

function isSearchTool(toolName: string) {
  return ['grep', 'rg', 'search', 'search_files', 'find'].includes(toolName)
}

function isWebTool(toolName: string) {
  return ['web_search', 'search_web', 'web_fetch', 'fetch_url'].includes(toolName)
}

function isTaskTool(toolName: string) {
  return ['task', 'agent', 'spawn_agent', 'delegate_task'].includes(toolName)
}

function languageFromPath(path?: string | null): string | null {
  if (!path) return null
  const ext = path.split('.').pop()?.toLowerCase()
  if (!ext || ext === path) return null
  if (ext === 'py') return 'python'
  if (ext === 'js') return 'javascript'
  if (ext === 'ts') return 'typescript'
  if (ext === 'tsx') return 'tsx'
  if (ext === 'jsx') return 'jsx'
  if (ext === 'md') return 'markdown'
  if (ext === 'sh') return 'bash'
  return ext
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null
}

function arrayValue(value: unknown): unknown[] | null {
  return Array.isArray(value) ? value : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
