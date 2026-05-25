import { Bot, ChevronDown, ChevronRight, FileText, Globe, Pencil, Search, Terminal, Wrench } from 'lucide-react'
import { useState } from 'react'
import type { DiffBlock as DiffChatBlock, ToolCallBlock as ToolCall, ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { DiffBlock } from './DiffBlock'
import { ToolResultBlock } from './ToolResultBlock'
import { CodeBlock } from './CodeBlock'
import { TodoListPreview, todosFromToolArguments } from './TodoListPreview'

type Props = {
  block: ToolCall
  result?: ToolResult | null
  diffs?: DiffChatBlock[]
}

export function ToolCallBlock({ block, result, diffs = [] }: Props) {
  const [expanded, setExpanded] = useState(false)
  const todos = block.toolName === 'todo_write' ? todosFromToolArguments(block.arguments) : []
  const showRawInput = Object.keys(block.arguments).length > 0 && todos.length === 0 && !isTerminalTool(block.toolName)
  const hasDetails = showRawInput || Boolean(result) || diffs.length > 0
  const detailsVisible = expanded || Boolean(result) || diffs.length > 0
  const summary = toolSummary(block)
  const status = effectiveToolStatus(block, result)
  const meta = toolCallMeta(result)
  const ToolIcon = toolIconForName(block.toolName)
  return (
    <article className={'tool-call-block tool-call-' + status}>
      <button type="button" className="tool-call-header" onClick={() => setExpanded((value) => !value)}>
        {hasDetails ? detailsVisible ? <ChevronDown size={14} /> : <ChevronRight size={14} /> : <Wrench size={14} />}
        <span className="tool-call-kind-icon" aria-hidden="true"><ToolIcon size={14} /></span>
        <strong className={isRunningToolCall(status) ? 'aether-shimmer-text' : undefined}>{block.toolName}</strong>
        {summary ? <span className="tool-call-summary">{summary}</span> : null}
        {meta.length > 0 ? <span className="tool-call-meta">{meta.join(' · ')}</span> : null}
        <em>{status}</em>
      </button>
      {detailsVisible ? (
        <div className="tool-call-body">
          {todos.length > 0 ? <TodoListPreview todos={todos} /> : null}
          {showRawInput ? (
            <CodeBlock code={JSON.stringify(block.arguments, null, 2)} language="json" title="Input" />
          ) : null}
          {diffs.map((diff) => <DiffBlock block={diff} key={diff.id} />)}
          {result ? <ToolResultBlock block={result} command={stringArg(block.arguments, 'command')} toolArguments={block.arguments} /> : null}
        </div>
      ) : null}
    </article>
  )
}

function effectiveToolStatus(block: ToolCall, result?: ToolResult | null): ToolCall['status'] {
  if (!result) return block.status
  return result.isError ? 'failed' : 'finished'
}

function isRunningToolCall(status: string): boolean {
  return status === 'running' || status === 'pending'
}

function toolCallMeta(result?: ToolResult | null): string[] {
  if (!result) return []
  const durationMs = numberMetadata(result.metadata, 'duration_ms')
    ?? numberMetadata(result.metadata, 'durationMs')
    ?? secondsToMillis(numberMetadata(result.metadata, 'duration_seconds') ?? numberMetadata(result.metadata, 'durationSeconds'))
  return [durationMs != null ? formatDurationMs(durationMs) : null].filter((item): item is string => Boolean(item))
}

function numberMetadata(metadata: Record<string, unknown>, key: string): number | null {
  const value = metadata[key]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

function secondsToMillis(value: number | null): number | null {
  return value == null ? null : value * 1000
}

function formatDurationMs(ms: number): string {
  const seconds = Math.max(0, ms / 1000)
  if (seconds < 10) return seconds.toFixed(1) + 's'
  if (seconds < 60) return Math.round(seconds) + 's'
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return minutes + 'm ' + rest + 's'
}

function toolSummary(block: ToolCall): string {
  const args = block.arguments
  const path = stringArg(args, 'path') || stringArg(args, 'file_path') || stringArg(args, 'filePath')
  if (path) return path
  const command = stringArg(args, 'command')
  if (command) return command
  const query = stringArg(args, 'query') || stringArg(args, 'pattern') || stringArg(args, 'url')
  if (query) return query
  const description = stringArg(args, 'description') || stringArg(args, 'prompt')
  if (description) return description
  return ''
}

function stringArg(args: Record<string, unknown>, key: string): string {
  return typeof args[key] === 'string' ? args[key] : ''
}

function isTerminalTool(toolName: string): boolean {
  return ['bash', 'shell', 'exec', 'exec_command', 'terminal', 'run_shell'].includes(toolName.toLowerCase())
}

function toolIconForName(toolName: string) {
  const normalized = toolName.toLowerCase()
  if (isTerminalTool(toolName)) return Terminal
  if (['read', 'read_file', 'view_file'].includes(normalized)) return FileText
  if (['write', 'write_file', 'create_file', 'edit', 'file_edit', 'replace', 'apply_patch'].includes(normalized)) return Pencil
  if (['grep', 'rg', 'search', 'search_files'].includes(normalized)) return Search
  if (['task', 'agent', 'spawn_agent'].includes(normalized)) return Bot
  if (['web_search', 'search_web', 'web_fetch', 'fetch_url'].includes(normalized)) return Globe
  return Wrench
}
