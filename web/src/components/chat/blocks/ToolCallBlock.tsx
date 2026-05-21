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
  const hasDetails = Object.keys(block.arguments).length > 0 || Boolean(result) || diffs.length > 0
  const detailsVisible = expanded || Boolean(result) || diffs.length > 0
  const summary = toolSummary(block)
  const ToolIcon = toolIconForName(block.toolName)
  return (
    <article className={'tool-call-block tool-call-' + block.status}>
      <button type="button" className="tool-call-header" onClick={() => setExpanded((value) => !value)}>
        {hasDetails ? detailsVisible ? <ChevronDown size={14} /> : <ChevronRight size={14} /> : <Wrench size={14} />}
        <span className="tool-call-kind-icon" aria-hidden="true"><ToolIcon size={14} /></span>
        <strong>{block.toolName}</strong>
        {summary ? <span className="tool-call-summary">{summary}</span> : null}
        <em>{block.status}</em>
      </button>
      {detailsVisible ? (
        <div className="tool-call-body">
          {todos.length > 0 ? <TodoListPreview todos={todos} /> : null}
          {Object.keys(block.arguments).length > 0 && todos.length === 0 ? (
            <CodeBlock code={JSON.stringify(block.arguments, null, 2)} language="json" title="Input" />
          ) : null}
          {diffs.map((diff) => <DiffBlock block={diff} key={diff.id} />)}
          {result ? <ToolResultBlock block={result} /> : null}
        </div>
      ) : null}
    </article>
  )
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

function toolIconForName(toolName: string) {
  const normalized = toolName.toLowerCase()
  if (['bash', 'shell', 'exec_command'].includes(normalized)) return Terminal
  if (['read', 'read_file', 'view_file'].includes(normalized)) return FileText
  if (['write', 'write_file', 'create_file', 'edit', 'file_edit', 'replace', 'apply_patch'].includes(normalized)) return Pencil
  if (['grep', 'rg', 'search', 'search_files'].includes(normalized)) return Search
  if (['task', 'agent', 'spawn_agent'].includes(normalized)) return Bot
  if (['web_search', 'search_web', 'web_fetch', 'fetch_url'].includes(normalized)) return Globe
  return Wrench
}
