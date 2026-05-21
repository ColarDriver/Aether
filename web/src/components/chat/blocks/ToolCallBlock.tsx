import { ChevronDown, ChevronRight, Wrench } from 'lucide-react'
import { useState } from 'react'
import type { DiffBlock as DiffChatBlock, ToolCallBlock as ToolCall, ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { DiffBlock } from './DiffBlock'
import { ToolResultBlock } from './ToolResultBlock'
import { CodeBlock } from './CodeBlock'

type Props = {
  block: ToolCall
  result?: ToolResult | null
  diffs?: DiffChatBlock[]
}

export function ToolCallBlock({ block, result, diffs = [] }: Props) {
  const [expanded, setExpanded] = useState(false)
  const hasDetails = Object.keys(block.arguments).length > 0 || Boolean(result) || diffs.length > 0
  const summary = toolSummary(block)
  return (
    <article className={'tool-call-block tool-call-' + block.status}>
      <button type="button" className="tool-call-header" onClick={() => setExpanded((value) => !value)}>
        {hasDetails ? expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} /> : <Wrench size={14} />}
        <strong>{block.toolName}</strong>
        {summary ? <span>{summary}</span> : null}
        <em>{block.status}</em>
      </button>
      {expanded || result || diffs.length > 0 ? (
        <div className="tool-call-body">
          {Object.keys(block.arguments).length > 0 ? (
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
