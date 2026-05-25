import { Activity, CheckCircle2, ChevronDown, ChevronRight, Loader2, XCircle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { DiffBlock, ToolCallBlock as ToolCall, ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { ToolCallBlock } from './ToolCallBlock'

type Props = {
  toolCalls: ToolCall[]
  results: Map<string, ToolResult>
  diffs: Map<string, DiffBlock[]>
}

export function ToolCallGroup({ toolCalls, results, diffs }: Props) {
  if (toolCalls.length === 1) {
    const toolCall = toolCalls[0]!
    return (
      <ToolCallBlock
        block={toolCall}
        result={results.get(toolCall.toolCallId)}
        diffs={diffs.get(toolCall.toolCallId) ?? []}
      />
    )
  }

  return <ToolCallGroupMulti toolCalls={toolCalls} results={results} diffs={diffs} />
}

function ToolCallGroupMulti({ toolCalls, results, diffs }: Props) {
  const status = activityStatus(toolCalls, results)
  const diffCount = toolCalls.reduce((count, toolCall) => count + (diffs.get(toolCall.toolCallId)?.length ?? 0), 0)
  const [expanded, setExpanded] = useState(status === 'running' || diffCount > 0)
  const summary = useMemo(() => activitySummary(toolCalls, results, diffCount), [diffCount, results, toolCalls])
  const StatusIcon = statusIcon(status)

  useEffect(() => {
    if (status === 'running' || diffCount > 0) setExpanded(true)
  }, [diffCount, status])

  return (
    <section className={'tool-call-group tool-activity tool-activity-' + status} aria-label="Tool activity">
      <button
        type="button"
        className="tool-activity-header"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Activity size={14} />
        <strong>Activity</strong>
        <span>{summary}</span>
        <em>
          <StatusIcon size={13} />
          <span className={status === 'running' ? 'aether-shimmer-text' : undefined}>{status}</span>
        </em>
      </button>
      {expanded ? (
        <div className="tool-activity-body">
          {toolCalls.map((toolCall) => (
            <ToolCallBlock
              block={toolCall}
              key={toolCall.id}
              result={results.get(toolCall.toolCallId)}
              diffs={diffs.get(toolCall.toolCallId) ?? []}
            />
          ))}
        </div>
      ) : null}
    </section>
  )
}

function activityStatus(toolCalls: ToolCall[], results: Map<string, ToolResult>): 'running' | 'failed' | 'finished' {
  if (toolCalls.some((toolCall) => results.get(toolCall.toolCallId)?.isError || toolCall.status === 'failed')) return 'failed'
  if (toolCalls.some((toolCall) => !results.has(toolCall.toolCallId) && (toolCall.status === 'running' || toolCall.status === 'pending'))) return 'running'
  return 'finished'
}

function statusIcon(status: 'running' | 'failed' | 'finished') {
  if (status === 'running') return Loader2
  if (status === 'failed') return XCircle
  return CheckCircle2
}

function activitySummary(toolCalls: ToolCall[], results: Map<string, ToolResult>, diffCount: number): string {
  const resultCount = toolCalls.filter((toolCall) => results.has(toolCall.toolCallId)).length
  const actionSummary = summarizeToolActions(toolCalls)
  const parts = [
    actionSummary || toolCalls.length + ' tool' + (toolCalls.length === 1 ? '' : 's'),
    resultCount > 0 ? resultCount + ' result' + (resultCount === 1 ? '' : 's') : '',
    diffCount > 0 ? diffCount + ' diff' + (diffCount === 1 ? '' : 's') : '',
  ].filter(Boolean)
  return parts.join(' · ')
}

function summarizeToolActions(toolCalls: ToolCall[]): string {
  const counts = new Map<string, number>()
  for (const toolCall of toolCalls) {
    const key = toolActionKey(toolCall.toolName)
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return Array.from(counts.entries())
    .map(([key, count]) => toolActionPhrase(key, count))
    .join(', ')
}

function toolActionKey(toolName: string): string {
  const normalized = toolName.toLowerCase()
  if (['read', 'read_file', 'view_file'].includes(normalized)) return 'read'
  if (['write', 'write_file', 'create_file'].includes(normalized)) return 'write'
  if (['edit', 'file_edit', 'replace', 'apply_patch'].includes(normalized)) return 'edit'
  if (['bash', 'shell', 'exec_command'].includes(normalized)) return 'shell'
  if (['grep', 'rg', 'search', 'search_files'].includes(normalized)) return 'search'
  if (['glob', 'list_dir', 'ls'].includes(normalized)) return 'list'
  if (['task', 'agent', 'spawn_agent'].includes(normalized)) return 'agent'
  if (['web_search', 'search_web'].includes(normalized)) return 'web_search'
  if (['web_fetch', 'fetch_url'].includes(normalized)) return 'web_fetch'
  return toolName
}

function toolActionPhrase(key: string, count: number): string {
  switch (key) {
    case 'read':
      return count === 1 ? 'read 1 file' : `read ${count} files`
    case 'write':
      return count === 1 ? 'created 1 file' : `created ${count} files`
    case 'edit':
      return count === 1 ? 'edited 1 file' : `edited ${count} files`
    case 'shell':
      return count === 1 ? 'ran 1 command' : `ran ${count} commands`
    case 'search':
      return count === 1 ? 'searched once' : `searched ${count} times`
    case 'list':
      return count === 1 ? 'listed files' : `listed files ${count} times`
    case 'agent':
      return count === 1 ? 'ran 1 subagent' : `ran ${count} subagents`
    case 'web_search':
      return count === 1 ? 'searched the web' : `searched the web ${count} times`
    case 'web_fetch':
      return count === 1 ? 'fetched 1 page' : `fetched ${count} pages`
    default:
      return count === 1 ? key : `${key} (${count})`
  }
}
