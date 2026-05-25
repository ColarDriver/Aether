import { Brain, Loader2, MessageSquareText, Wrench } from 'lucide-react'
import type { StreamingStatusBlock as StreamingStatus } from '../../../chat-rendering'
import { tokenUsageSummary } from '../../../chat-rendering'

type Props = {
  block: StreamingStatus
}

export function StreamingStatusBlock({ block }: Props) {
  const tokenSummary = tokenUsageSummary(block.tokens)
  const Icon = statusIcon(block.state)
  return (
    <div className={'chat-block chat-block-status chat-block-status-' + statusTone(block.state)} role="status">
      <span className="status-icon" aria-hidden="true">
        <Icon size={14} />
      </span>
      <strong className="aether-shimmer-text">{statusLabel(block.state)}</strong>
      {block.detail ? <span className="status-detail">{block.detail}</span> : null}
      {tokenSummary ? <span className="status-meta">{tokenSummary}</span> : null}
    </div>
  )
}

function statusLabel(state: string): string {
  if (state === 'tool_use') return 'Running tool'
  if (state === 'responding') return 'Responding'
  if (state === 'thinking') return 'Thinking'
  return state || 'Working'
}

function statusIcon(state: string) {
  if (state === 'thinking') return Brain
  if (state === 'tool_use') return Wrench
  if (state === 'responding') return MessageSquareText
  return Loader2
}

function statusTone(state: string): string {
  if (state === 'tool_use') return 'tool'
  if (state === 'thinking') return 'thinking'
  if (state === 'responding') return 'responding'
  return 'working'
}
