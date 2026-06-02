import { Brain, Loader2, MessageSquareText, Wrench } from 'lucide-react'
import type { StreamingStatusBlock as StreamingStatus } from '../../../chat-rendering'
import { spinnerVerbForSeed, tokenUsageSummary } from '../../../chat-rendering'

type Props = {
  block: StreamingStatus
}

export function StreamingStatusBlock({ block }: Props) {
  const tokenSummary = tokenUsageSummary(block.tokens)
  // ↑ while waiting on the model, ↓ once output streams back — accented for emphasis.
  const tokenArrow = block.state === 'requesting' ? '↑' : '↓'
  const detail = friendlyDetail(block.detail)
  const Icon = statusIcon(block.state)
  // Whimsical verb alongside the explicit mode label, seeded by the run/block
  // so it's stable across re-renders but varies per turn (Claude Code style).
  const verb = spinnerVerbForSeed(block.runId ?? block.id)
  return (
    <div className={'chat-block chat-block-status chat-block-status-' + statusTone(block.state)} role="status">
      <span className="status-icon" aria-hidden="true">
        <Icon size={14} />
      </span>
      <span className="status-headline">
        <strong className="aether-shimmer-text">{statusLabel(block.state)}</strong>
        <span className="status-verb">{verb}…</span>
      </span>
      {detail ? <span className="status-detail">{detail}</span> : null}
      {tokenSummary ? (
        <span className="status-meta">
          <span className="status-arrow">{tokenArrow}</span> {tokenSummary}
        </span>
      ) : null}
    </div>
  )
}

// Drop raw, all-caps event codes (e.g. LLM_CALL) that leak through as details;
// humanize snake_case otherwise.
function friendlyDetail(detail?: string | null): string | null {
  if (!detail) return null
  if (/^[A-Z][A-Z0-9_]*$/.test(detail)) return null
  return detail.replace(/_/g, ' ')
}

function statusLabel(state: string): string {
  if (state === 'requesting') return 'Requesting'
  if (state === 'tool_input') return 'Tool input'
  if (state === 'tool_use') return 'Running tool'
  if (state === 'responding') return 'Responding'
  if (state === 'thinking') return 'Thinking'
  return state || 'Working'
}

function statusIcon(state: string) {
  if (state === 'thinking') return Brain
  if (state === 'tool_input' || state === 'tool_use') return Wrench
  if (state === 'responding') return MessageSquareText
  return Loader2
}

function statusTone(state: string): string {
  if (state === 'tool_input' || state === 'tool_use') return 'tool'
  if (state === 'thinking') return 'thinking'
  if (state === 'responding') return 'responding'
  return 'working'
}
