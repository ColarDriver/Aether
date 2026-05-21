import type { StreamingStatusBlock as StreamingStatus } from '../../../chat-rendering'

type Props = {
  block: StreamingStatus
}

export function StreamingStatusBlock({ block }: Props) {
  const outputTokens = block.tokens?.output_tokens ?? 0
  return (
    <div className="chat-block chat-block-status" role="status">
      <span className="status-spark">✦</span>
      <span>{statusLabel(block.state)}</span>
      {block.detail ? <span className="muted">{block.detail}</span> : null}
      {outputTokens > 0 ? <span className="muted">· {outputTokens} out</span> : null}
    </div>
  )
}

function statusLabel(state: string): string {
  if (state === 'tool_use') return 'Running tool'
  if (state === 'responding') return 'Responding'
  if (state === 'thinking') return 'Thinking'
  return state || 'Working'
}
