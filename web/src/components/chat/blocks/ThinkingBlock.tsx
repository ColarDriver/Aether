import { useState } from 'react'
import type { ThinkingBlock as Thinking } from '../../../chat-rendering'
import { firstNonEmptyLine } from '../../../chat-rendering'

type Props = {
  block: Thinking
}

export function ThinkingBlock({ block }: Props) {
  const [expanded, setExpanded] = useState(false)
  const preview = firstNonEmptyLine(block.content)
  return (
    <article className="chat-block chat-block-thinking">
      <button type="button" className="thinking-toggle" onClick={() => setExpanded((value) => !value)}>
        <span>{expanded ? '▾' : '▸'}</span>
        <strong>thinking{block.isActive ? '...' : ''}</strong>
        {!expanded && preview ? <span className="thinking-preview">{preview}</span> : null}
      </button>
      {expanded ? <pre className="thinking-content">{block.content}</pre> : null}
    </article>
  )
}
