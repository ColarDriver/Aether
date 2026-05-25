import { Brain, ChevronDown, ChevronRight, Loader2 } from 'lucide-react'
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
    <article className={block.isActive ? 'chat-block chat-block-thinking chat-block-thinking-active' : 'chat-block chat-block-thinking'}>
      <button
        type="button"
        className="thinking-toggle"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span className="thinking-icon" aria-hidden="true">
          {block.isActive ? <Loader2 size={14} /> : <Brain size={14} />}
        </span>
        <strong className={block.isActive ? 'aether-shimmer-text' : undefined}>thinking</strong>
        {!expanded && preview ? <span className="thinking-preview">{preview}</span> : null}
        <em>{block.isActive ? 'active' : 'saved'}</em>
      </button>
      {expanded ? <pre className="thinking-content">{block.content}</pre> : null}
    </article>
  )
}
