import type { AssistantMessageBlock as AssistantMessage } from '../../../chat-rendering'
import { MarkdownRenderer } from '../MarkdownRenderer'

type Props = {
  block: AssistantMessage
}

export function AssistantMessageBlock({ block }: Props) {
  if (!block.content.trim()) return null
  const documentLayout = shouldUseDocumentLayout(block.content)
  return (
    <article className={'chat-block chat-block-assistant' + (block.isError ? ' chat-block-error' : '')}>
      <div className="chat-block-label">assistant</div>
      <div className={documentLayout ? 'chat-message-document' : 'chat-message-shell'}>
        <MarkdownRenderer text={block.content} />
        {block.isStreaming ? <span className="streaming-caret" /> : null}
      </div>
    </article>
  )
}

export function shouldUseDocumentLayout(content: string): boolean {
  const normalized = content.trim()
  if (!normalized) return false
  if (/```/.test(normalized)) return true
  if (/^\s{0,3}(#{1,6}\s|[-*+]\s|\d+\.\s|>\s|\|.+\|)/m.test(normalized)) return true
  const paragraphs = normalized.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean)
  return paragraphs.length >= 2 || normalized.split('\n').filter((line) => line.trim()).length >= 8
}
