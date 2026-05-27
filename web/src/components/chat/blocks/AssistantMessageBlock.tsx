import { Bot } from 'lucide-react'
import type { AssistantMessageBlock as AssistantMessage } from '../../../chat-rendering'
import { MarkdownRenderer } from '../MarkdownRenderer'
import { MessageActionBar } from '../MessageActionBar'

type Props = {
  block: AssistantMessage
  actionsDisabled?: boolean
  onFork?: (block: AssistantMessage) => void
  onQuote?: (block: AssistantMessage) => void
  onRetry?: (block: AssistantMessage) => void
  onRewind?: (block: AssistantMessage) => void
}

export function AssistantMessageBlock({ block, actionsDisabled = false, onFork, onQuote, onRetry, onRewind }: Props) {
  if (!block.content.trim()) return null
  const documentLayout = shouldUseDocumentLayout(block.content)
  const showActions = !block.isStreaming
  const persisted = block.source === 'transcript' && typeof block.messageIndex === 'number'
  return (
    <article className={'chat-block chat-block-assistant chat-message-group' + (block.isError ? ' chat-block-error' : '')}>
      <div className="chat-block-label">
        <span className="chat-role-icon" aria-hidden="true"><Bot size={13} /></span>
        <span>assistant</span>
      </div>
      <div className={documentLayout ? 'chat-message-document' : 'chat-message-shell'}>
        <MarkdownRenderer text={block.content} streaming={Boolean(block.isStreaming)} />
      </div>
      <MessageActionBar
        copyText={showActions ? block.content : undefined}
        copyLabel="Copy reply"
        actions={showActions ? [
          ...(onQuote ? [{ kind: 'quote' as const, label: 'Quote reply', onClick: () => onQuote(block), disabled: actionsDisabled }] : []),
          ...(onRetry && persisted ? [{ kind: 'retry' as const, label: 'Retry reply', onClick: () => onRetry(block), disabled: actionsDisabled }] : []),
          ...(onRewind && persisted ? [{ kind: 'rewind' as const, label: 'Rewind to reply', onClick: () => onRewind(block), disabled: actionsDisabled }] : []),
          ...(onFork && persisted ? [{ kind: 'fork' as const, label: 'Fork from reply', onClick: () => onFork(block), disabled: actionsDisabled }] : []),
        ] : []}
      />
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
