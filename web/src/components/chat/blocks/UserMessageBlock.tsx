import type { UserMessageBlock as UserMessage } from '../../../chat-rendering'
import { AttachmentGallery } from '../AttachmentGallery'
import { MessageActionBar } from '../MessageActionBar'
import { MessageMeta } from './MessageMeta'

type Props = {
  block: UserMessage
  actionsDisabled?: boolean
  onEdit?: (block: UserMessage) => void
  onFork?: (block: UserMessage) => void
  onQuote?: (block: UserMessage) => void
  onRewind?: (block: UserMessage) => void
  onRetry?: (block: UserMessage) => void
}

export function UserMessageBlock({ block, actionsDisabled = false, onEdit, onFork, onQuote, onRewind, onRetry }: Props) {
  const hasText = Boolean(block.content.trim())
  const persisted = block.source === 'transcript' && typeof block.messageIndex === 'number'
  return (
    <article className="chat-block chat-block-user chat-message-group">
      <div className="chat-block-label"><MessageMeta role="user" timestamp={block.timestamp} /></div>
      <AttachmentGallery attachments={block.attachments} align="end" />
      {hasText ? <pre className="chat-user-text">{block.content.trim()}</pre> : null}
      {block.pending ? <span className="chat-state-pill">pending</span> : null}
      <MessageActionBar
        copyText={block.content}
        copyLabel="Copy prompt"
        align="end"
        actions={[
          ...(hasText && onQuote ? [{ kind: 'quote' as const, label: 'Quote prompt', onClick: () => onQuote(block), disabled: actionsDisabled }] : []),
          ...(hasText && onEdit ? [{ kind: 'edit' as const, label: 'Edit prompt', onClick: () => onEdit(block), disabled: actionsDisabled }] : []),
          ...((hasText || (block.attachments?.length ?? 0) > 0) && onRetry ? [{ kind: 'retry' as const, label: 'Retry prompt', onClick: () => onRetry(block), disabled: actionsDisabled }] : []),
          ...(onRewind && persisted ? [{ kind: 'rewind' as const, label: 'Rewind to prompt', onClick: () => onRewind(block), disabled: actionsDisabled }] : []),
          ...(onFork && persisted ? [{ kind: 'fork' as const, label: 'Fork from prompt', onClick: () => onFork(block), disabled: actionsDisabled }] : []),
        ]}
      />
    </article>
  )
}
