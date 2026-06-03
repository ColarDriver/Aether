import type { UserMessageBlock as UserMessage } from '../../../chat-rendering'
import { AttachmentGallery } from '../AttachmentGallery'
import { MessageActionBar } from '../MessageActionBar'
import { MessageMeta } from './MessageMeta'

type Props = {
  block: UserMessage
  actionsDisabled?: boolean
  onEdit?: (block: UserMessage) => void
  onRetry?: (block: UserMessage) => void
}

export function UserMessageBlock({ block, actionsDisabled = false, onEdit, onRetry }: Props) {
  const hasText = Boolean(block.content.trim())
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
          ...(hasText && onEdit ? [{ kind: 'edit' as const, label: 'Edit prompt', onClick: () => onEdit(block), disabled: actionsDisabled }] : []),
          ...((hasText || (block.attachments?.length ?? 0) > 0) && onRetry ? [{ kind: 'retry' as const, label: 'Retry prompt', onClick: () => onRetry(block), disabled: actionsDisabled }] : []),
        ]}
      />
    </article>
  )
}
