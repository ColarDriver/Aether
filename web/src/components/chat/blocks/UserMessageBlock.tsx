import type { UserMessageBlock as UserMessage } from '../../../chat-rendering'
import { AttachmentGallery } from '../AttachmentGallery'
import { MessageActionBar } from '../MessageActionBar'

type Props = {
  block: UserMessage
}

export function UserMessageBlock({ block }: Props) {
  return (
    <article className="chat-block chat-block-user chat-message-group">
      <div className="chat-block-label">user</div>
      <AttachmentGallery attachments={block.attachments} align="end" />
      {block.content.trim() ? <pre className="chat-user-text">{block.content.trim()}</pre> : null}
      {block.pending ? <span className="chat-state-pill">pending</span> : null}
      <MessageActionBar copyText={block.content} copyLabel="Copy prompt" align="end" />
    </article>
  )
}
