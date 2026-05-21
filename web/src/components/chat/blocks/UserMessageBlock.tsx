import type { UserMessageBlock as UserMessage } from '../../../chat-rendering'
import { MessageActionBar } from '../MessageActionBar'

type Props = {
  block: UserMessage
}

export function UserMessageBlock({ block }: Props) {
  return (
    <article className="chat-block chat-block-user chat-message-group">
      <div className="chat-block-label">user</div>
      <pre className="chat-user-text">{block.content}</pre>
      {block.pending ? <span className="chat-state-pill">pending</span> : null}
      <MessageActionBar copyText={block.content} copyLabel="Copy prompt" align="end" />
    </article>
  )
}
