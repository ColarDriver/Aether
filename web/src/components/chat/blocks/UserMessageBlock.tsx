import type { UserMessageBlock as UserMessage } from '../../../chat-rendering'

type Props = {
  block: UserMessage
}

export function UserMessageBlock({ block }: Props) {
  return (
    <article className="chat-block chat-block-user">
      <div className="chat-block-label">user</div>
      <pre className="chat-user-text">{block.content}</pre>
      {block.pending ? <span className="chat-state-pill">pending</span> : null}
    </article>
  )
}
