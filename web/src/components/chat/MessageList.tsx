import type { ChatMessage } from '../../stores/chatStore'

type Props = {
  messages: ChatMessage[]
}

export function MessageList({ messages }: Props) {
  if (messages.length === 0) {
    return <div className="empty-chat">No messages in this session yet.</div>
  }
  return (
    <div className="message-list">
      {messages.map((message) => (
        <article
          key={message.id}
          className={`message message-${message.role}${message.isError ? ' message-error' : ''}`}
        >
          <div className="message-role">{message.role}</div>
          <pre>{message.text}</pre>
          {message.isStreaming ? <span className="streaming-caret" /> : null}
        </article>
      ))}
    </div>
  )
}
