type Props = {
  role: 'user' | 'assistant'
  timestamp: number
}

export function MessageMeta({ role, timestamp }: Props) {
  const label = role === 'user' ? 'You' : 'Aether'
  const initial = role === 'user' ? 'Y' : 'A'
  const time = formatMessageTime(timestamp)
  return (
    <div className="chat-message-meta">
      <span className={'chat-role-avatar chat-role-avatar-' + role} aria-hidden="true">{initial}</span>
      <span className="chat-message-author">{label}</span>
      {time ? <time className="chat-message-time" dateTime={time.iso}>{time.label}</time> : null}
    </div>
  )
}

export function formatMessageTime(timestamp: number, fallbackNow = Date.now()): { label: string; iso: string } | null {
  if (!Number.isFinite(timestamp)) return null
  const millis = timestamp > 1_000_000_000_000
    ? timestamp
    : timestamp > 1_000_000_000
      ? timestamp * 1000
      : fallbackNow
  const date = new Date(millis)
  if (Number.isNaN(date.getTime())) return null
  return {
    label: new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }).format(date).replace(' ', ''),
    iso: date.toISOString(),
  }
}
