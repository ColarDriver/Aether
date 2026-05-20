type Props = {
  title: string
  status: 'online' | 'offline'
  provider?: string
  model?: string
}

export function TopBar({ title, status, provider, model }: Props) {
  return (
    <header className="top-bar">
      <div>
        <h1>{title}</h1>
        <p>{provider && model ? `${provider} / ${model}` : 'Provider not loaded'}</p>
      </div>
      <div className={`status-pill ${status === 'online' ? 'status-pill-online' : ''}`}>
        <span />
        {status}
      </div>
    </header>
  )
}
