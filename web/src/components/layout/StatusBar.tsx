type Props = {
  health: string
  services: number
  sessions: number
  activeSession: string | null
}

export function StatusBar({ health, services, sessions, activeSession }: Props) {
  return (
    <footer className="status-bar">
      <span>health: {health}</span>
      <span>services: {services}</span>
      <span>sessions: {sessions}</span>
      <span>active: {activeSession ? activeSession.slice(0, 8) : 'none'}</span>
    </footer>
  )
}
