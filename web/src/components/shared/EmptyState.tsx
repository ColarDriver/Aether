import type { ReactNode } from 'react'

type Props = {
  icon: ReactNode
  title: string
  description: string
}

export function EmptyState({ icon, title, description }: Props) {
  return (
    <div className="empty-state">
      <div className="empty-icon">{icon}</div>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  )
}
