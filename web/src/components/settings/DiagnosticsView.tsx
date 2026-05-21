import type { HealthStatus } from '../../api/types'

type Props = {
  health: HealthStatus | null
}

export function DiagnosticsView({ health }: Props) {
  return (
    <div className="settings-panel">
      <header className="panel-header"><div><h2>Diagnostics</h2><p>Runtime health, provider auth readiness, and service availability.</p></div></header>
      <div className="info-grid compact-grid">
        <div className="info-row"><span>Overall</span><strong>{health?.status || 'unknown'}</strong></div>
        <div className="info-row"><span>Python</span><strong>{health?.runtime.python_version || '-'}</strong></div>
        <div className="info-row"><span>Implementation</span><strong>{health?.runtime.implementation || '-'}</strong></div>
        <div className="info-row"><span>Diagnostics</span><strong>{health?.diagnostics?.enabled ? 'enabled' : 'disabled'}</strong></div>
      </div>
      <div className="catalog-list">
        {(health?.services ?? []).map((service) => (
          <section className="catalog-card status-card" key={service.name}>
            <div className="catalog-card-header"><strong>{service.name}</strong><span>{service.available ? 'available' : 'unavailable'}</span></div>
            <p>{service.detail || service.status}</p>
          </section>
        ))}
      </div>
    </div>
  )
}
