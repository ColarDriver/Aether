import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { SkillSummary } from '../../api/types'
import { Spinner } from '../shared/Spinner'

export function SkillsView() {
  const [skills, setSkills] = useState<SkillSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    api.skills()
      .then((result) => setSkills(result.skills))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="settings-panel">
      <header className="panel-header"><div><h2>Skills</h2><p>Local skills discoverable by the runtime.</p></div></header>
      {loading ? <Spinner label="Loading skills" /> : null}
      {error ? <div className="notice notice-error">{error}</div> : null}
      <div className="catalog-list two-column-list">
        {skills.map((skill) => (
          <section className="catalog-card" key={skill.name}>
            <div className="catalog-card-header"><strong>{skill.name}</strong><span>{skill.source.source}</span></div>
            <p>{skill.description || skill.when_to_use || 'No description'}</p>
            {skill.source.path ? <small>{skill.source.path}</small> : null}
          </section>
        ))}
        {skills.length === 0 && !loading ? <div className="muted">No skills discovered.</div> : null}
      </div>
    </div>
  )
}
