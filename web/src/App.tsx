import { Activity, Bot, Boxes, Brain, CircleAlert, Settings, ShieldCheck, Wrench } from 'lucide-react'
import { useEffect } from 'react'
import { useAppStore } from './stores/appStore'
import { useProviderStore } from './stores/providerStore'
import { useSessionStore } from './stores/sessionStore'
import { Sidebar } from './components/layout/Sidebar'
import { StatusBar } from './components/layout/StatusBar'
import { TopBar } from './components/layout/TopBar'
import { EmptyState } from './components/shared/EmptyState'
import { Spinner } from './components/shared/Spinner'

export function App() {
  const { status, health, activeView, isLoading, error, bootstrap, setActiveView } = useAppStore()
  const { sessions, activeSessionId, isLoading: sessionsLoading, createSession, setActiveSession } = useSessionStore()
  const { current, providers, loadProviders } = useProviderStore()

  useEffect(() => {
    void bootstrap()
    void useSessionStore.getState().loadSessions()
    void loadProviders()
  }, [bootstrap, loadProviders])

  const activeSession = sessions.find((session) => session.session_id === activeSessionId) ?? null

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        activeView={activeView}
        onSelectSession={(id) => {
          setActiveSession(id)
          setActiveView('chat')
        }}
        onSelectView={setActiveView}
        onNewSession={() => {
          const provider = current?.provider_name || providers[0]?.name || 'openai'
          const model = current?.model || 'gpt-5.4'
          void createSession({ provider, model })
        }}
      />
      <main className="workspace">
        <TopBar
          title={viewTitle(activeView)}
          status={status?.ok ? 'online' : 'offline'}
          provider={current?.provider_name}
          model={current?.model}
        />
        <section className="content-pane">
          {isLoading || sessionsLoading ? <Spinner label="Loading console" /> : null}
          {error ? (
            <div className="notice notice-error">
              <CircleAlert size={16} />
              <span>{error}</span>
            </div>
          ) : null}
          {activeView === 'chat' ? (
            <ChatPlaceholder sessionTitle={activeSession?.summary || activeSession?.session_id || null} />
          ) : null}
          {activeView === 'models' ? (
            <InfoGrid
              items={[
                ['Provider', current?.provider_name || 'unconfigured'],
                ['Family', current?.family || '-'],
                ['Model', current?.model || '-'],
                ['Credential', current?.credential?.configured ? 'configured' : 'missing'],
              ]}
            />
          ) : null}
          {activeView === 'tools' ? <CatalogPlaceholder icon={<Wrench />} title="Tool catalog" /> : null}
          {activeView === 'skills' ? <CatalogPlaceholder icon={<Brain />} title="Skill catalog" /> : null}
          {activeView === 'diagnostics' ? (
            <InfoGrid
              items={[
                ['Health', health?.status || 'unknown'],
                ['Diagnostics', health?.diagnostics?.enabled ? 'enabled' : 'disabled'],
                ['Runtime', health?.runtime?.python_version || '-'],
              ]}
            />
          ) : null}
          {activeView === 'settings' ? <CatalogPlaceholder icon={<Settings />} title="Settings" /> : null}
        </section>
        <StatusBar
          health={health?.status || 'unknown'}
          services={health?.services?.length ?? 0}
          sessions={sessions.length}
          activeSession={activeSession?.session_id ?? null}
        />
      </main>
    </div>
  )
}

function ChatPlaceholder({ sessionTitle }: { sessionTitle: string | null }) {
  if (!sessionTitle) {
    return (
      <EmptyState
        icon={<Bot />}
        title="No session selected"
        description="Create or select a session to start using the browser console."
      />
    )
  }
  return (
    <div className="chat-surface">
      <div className="chat-header">
        <Bot size={18} />
        <div>
          <div className="chat-title">{sessionTitle}</div>
          <div className="muted">Transcript and streaming run UI lands in PR20.5.</div>
        </div>
      </div>
      <div className="composer-placeholder">Message composer reserved for chat implementation</div>
    </div>
  )
}

function CatalogPlaceholder({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <EmptyState
      icon={icon}
      title={title}
      description="The REST API is available; the detailed browser view lands in the next UI slice."
    />
  )
}

function InfoGrid({ items }: { items: Array<[string, string]> }) {
  return (
    <div className="info-grid">
      {items.map(([label, value]) => (
        <div className="info-row" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
      <div className="info-row">
        <span>Surface</span>
        <strong>
          <ShieldCheck size={15} /> service-backed
        </strong>
      </div>
    </div>
  )
}

function viewTitle(view: string) {
  const titles: Record<string, string> = {
    chat: 'Chat',
    models: 'Models',
    tools: 'Tools',
    skills: 'Skills',
    diagnostics: 'Diagnostics',
    settings: 'Settings',
  }
  return titles[view] || 'Aether'
}

export const navItems = [
  { id: 'chat', label: 'Chat', icon: Activity },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'skills', label: 'Skills', icon: Brain },
  { id: 'diagnostics', label: 'Diagnostics', icon: ShieldCheck },
  { id: 'settings', label: 'Settings', icon: Settings },
] as const
