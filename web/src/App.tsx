import { Activity, Boxes, Brain, CircleAlert, FileText, KeyRound, Settings, ShieldCheck, Wrench } from 'lucide-react'
import { useEffect } from 'react'
import { useAppStore } from './stores/appStore'
import { useProviderStore } from './stores/providerStore'
import { useSessionStore } from './stores/sessionStore'
import { Sidebar } from './components/layout/Sidebar'
import { StatusBar } from './components/layout/StatusBar'
import { TopBar } from './components/layout/TopBar'
import { ChatView } from './components/chat/ChatView'
import { DiagnosticsView } from './components/settings/DiagnosticsView'
import { ProviderSettings } from './components/settings/ProviderSettings'
import { SettingsView } from './components/settings/SettingsView'
import { SkillsView } from './components/settings/SkillsView'
import { LogsView } from './components/settings/LogsView'
import { EnvironmentView } from './components/settings/EnvironmentView'
import { ToolsView } from './components/settings/ToolsView'
import { Spinner } from './components/shared/Spinner'
import { ToastViewport } from './components/shared/ToastViewport'

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
          {activeView === 'chat' ? <ChatView session={activeSession} /> : null}
          {activeView === 'models' ? <ProviderSettings /> : null}
          {activeView === 'tools' ? <ToolsView /> : null}
          {activeView === 'skills' ? <SkillsView /> : null}
          {activeView === 'diagnostics' ? <DiagnosticsView health={health} /> : null}
          {activeView === 'logs' ? <LogsView /> : null}
          {activeView === 'environment' ? <EnvironmentView /> : null}
          {activeView === 'settings' ? <SettingsView /> : null}
        </section>
        <StatusBar
          health={health?.status || 'unknown'}
          services={health?.services?.length ?? 0}
          sessions={sessions.length}
          activeSession={activeSession?.session_id ?? null}
        />
      </main>
      <ToastViewport />
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
    logs: 'Logs',
    environment: 'Environment',
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
  { id: 'logs', label: 'Logs', icon: FileText },
  { id: 'environment', label: 'Environment', icon: KeyRound },
  { id: 'settings', label: 'Settings', icon: Settings },
] as const
