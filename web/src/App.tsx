import { Activity, BarChart3, BookOpen, Boxes, Brain, CircleAlert, FileText, Folder, KeyRound, MessagesSquare, Settings, ShieldCheck, Wrench } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useAppStore } from './stores/appStore'
import { useAppearanceStore } from "./stores/appearanceStore"
import { useProviderStore } from './stores/providerStore'
import { useSessionStore } from './stores/sessionStore'
import { Sidebar } from './components/layout/Sidebar'
import { StatusBar } from './components/layout/StatusBar'
import { TopBar } from './components/layout/TopBar'
import { ChatView } from './components/chat/ChatView'
import { WorkspaceRail } from './components/chat/WorkspaceRail'
import { DiagnosticsView } from './components/settings/DiagnosticsView'
import { ProviderSettings } from './components/settings/ProviderSettings'
import { SettingsView } from './components/settings/SettingsView'
import { SkillsView } from './components/settings/SkillsView'
import { LogsView } from './components/settings/LogsView'
import { EnvironmentView } from './components/settings/EnvironmentView'
import { DocsView } from './components/settings/DocsView'
import { WorkspaceView } from './components/settings/WorkspaceView'
import { AnalyticsView } from './components/settings/AnalyticsView'
import { SessionsView } from './components/settings/SessionsView'
import { ToolsView } from './components/settings/ToolsView'
import { Spinner } from './components/shared/Spinner'
import { ToastViewport } from './components/shared/ToastViewport'

export function App() {
  const { status, health, activeView, isLoading, error, bootstrap, setActiveView } = useAppStore()
  const { sessions, activeSessionId, isLoading: sessionsLoading, createSession, setActiveSession } = useSessionStore()
  const { current, providers, loadProviders } = useProviderStore()
  const bootstrapAppearance = useAppearanceStore((state) => state.bootstrap)
  const [workspaceRailOpen, setWorkspaceRailOpen] = useState(true)

  useEffect(() => {
    void bootstrapAppearance()
    void bootstrap()
    void useSessionStore.getState().loadSessions()
    void loadProviders()
  }, [bootstrap, bootstrapAppearance, loadProviders])

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
      <main className={activeView === 'chat' ? 'workspace workspace-chat' : 'workspace'}>
        <TopBar
          title={activeView === 'chat' ? activeSession?.summary || 'Chat' : viewTitle(activeView)}
          status={status?.ok ? 'online' : 'offline'}
          provider={current?.provider_name}
          model={current?.model}
        />
        <section className={activeView === 'chat' ? 'content-pane content-pane-chat' : 'content-pane'}>
          {isLoading || sessionsLoading ? <Spinner label="Loading console" /> : null}
          {error ? (
            <div className="notice notice-error">
              <CircleAlert size={16} />
              <span>{error}</span>
            </div>
          ) : null}
          {activeView === 'chat' ? (
            <div className={workspaceRailOpen ? 'chat-workbench' : 'chat-workbench chat-workbench-rail-closed'}>
              <ChatView session={activeSession} />
              {workspaceRailOpen ? (
                <WorkspaceRail
                  onClose={() => setWorkspaceRailOpen(false)}
                  onOpenWorkspace={() => setActiveView('workspace')}
                />
              ) : (
                <button
                  type="button"
                  className="workspace-rail-edge"
                  onClick={() => setWorkspaceRailOpen(true)}
                >
                  Files
                </button>
              )}
            </div>
          ) : null}
          {activeView === 'models' ? <ProviderSettings /> : null}
          {activeView === 'sessions' ? <SessionsView /> : null}
          {activeView === 'tools' ? <ToolsView /> : null}
          {activeView === 'skills' ? <SkillsView /> : null}
          {activeView === 'diagnostics' ? <DiagnosticsView health={health} /> : null}
          {activeView === 'logs' ? <LogsView /> : null}
          {activeView === 'analytics' ? <AnalyticsView /> : null}
          {activeView === 'docs' ? <DocsView /> : null}
          {activeView === 'workspace' ? <WorkspaceView /> : null}
          {activeView === 'environment' ? <EnvironmentView /> : null}
          {activeView === 'settings' ? <SettingsView /> : null}
        </section>
        {activeView !== 'chat' ? (
          <StatusBar
            health={health?.status || 'unknown'}
            services={health?.services?.length ?? 0}
            sessions={sessions.length}
            activeSession={activeSession?.session_id ?? null}
          />
        ) : null}
      </main>
      <ToastViewport />
    </div>
  )
}

function viewTitle(view: string) {
  const titles: Record<string, string> = {
    chat: 'Chat',
    models: 'Models',
    sessions: 'Sessions',
    tools: 'Tools',
    skills: 'Skills',
    diagnostics: 'Diagnostics',
    logs: 'Logs',
    analytics: 'Analytics',
    docs: 'Docs',
    workspace: 'Workspace',
    environment: 'Environment',
    settings: 'Settings',
  }
  return titles[view] || 'Aether'
}

export const navItems = [
  { id: 'chat', label: 'Chat', icon: Activity },
  { id: 'sessions', label: 'Sessions', icon: MessagesSquare },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'skills', label: 'Skills', icon: Brain },
  { id: 'diagnostics', label: 'Diagnostics', icon: ShieldCheck },
  { id: 'logs', label: 'Logs', icon: FileText },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'docs', label: 'Docs', icon: BookOpen },
  { id: 'workspace', label: 'Workspace', icon: Folder },
  { id: 'environment', label: 'Environment', icon: KeyRound },
  { id: 'settings', label: 'Settings', icon: Settings },
] as const
