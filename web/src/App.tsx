import { CircleAlert } from 'lucide-react'
import type { CSSProperties, KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from './api/client'
import type { WorkspaceFile } from './api/types'
import { useAppStore } from './stores/appStore'
import { useAppearanceStore } from "./stores/appearanceStore"
import { useChatStore } from './stores/chatStore'
import { useProviderStore } from './stores/providerStore'
import { useSessionStore } from './stores/sessionStore'
import { useTaskStore } from './stores/taskStore'
import { useToastStore } from './stores/toastStore'
import { AppRail, SessionSidebar } from './components/layout/Sidebar'
import { StatusBar } from './components/layout/StatusBar'
import { TopBar } from './components/layout/TopBar'
import { ChatWorkbenchHeader } from './components/chat/ChatWorkbenchHeader'
import { ChatView } from './components/chat/ChatView'
import { WorkspaceFilePanel } from './components/chat/WorkspaceFilePanel'
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

const SIDEBAR_WIDTH_STORAGE_KEY = 'aether-web-sidebar-width'
const WORKSPACE_RAIL_WIDTH_STORAGE_KEY = 'aether-web-workspace-rail-width'
const WORKSPACE_FILE_PANEL_WIDTH_STORAGE_KEY = 'aether-web-workspace-file-panel-width'
const PANEL_SWAP_STORAGE_KEY = 'aether-web-panels-swapped'
const APP_RAIL_WIDTH = 60
const LAYOUT_RESIZER_WIDTH = 6
const WORKSPACE_RAIL_BREAKPOINT = 1180
const SIDEBAR_WIDTH_DEFAULT = 304
const SIDEBAR_WIDTH_MIN = 248
const SIDEBAR_WIDTH_MAX = 440
const WORKSPACE_RAIL_WIDTH_DEFAULT = 360
const WORKSPACE_RAIL_WIDTH_MIN = 320
const WORKSPACE_RAIL_WIDTH_MAX = 640
const WORKSPACE_FILE_PANEL_WIDTH_DEFAULT = 560
const WORKSPACE_FILE_PANEL_WIDTH_MIN = 320
const MIN_CHAT_COLUMN_WIDTH = 320
const MIN_CHAT_WORKSPACE_WIDTH = MIN_CHAT_COLUMN_WIDTH
const RESIZE_KEY_STEP = 16

type AppShellStyle = CSSProperties & {
  '--sidebar-width'?: string
  '--workspace-rail-width'?: string
  '--workspace-file-panel-width'?: string
}

type WorkspaceFilePanelClampContext = {
  sidebarWidth: number
  workspaceRailWidth: number
  workspaceRailOpen: boolean
  panelsSwappedInChat: boolean
}

export function App() {
  const { status, health, activeView, isLoading, error, bootstrap, setActiveView } = useAppStore()
  const { sessions, activeSessionId, isLoading: sessionsLoading, createSession, deleteSession, setActiveSession } = useSessionStore()
  const { current, providers, loadProviders } = useProviderStore()
  const connectChat = useChatStore((state) => state.connect)
  const clearChatSession = useChatStore((state) => state.clearSession)
  const clearSessionTasks = useTaskStore((state) => state.clearSessionTasks)
  const notify = useToastStore((state) => state.notify)
  const bootstrapAppearance = useAppearanceStore((state) => state.bootstrap)
  const [workspaceRailOpen, setWorkspaceRailOpen] = useState(true)
  const [sidebarWidth, setSidebarWidth] = useState(() => readStoredNumber(SIDEBAR_WIDTH_STORAGE_KEY, SIDEBAR_WIDTH_DEFAULT))
  const [workspaceRailWidth, setWorkspaceRailWidth] = useState(() => readStoredNumber(WORKSPACE_RAIL_WIDTH_STORAGE_KEY, WORKSPACE_RAIL_WIDTH_DEFAULT))
  const [workspaceFilePanelWidth, setWorkspaceFilePanelWidth] = useState(() => readStoredNumber(WORKSPACE_FILE_PANEL_WIDTH_STORAGE_KEY, WORKSPACE_FILE_PANEL_WIDTH_DEFAULT))
  const [panelsSwapped, setPanelsSwapped] = useState(() => readStoredBoolean(PANEL_SWAP_STORAGE_KEY, false))
  const [workspacePreviewPath, setWorkspacePreviewPath] = useState<string | null>(null)
  const [workspacePreviewFile, setWorkspacePreviewFile] = useState<WorkspaceFile | null>(null)
  const [workspacePreviewLoading, setWorkspacePreviewLoading] = useState(false)
  const [workspacePreviewError, setWorkspacePreviewError] = useState<string | null>(null)
  const panelsSwappedInChat = activeView === 'chat' && panelsSwapped
  const workspaceFilePanelClampContext = useMemo<WorkspaceFilePanelClampContext>(() => ({
    sidebarWidth,
    workspaceRailWidth,
    workspaceRailOpen,
    panelsSwappedInChat,
  }), [panelsSwappedInChat, sidebarWidth, workspaceRailOpen, workspaceRailWidth])
  const workspaceFilePanelMaxWidth = useMemo(
    () => clampWorkspaceFilePanelWidth(Number.POSITIVE_INFINITY, workspaceFilePanelClampContext),
    [workspaceFilePanelClampContext],
  )

  useEffect(() => {
    void bootstrapAppearance()
    void bootstrap()
    void useSessionStore.getState().loadSessions()
    void loadProviders()
    connectChat()
  }, [bootstrap, bootstrapAppearance, connectChat, loadProviders])

  useEffect(() => {
    const handleResize = () => {
      setSidebarWidth((width) => clampSidebarWidth(width))
      setWorkspaceRailWidth((width) => clampWorkspaceRailWidth(width, sidebarWidth))
      setWorkspaceFilePanelWidth((width) => clampWorkspaceFilePanelWidth(width, workspaceFilePanelClampContext))
    }
    window.addEventListener('resize', handleResize)
    handleResize()
    return () => window.removeEventListener('resize', handleResize)
  }, [sidebarWidth, workspaceFilePanelClampContext])

  useEffect(() => {
    writeStoredNumber(SIDEBAR_WIDTH_STORAGE_KEY, sidebarWidth)
  }, [sidebarWidth])

  useEffect(() => {
    writeStoredNumber(WORKSPACE_RAIL_WIDTH_STORAGE_KEY, workspaceRailWidth)
  }, [workspaceRailWidth])

  useEffect(() => {
    writeStoredNumber(WORKSPACE_FILE_PANEL_WIDTH_STORAGE_KEY, workspaceFilePanelWidth)
  }, [workspaceFilePanelClampContext, workspaceFilePanelWidth])

  useEffect(() => {
    writeStoredBoolean(PANEL_SWAP_STORAGE_KEY, panelsSwapped)
  }, [panelsSwapped])

  const appShellStyle = useMemo<AppShellStyle>(() => ({
    '--sidebar-width': sidebarWidth + 'px',
    '--workspace-rail-width': workspaceRailWidth + 'px',
    '--workspace-file-panel-width': workspaceFilePanelWidth + 'px',
  }), [sidebarWidth, workspaceFilePanelWidth, workspaceRailWidth])
  const appShellClassName = [
    'app-shell',
    panelsSwappedInChat ? 'app-shell-panels-swapped' : '',
    panelsSwappedInChat && !workspaceRailOpen ? 'app-shell-workspace-closed' : '',
  ].filter(Boolean).join(' ')

  const startSidebarResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    const startX = event.clientX
    const startWidth = sidebarWidth
    const direction = panelsSwappedInChat ? -1 : 1
    document.body.classList.add('is-resizing-layout')

    const handlePointerMove = (moveEvent: PointerEvent) => {
      setSidebarWidth(clampSidebarWidth(startWidth + direction * (moveEvent.clientX - startX)))
    }
    const handlePointerUp = () => {
      document.body.classList.remove('is-resizing-layout')
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp, { once: true })
  }, [panelsSwappedInChat, sidebarWidth])

  const startWorkspaceRailResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    const startX = event.clientX
    const startWidth = workspaceRailWidth
    const direction = panelsSwappedInChat ? 1 : -1
    document.body.classList.add('is-resizing-layout')

    const handlePointerMove = (moveEvent: PointerEvent) => {
      setWorkspaceRailWidth(clampWorkspaceRailWidth(startWidth + direction * (moveEvent.clientX - startX), sidebarWidth))
    }
    const handlePointerUp = () => {
      document.body.classList.remove('is-resizing-layout')
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp, { once: true })
  }, [panelsSwappedInChat, sidebarWidth, workspaceRailWidth])

  const startWorkspaceFilePanelResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    const startX = event.clientX
    const startWidth = workspaceFilePanelWidth
    document.body.classList.add('is-resizing-layout')

    const handlePointerMove = (moveEvent: PointerEvent) => {
      setWorkspaceFilePanelWidth(clampWorkspaceFilePanelWidth(startWidth + (moveEvent.clientX - startX), workspaceFilePanelClampContext))
    }
    const handlePointerUp = () => {
      document.body.classList.remove('is-resizing-layout')
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp, { once: true })
  }, [workspaceFilePanelClampContext, workspaceFilePanelWidth])

  const handleSidebarResizeKey = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault()
      const growKey = panelsSwappedInChat ? 'ArrowLeft' : 'ArrowRight'
      const delta = event.key === growKey ? RESIZE_KEY_STEP : -RESIZE_KEY_STEP
      setSidebarWidth((width) => clampSidebarWidth(width + delta))
    } else if (event.key === 'Home') {
      event.preventDefault()
      setSidebarWidth(SIDEBAR_WIDTH_MIN)
    } else if (event.key === 'End') {
      event.preventDefault()
      setSidebarWidth(clampSidebarWidth(SIDEBAR_WIDTH_MAX))
    }
  }, [panelsSwappedInChat])

  const handleWorkspaceRailResizeKey = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault()
      const growKey = panelsSwappedInChat ? 'ArrowRight' : 'ArrowLeft'
      const delta = event.key === growKey ? RESIZE_KEY_STEP : -RESIZE_KEY_STEP
      setWorkspaceRailWidth((width) => clampWorkspaceRailWidth(width + delta, sidebarWidth))
    } else if (event.key === 'Home') {
      event.preventDefault()
      setWorkspaceRailWidth(WORKSPACE_RAIL_WIDTH_MIN)
    } else if (event.key === 'End') {
      event.preventDefault()
      setWorkspaceRailWidth(clampWorkspaceRailWidth(WORKSPACE_RAIL_WIDTH_MAX, sidebarWidth))
    }
  }, [panelsSwappedInChat, sidebarWidth])

  const handleWorkspaceFilePanelResizeKey = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault()
      const delta = event.key === 'ArrowRight' ? RESIZE_KEY_STEP : -RESIZE_KEY_STEP
      setWorkspaceFilePanelWidth((width) => clampWorkspaceFilePanelWidth(width + delta, workspaceFilePanelClampContext))
    } else if (event.key === 'Home') {
      event.preventDefault()
      setWorkspaceFilePanelWidth(WORKSPACE_FILE_PANEL_WIDTH_MIN)
    } else if (event.key === 'End') {
      event.preventDefault()
      setWorkspaceFilePanelWidth(workspaceFilePanelMaxWidth)
    }
  }, [workspaceFilePanelClampContext, workspaceFilePanelMaxWidth])

  const handleSelectWorkspaceFile = useCallback((path: string) => {
    setWorkspacePreviewPath(path)
    setWorkspacePreviewFile(null)
    setWorkspacePreviewError(null)
    setWorkspacePreviewLoading(true)
    api.workspaceFile(path)
      .then((file) => {
        setWorkspacePreviewFile(file)
        setWorkspacePreviewError(null)
      })
      .catch((err: unknown) => {
        setWorkspacePreviewFile(null)
        setWorkspacePreviewError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => setWorkspacePreviewLoading(false))
  }, [])

  const closeWorkspacePreview = useCallback(() => {
    setWorkspacePreviewPath(null)
    setWorkspacePreviewFile(null)
    setWorkspacePreviewError(null)
    setWorkspacePreviewLoading(false)
  }, [])

  const handleSaveWorkspaceFile = useCallback(async (path: string, content: string) => {
    const saved = await api.workspaceSaveFile(path, content)
    const nextFile = { ...saved, path: saved.path || path }
    setWorkspacePreviewFile(nextFile)
    setWorkspacePreviewPath(nextFile.path)
    setWorkspacePreviewError(null)
    notify('Saved ' + nextFile.path, 'success')
    return nextFile
  }, [notify])

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    try {
      await deleteSession(sessionId)
      clearChatSession(sessionId)
      clearSessionTasks(sessionId)
      notify('Deleted session ' + sessionId.slice(0, 8), 'success')
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      notify(message, 'error')
      throw err
    }
  }, [clearChatSession, clearSessionTasks, deleteSession, notify])

  const activeSession = sessions.find((session) => session.session_id === activeSessionId) ?? null
  const workspacePreviewOpen = Boolean(workspacePreviewPath)

  return (
    <div className={appShellClassName} style={appShellStyle}>
      <AppRail activeView={activeView} onSelectView={setActiveView} />
      {panelsSwappedInChat ? (
        workspaceRailOpen ? (
          <WorkspaceRail
            side="left"
            selectedFilePath={workspacePreviewPath}
            onSelectFile={handleSelectWorkspaceFile}
            onClose={() => setWorkspaceRailOpen(false)}
            onOpenWorkspace={() => setActiveView('workspace')}
          />
        ) : null
      ) : (
        <SessionSidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={(id) => {
            setActiveSession(id)
            setActiveView('chat')
          }}
          onNewSession={() => {
            const provider = current?.provider_name || providers[0]?.name || 'openai'
            const model = current?.model || 'gpt-5.4'
            void createSession({ provider, model })
          }}
          onDeleteSession={handleDeleteSession}
        />
      )}
      {panelsSwappedInChat ? (
        workspaceRailOpen ? (
          <div
            className="layout-resizer app-shell-resizer app-shell-workspace-resizer"
            role="separator"
            aria-label="Resize workspace panel"
            aria-orientation="vertical"
            aria-valuemin={WORKSPACE_RAIL_WIDTH_MIN}
            aria-valuemax={clampWorkspaceRailWidth(WORKSPACE_RAIL_WIDTH_MAX, sidebarWidth)}
            aria-valuenow={workspaceRailWidth}
            tabIndex={0}
            onPointerDown={startWorkspaceRailResize}
            onKeyDown={handleWorkspaceRailResizeKey}
          />
        ) : null
      ) : (
        <div
          className="layout-resizer app-shell-resizer"
          role="separator"
          aria-label="Resize sessions sidebar"
          aria-orientation="vertical"
          aria-valuemin={SIDEBAR_WIDTH_MIN}
          aria-valuemax={clampSidebarWidth(SIDEBAR_WIDTH_MAX)}
          aria-valuenow={sidebarWidth}
          tabIndex={0}
          onPointerDown={startSidebarResize}
          onKeyDown={handleSidebarResizeKey}
        />
      )}
      <main className={activeView === 'chat' ? 'workspace workspace-chat' : 'workspace'}>
        {activeView === 'chat' ? (
          <ChatWorkbenchHeader
            session={activeSession}
            online={Boolean(status?.ok)}
            workspaceRailOpen={workspaceRailOpen}
            panelsSwapped={panelsSwapped}
            onToggleWorkspaceRail={() => setWorkspaceRailOpen((value) => !value)}
            onSwapPanels={() => setPanelsSwapped((value) => !value)}
          />
        ) : (
          <TopBar
            title={viewTitle(activeView)}
            status={status?.ok ? 'online' : 'offline'}
            provider={current?.provider_name}
            model={current?.model}
          />
        )}
        <section className={activeView === 'chat' ? 'content-pane content-pane-chat' : 'content-pane'}>
          {isLoading || sessionsLoading ? <Spinner label="Loading console" /> : null}
          {error ? (
            <div className="notice notice-error">
              <CircleAlert size={16} />
              <span>{error}</span>
            </div>
          ) : null}
          {activeView === 'chat' ? (
            <div className={[
              'chat-workbench',
              panelsSwappedInChat ? 'chat-workbench-panels-swapped' : '',
              !panelsSwappedInChat && !workspaceRailOpen ? 'chat-workbench-rail-closed' : '',
              workspacePreviewOpen ? 'chat-workbench-file-open' : '',
            ].filter(Boolean).join(' ')}>
              {workspacePreviewOpen ? (
                <>
                  <WorkspaceFilePanel
                    preview={{
                      path: workspacePreviewPath,
                      file: workspacePreviewFile,
                      loading: workspacePreviewLoading,
                      error: workspacePreviewError,
                    }}
                    onClose={closeWorkspacePreview}
                    onSave={handleSaveWorkspaceFile}
                  />
                  <div
                    className="layout-resizer chat-workbench-resizer chat-workbench-file-resizer"
                    role="separator"
                    aria-label="Resize workspace file preview"
                    aria-orientation="vertical"
                    aria-valuemin={WORKSPACE_FILE_PANEL_WIDTH_MIN}
                    aria-valuemax={workspaceFilePanelMaxWidth}
                    aria-valuenow={workspaceFilePanelWidth}
                    tabIndex={0}
                    onPointerDown={startWorkspaceFilePanelResize}
                    onKeyDown={handleWorkspaceFilePanelResizeKey}
                  />
                </>
              ) : null}
              <ChatView session={activeSession} />
              {panelsSwappedInChat ? (
                <>
                  <div
                    className="layout-resizer chat-workbench-resizer chat-workbench-session-resizer"
                    role="separator"
                    aria-label="Resize sessions sidebar"
                    aria-orientation="vertical"
                    aria-valuemin={SIDEBAR_WIDTH_MIN}
                    aria-valuemax={clampSidebarWidth(SIDEBAR_WIDTH_MAX)}
                    aria-valuenow={sidebarWidth}
                    tabIndex={0}
                    onPointerDown={startSidebarResize}
                    onKeyDown={handleSidebarResizeKey}
                  />
                  <SessionSidebar
                    placement="right"
                    sessions={sessions}
                    activeSessionId={activeSessionId}
                    onSelectSession={(id) => {
                      setActiveSession(id)
                      setActiveView('chat')
                    }}
                    onNewSession={() => {
                      const provider = current?.provider_name || providers[0]?.name || 'openai'
                      const model = current?.model || 'gpt-5.4'
                      void createSession({ provider, model })
                    }}
                    onDeleteSession={handleDeleteSession}
                  />
                </>
              ) : workspaceRailOpen ? (
                <>
                  <div
                    className="layout-resizer chat-workbench-resizer"
                    role="separator"
                    aria-label="Resize workspace panel"
                    aria-orientation="vertical"
                    aria-valuemin={WORKSPACE_RAIL_WIDTH_MIN}
                    aria-valuemax={clampWorkspaceRailWidth(WORKSPACE_RAIL_WIDTH_MAX, sidebarWidth)}
                    aria-valuenow={workspaceRailWidth}
                    tabIndex={0}
                    onPointerDown={startWorkspaceRailResize}
                    onKeyDown={handleWorkspaceRailResizeKey}
                  />
                  <WorkspaceRail
                    side="right"
                    selectedFilePath={workspacePreviewPath}
                    onSelectFile={handleSelectWorkspaceFile}
                    onClose={() => setWorkspaceRailOpen(false)}
                    onOpenWorkspace={() => setActiveView('workspace')}
                  />
                </>
              ) : null}
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

export { navItems } from './navItems'

function readStoredNumber(key: string, fallback: number): number {
  if (typeof window === 'undefined') return fallback
  const value = Number(window.localStorage.getItem(key))
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function writeStoredNumber(key: string, value: number): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(key, String(Math.round(value)))
}

function readStoredBoolean(key: string, fallback: boolean): boolean {
  if (typeof window === 'undefined') return fallback
  const value = window.localStorage.getItem(key)
  if (value === null) return fallback
  return value === 'true'
}

function writeStoredBoolean(key: string, value: boolean): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(key, value ? 'true' : 'false')
}

function clampSidebarWidth(width: number): number {
  const viewportWidth = typeof window === 'undefined' ? 1440 : window.innerWidth
  const workspaceRailReserve = viewportWidth > WORKSPACE_RAIL_BREAKPOINT
    ? WORKSPACE_RAIL_WIDTH_MIN + LAYOUT_RESIZER_WIDTH
    : 0
  const maxByViewport = viewportWidth - APP_RAIL_WIDTH - LAYOUT_RESIZER_WIDTH - MIN_CHAT_WORKSPACE_WIDTH - workspaceRailReserve
  return clamp(width, SIDEBAR_WIDTH_MIN, Math.max(SIDEBAR_WIDTH_MIN, Math.min(SIDEBAR_WIDTH_MAX, maxByViewport)))
}

function clampWorkspaceRailWidth(width: number, sidebarWidth: number): number {
  const viewportWidth = typeof window === 'undefined' ? 1440 : window.innerWidth
  const availableWidth = viewportWidth - APP_RAIL_WIDTH - LAYOUT_RESIZER_WIDTH - sidebarWidth
  const maxByViewport = availableWidth - MIN_CHAT_WORKSPACE_WIDTH - LAYOUT_RESIZER_WIDTH
  return clamp(width, WORKSPACE_RAIL_WIDTH_MIN, Math.max(WORKSPACE_RAIL_WIDTH_MIN, Math.min(WORKSPACE_RAIL_WIDTH_MAX, maxByViewport)))
}

function clampWorkspaceFilePanelWidth(width: number, context: WorkspaceFilePanelClampContext): number {
  const viewportWidth = typeof window === 'undefined' ? 1440 : window.innerWidth
  const appShellSideWidth = context.panelsSwappedInChat && context.workspaceRailOpen
    ? context.workspaceRailWidth
    : context.panelsSwappedInChat
      ? 0
      : context.sidebarWidth
  const workbenchSideWidth = context.panelsSwappedInChat
    ? context.sidebarWidth
    : context.workspaceRailOpen
      ? context.workspaceRailWidth
      : 0
  const workbenchSideResizerWidth = workbenchSideWidth > 0 ? LAYOUT_RESIZER_WIDTH : 0
  const appShellResizerWidth = appShellSideWidth > 0 ? LAYOUT_RESIZER_WIDTH : 0
  const availableWorkbenchWidth = viewportWidth - APP_RAIL_WIDTH - appShellSideWidth - appShellResizerWidth
  const maxByWorkbench = availableWorkbenchWidth - MIN_CHAT_COLUMN_WIDTH - LAYOUT_RESIZER_WIDTH - workbenchSideResizerWidth - workbenchSideWidth
  const maxWidth = Math.max(WORKSPACE_FILE_PANEL_WIDTH_MIN, maxByWorkbench)
  return clamp(width, WORKSPACE_FILE_PANEL_WIDTH_MIN, maxWidth)
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}
