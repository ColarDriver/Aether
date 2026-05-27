import { ArrowLeftRight, Bot, Circle, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen } from 'lucide-react'
import type { SessionInfo } from '../../api/types'
import { AppearanceControls } from '../shared/AppearanceControls'

type SocketConnectionState = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'disconnected'

type Props = {
  session: SessionInfo | null
  online: boolean
  socketState?: SocketConnectionState
  socketDetail?: string | null
  workspaceRailOpen: boolean
  panelsSwapped?: boolean
  onToggleWorkspaceRail: () => void
  onSwapPanels?: () => void
}

export function ChatWorkbenchHeader({
  session,
  online,
  socketState = 'idle',
  socketDetail,
  workspaceRailOpen,
  panelsSwapped = false,
  onToggleWorkspaceRail,
  onSwapPanels,
}: Props) {
  const title = session?.summary || session?.session_id.slice(0, 8) || 'New chat'
  const messageCount = session?.message_count ?? 0
  const WorkspaceToggleIcon = workspaceRailOpen
    ? (panelsSwapped ? PanelLeftClose : PanelRightClose)
    : (panelsSwapped ? PanelLeftOpen : PanelRightOpen)
  const socket = socketStatus(socketState, socketDetail)

  return (
    <header className="chat-workbench-header">
      <div className="chat-workbench-title">
        <span className="chat-workbench-icon" aria-hidden="true">
          <Bot size={16} />
        </span>
        <div>
          <h1>{title}</h1>
          <p>
            {session ? messageCount + ' message' + (messageCount === 1 ? '' : 's') : 'Select a session or start a new run'}
          </p>
        </div>
      </div>
      <div className="chat-workbench-meta" aria-label="Chat session status">
        <span className={online ? 'workbench-chip workbench-chip-online' : 'workbench-chip workbench-chip-offline'}>
          <Circle size={9} fill="currentColor" />
          <span>
            <strong>{online ? 'online' : 'offline'}</strong>
            <small>runtime</small>
          </span>
        </span>
        <span className={'workbench-chip workbench-chip-' + socket.tone} title={socket.title}>
          <Circle size={9} fill="currentColor" />
          <span>
            <strong>{socket.label}</strong>
            <small>stream</small>
          </span>
        </span>
      </div>
      <div className="chat-workbench-actions">
        <AppearanceControls compact />
        {onSwapPanels ? (
          <button
            type="button"
            className="workbench-icon-button"
            onClick={onSwapPanels}
            aria-pressed={panelsSwapped}
            aria-label="Swap sessions and workspace panels"
            title="Swap sessions and workspace panels"
          >
            <ArrowLeftRight size={16} />
          </button>
        ) : null}
        <button
          type="button"
          className="workbench-icon-button"
          onClick={onToggleWorkspaceRail}
          aria-pressed={workspaceRailOpen}
          aria-label={workspaceRailOpen ? 'Hide workspace panel' : 'Show workspace panel'}
          title={workspaceRailOpen ? 'Hide workspace panel' : 'Show workspace panel'}
        >
          <WorkspaceToggleIcon size={16} />
        </button>
      </div>
    </header>
  )
}

function socketStatus(state: SocketConnectionState, detail?: string | null): { label: string; tone: string; title: string } {
  if (state === 'connected') return { label: 'connected', tone: 'online', title: detail || 'Run stream connected' }
  if (state === 'connecting') return { label: 'connecting', tone: 'pending', title: detail || 'Opening run stream' }
  if (state === 'reconnecting') return { label: 'reconnecting', tone: 'pending', title: detail || 'Reconnecting run stream' }
  if (state === 'disconnected') return { label: 'disconnected', tone: 'offline', title: detail || 'Run stream disconnected' }
  return { label: 'idle', tone: 'muted', title: detail || 'Run stream idle' }
}
