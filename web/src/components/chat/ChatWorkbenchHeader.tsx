import { ArrowLeftRight, Bot, Circle, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen } from 'lucide-react'
import type { SessionInfo } from '../../api/types'
import { AppearanceControls } from '../shared/AppearanceControls'

type Props = {
  session: SessionInfo | null
  online: boolean
  workspaceRailOpen: boolean
  panelsSwapped?: boolean
  onToggleWorkspaceRail: () => void
  onSwapPanels?: () => void
}

export function ChatWorkbenchHeader({
  session,
  online,
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
