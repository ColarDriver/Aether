import { Bot } from 'lucide-react'
import { useEffect } from 'react'
import type { SessionInfo } from '../../api/types'
import { useChatStore } from '../../stores/chatStore'
import { EmptyState } from '../shared/EmptyState'
import { ApprovalDialog } from './ApprovalDialog'
import { Composer } from './Composer'
import { MessageList } from './MessageList'
import { PermissionDialog } from './PermissionDialog'
import { ToolCallBlock } from './ToolCallBlock'

type Props = {
  session: SessionInfo | null
}

export function ChatView({ session }: Props) {
  const loadTranscript = useChatStore((state) => state.loadTranscript)
  const messages = useChatStore((state) => (session ? state.messagesBySession[session.session_id] ?? [] : []))
  const tools = useChatStore((state) => (session ? state.toolsBySession[session.session_id] ?? [] : []))
  const tokenUsageByRun = useChatStore((state) => state.tokenUsageByRun)
  const startRun = useChatStore((state) => state.startRun)
  const cancelRun = useChatStore((state) => state.cancelRun)
  const activeRunId = useChatStore((state) => state.activeRunId)
  const pendingPermission = useChatStore((state) => state.pendingPermission)
  const pendingApproval = useChatStore((state) => state.pendingApproval)
  const respondPermission = useChatStore((state) => state.respondPermission)
  const respondApproval = useChatStore((state) => state.respondApproval)

  useEffect(() => {
    if (!session) return
    void loadTranscript(session.session_id)
  }, [loadTranscript, session])

  if (!session) {
    return (
      <EmptyState
        icon={<Bot />}
        title="No session selected"
        description="Create or select a session to start using the browser console."
      />
    )
  }

  const usage = activeRunId ? tokenUsageByRun[activeRunId] : undefined

  return (
    <div className="chat-surface">
      <div className="chat-header">
        <Bot size={18} />
        <div>
          <div className="chat-title">{session.summary || session.session_id}</div>
          <div className="muted">{session.provider} / {session.model}</div>
        </div>
        {usage ? (
          <div className="token-pill">{usage.input_tokens ?? 0} in / {usage.output_tokens ?? 0} out</div>
        ) : null}
      </div>
      <div className="chat-scroll">
        <MessageList messages={messages} />
        {tools.length > 0 ? (
          <div className="tool-stack">
            {tools.map((tool) => <ToolCallBlock key={tool.id} tool={tool} />)}
          </div>
        ) : null}
      </div>
      <Composer
        disabled={!session}
        running={Boolean(activeRunId)}
        onSend={(message) => startRun(session.session_id, message)}
        onCancel={() => cancelRun(session.session_id)}
      />
      {pendingPermission ? (
        <PermissionDialog
          prompt={pendingPermission}
          onAllow={() => respondPermission({ type: 'allow_once' })}
          onAllowSession={() => respondPermission({ type: 'allow_session' })}
          onDeny={() => respondPermission({ type: 'deny' })}
        />
      ) : null}
      {pendingApproval ? (
        <ApprovalDialog
          prompt={pendingApproval}
          onApprove={() => respondApproval({ confirmed: true })}
          onReject={() => respondApproval({ confirmed: false })}
        />
      ) : null}
    </div>
  )
}
