import { ArrowDown, Bot } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { SessionInfo } from '../../api/types'
import { useChatStore } from '../../stores/chatStore'
import { useSessionStore } from '../../stores/sessionStore'
import { EmptyState } from '../shared/EmptyState'
import { ApprovalDialog } from './ApprovalDialog'
import { ChatTimeline } from './ChatTimeline'
import { Composer } from './Composer'
import { PermissionDialog } from './PermissionDialog'
import { executeWebSlashCommand } from './slashExecute'

type Props = {
  session: SessionInfo | null
}

const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 48

export function ChatView({ session }: Props) {
  const loadTranscript = useChatStore((state) => state.loadTranscript)
  const blocks = useChatStore((state) => (session ? state.blocksBySession[session.session_id] ?? [] : []))
  const tokenUsageByRun = useChatStore((state) => state.tokenUsageByRun)
  const startRun = useChatStore((state) => state.startRun)
  const cancelRun = useChatStore((state) => state.cancelRun)
  const appendLocalNotice = useChatStore((state) => state.appendLocalNotice)
  const appendLocalError = useChatStore((state) => state.appendLocalError)
  const activeRunId = useChatStore((state) => state.activeRunId)
  const pendingPermission = useChatStore((state) => state.pendingPermission)
  const pendingApproval = useChatStore((state) => state.pendingApproval)
  const respondPermission = useChatStore((state) => state.respondPermission)
  const respondApproval = useChatStore((state) => state.respondApproval)
  const setSessionMode = useSessionStore((state) => state.setSessionMode)
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const shouldAutoScrollRef = useRef(true)
  const programmaticScrollRef = useRef(false)
  const lastSessionIdRef = useRef<string | null>(null)
  const [showJumpToLatest, setShowJumpToLatest] = useState(false)

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'auto') => {
    shouldAutoScrollRef.current = true
    programmaticScrollRef.current = true
    bottomRef.current?.scrollIntoView({ behavior, block: 'end' })
    setShowJumpToLatest(false)
    requestAnimationFrame(() => {
      programmaticScrollRef.current = false
    })
  }, [])

  const handleScroll = useCallback(() => {
    if (programmaticScrollRef.current) return
    const element = scrollRef.current
    if (!element) return
    const atBottom = isNearChatBottom(element)
    shouldAutoScrollRef.current = atBottom
    setShowJumpToLatest(!atBottom)
  }, [])

  useEffect(() => {
    if (!session) return
    void loadTranscript(session.session_id)
  }, [loadTranscript, session])

  useEffect(() => {
    const sessionId = session?.session_id ?? null
    if (lastSessionIdRef.current === sessionId) return
    lastSessionIdRef.current = sessionId
    shouldAutoScrollRef.current = true
    setShowJumpToLatest(false)
    requestAnimationFrame(() => scrollToBottom('auto'))
  }, [scrollToBottom, session?.session_id])

  useEffect(() => {
    if (!session) return
    if (!shouldAutoScrollRef.current) {
      setShowJumpToLatest(true)
      return
    }
    requestAnimationFrame(() => scrollToBottom('smooth'))
  }, [blocks, scrollToBottom, session])

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

  const handleSlashCommand = (command: string) => {
    void executeWebSlashCommand(command, { session, onSessionMode: setSessionMode })
      .then((result) => {
        if (result.type === 'notice') {
          appendLocalNotice(session.session_id, result.message)
          return
        }
        if (result.type === 'error') {
          appendLocalError(session.session_id, result.message)
          return
        }
        startRun(session.session_id, result.message)
      })
      .catch((error) => {
        appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
      })
  }

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
      <div className="chat-scroll" onScroll={handleScroll} ref={scrollRef}>
        <ChatTimeline
          blocks={blocks}
          onRespondPermission={respondPermission}
          onRespondApproval={respondApproval}
        />
        <div ref={bottomRef} />
        {showJumpToLatest ? (
          <button type="button" className="chat-jump-latest" onClick={() => scrollToBottom('smooth')}>
            <ArrowDown size={14} />
            Jump to latest
          </button>
        ) : null}
      </div>
      <Composer
        disabled={!session}
        running={Boolean(activeRunId)}
        onSend={(message, attachments) => startRun(session.session_id, message, attachments)}
        onSlashCommand={handleSlashCommand}
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
          onApprove={(answers) => respondApproval({ confirmed: true, ...(answers ? { answers } : {}) })}
          onReject={() => respondApproval({ confirmed: false })}
        />
      ) : null}
    </div>
  )
}

export function isNearChatBottom(element: Pick<HTMLElement, 'scrollHeight' | 'scrollTop' | 'clientHeight'>): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= AUTO_SCROLL_BOTTOM_THRESHOLD_PX
}
