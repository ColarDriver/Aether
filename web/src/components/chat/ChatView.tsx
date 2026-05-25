import { ArrowDown, Bot, FileSearch, Route, ShieldCheck } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { SessionInfo, TaskSummary } from '../../api/types'
import type { ChatBlock, RunStatusSnapshot, TokenUsage } from '../../chat-rendering'
import { tokenUsageFromRecord, tokenUsageTotal } from '../../chat-rendering'
import { useAppStore } from '../../stores/appStore'
import { useChatStore } from '../../stores/chatStore'
import { useProviderStore } from '../../stores/providerStore'
import { useSessionStore } from '../../stores/sessionStore'
import { useTaskStore } from '../../stores/taskStore'
import { ActivityBar } from './ActivityBar'
import { ApprovalDialog } from './ApprovalDialog'
import { ChatTimeline } from './ChatTimeline'
import { Composer, type ComposerDraftPatch } from './Composer'
import { PermissionDialog } from './PermissionDialog'
import { isTaskTerminal, SessionTaskBar } from './SessionTaskBar'
import { executeWebSlashCommand } from './slashExecute'
import { TaskDetailDialog } from './TaskDetailDialog'

type Props = {
  session: SessionInfo | null
}

const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 48
const EMPTY_CHAT_BLOCKS: ChatBlock[] = []
const EMPTY_TASKS: TaskSummary[] = []

export type ScrollSnapshot = {
  scrollTop: number
  atBottom: boolean
}

export function ChatView({ session }: Props) {
  const sessionId = session?.session_id ?? null
  const loadTranscript = useChatStore((state) => state.loadTranscript)
  const blocks = useChatStore((state) => (sessionId ? state.blocksBySession[sessionId] ?? EMPTY_CHAT_BLOCKS : EMPTY_CHAT_BLOCKS))
  const tokenUsageByRun = useChatStore((state) => state.tokenUsageByRun)
  const statusByRun = useChatStore((state) => state.statusByRun)
  const startRun = useChatStore((state) => state.startRun)
  const cancelRun = useChatStore((state) => state.cancelRun)
  const appendLocalNotice = useChatStore((state) => state.appendLocalNotice)
  const appendLocalError = useChatStore((state) => state.appendLocalError)
  const clearSession = useChatStore((state) => state.clearSession)
  const activeRunId = useChatStore((state) => state.activeRunId)
  const pendingPermission = useChatStore((state) => state.pendingPermission)
  const pendingApproval = useChatStore((state) => state.pendingApproval)
  const respondPermission = useChatStore((state) => state.respondPermission)
  const respondApproval = useChatStore((state) => state.respondApproval)
  const tasks = useTaskStore((state) => (sessionId ? state.tasksBySession[sessionId] ?? EMPTY_TASKS : EMPTY_TASKS))
  const loadSessionTasks = useTaskStore((state) => state.loadSessionTasks)
  const setActiveView = useAppStore((state) => state.setActiveView)
  const createSession = useSessionStore((state) => state.createSession)
  const resumeSession = useSessionStore((state) => state.resumeSession)
  const updateSession = useSessionStore((state) => state.updateSession)
  const setSessionMode = useSessionStore((state) => state.setSessionMode)
  const currentProvider = useProviderStore((state) => state.current)
  const providers = useProviderStore((state) => state.providers)
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const shouldAutoScrollRef = useRef(true)
  const programmaticScrollRef = useRef(false)
  const lastSessionIdRef = useRef<string | null>(null)
  const scrollSnapshotsRef = useRef<Record<string, ScrollSnapshot>>({})
  const [showJumpToLatest, setShowJumpToLatest] = useState(false)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [composerDraftPatch, setComposerDraftPatch] = useState<ComposerDraftPatch | null>(null)
  const composerDraftPatchIdRef = useRef(0)

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'auto') => {
    shouldAutoScrollRef.current = true
    programmaticScrollRef.current = true
    if (typeof bottomRef.current?.scrollIntoView === 'function') {
      bottomRef.current.scrollIntoView({ behavior, block: 'end' })
    }
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
    if (sessionId) {
      scrollSnapshotsRef.current[sessionId] = {
        scrollTop: element.scrollTop,
        atBottom,
      }
    }
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return
    void loadTranscript(sessionId)
  }, [loadTranscript, sessionId])

  useEffect(() => {
    if (!sessionId) return
    void loadSessionTasks(sessionId)
  }, [loadSessionTasks, sessionId])

  useEffect(() => {
    if (!sessionId) return
    const hasActiveTasks = tasks.some((task) => !isTaskTerminal(task))
    if (!activeRunId && !hasActiveTasks) return
    const interval = window.setInterval(() => {
      void loadSessionTasks(sessionId)
    }, 2000)
    return () => window.clearInterval(interval)
  }, [activeRunId, loadSessionTasks, sessionId, tasks])

  useEffect(() => {
    if (lastSessionIdRef.current === sessionId) return
    lastSessionIdRef.current = sessionId
    setSelectedTaskId(null)
    const snapshot = sessionId ? scrollSnapshotsRef.current[sessionId] : undefined
    if (snapshot && !snapshot.atBottom) {
      shouldAutoScrollRef.current = false
      setShowJumpToLatest(true)
      requestAnimationFrame(() => {
        const element = scrollRef.current
        if (!element) return
        element.scrollTop = restoredChatScrollTop(snapshot, element)
      })
      return
    }
    shouldAutoScrollRef.current = true
    setShowJumpToLatest(false)
    requestAnimationFrame(() => scrollToBottom('auto'))
  }, [scrollToBottom, sessionId])

  useEffect(() => {
    if (!session) return
    if (!shouldAutoScrollRef.current) {
      setShowJumpToLatest(true)
      return
    }
    requestAnimationFrame(() => scrollToBottom('smooth'))
  }, [blocks, scrollToBottom, session])

  const activeStatus = activeRunId ? statusByRun[activeRunId] : undefined
  const activeUsage = activeRunId ? tokenUsageByRun[activeRunId] ?? activeStatus?.tokens : undefined
  const latestSessionUsage = session ? latestTokenUsageForSession(session.session_id, blocks, statusByRun) : undefined
  const usage = activeRunId ? activeUsage : latestSessionUsage
  const fallbackProvider = currentProvider?.provider_name || providers[0]?.name || 'openai'
  const fallbackModel = currentProvider?.model || 'gpt-5.4'

  const patchComposerDraft = (mode: ComposerDraftPatch['mode'], text: string, attachments?: ComposerDraftPatch['attachments']) => {
    composerDraftPatchIdRef.current += 1
    setComposerDraftPatch({ id: composerDraftPatchIdRef.current, mode, text, ...(attachments ? { attachments } : {}) })
  }

  const quoteMessage = (role: 'user' | 'assistant', content: string) => {
    const quoted = formatQuotedDraft(role, content)
    if (quoted) patchComposerDraft('append', quoted)
  }

  const createSessionAndRun = (message: string, attachments?: Parameters<typeof startRun>[2]) => {
    void createSession({ provider: fallbackProvider, model: fallbackModel })
      .then((created) => startRun(created.session_id, message, attachments))
      .catch((error) => {
        // There is no session transcript yet, so surface the failure in devtools
        // and keep the composer usable for retry.
        console.error('Failed to create session before run', error)
      })
  }

  if (!session) {
    return (
      <div className="chat-surface chat-surface-empty">
        <div className="chat-scroll">
          <div className="chat-start">
            <section className="chat-start-panel" aria-label="Aether start">
              <div className="chat-start-brand">
                <span className="chat-start-icon" aria-hidden="true"><Bot size={18} /></span>
                <div>
                  <h2>Aether</h2>
                  <p>Workspace session</p>
                </div>
              </div>
              <div className="chat-start-actions" aria-label="Starter prompts">
                {starterPrompts.map((prompt) => {
                  const Icon = prompt.icon
                  return (
                    <button key={prompt.prompt} type="button" onClick={() => createSessionAndRun(prompt.prompt)}>
                      <Icon size={16} />
                      <span>
                        <strong>{prompt.title}</strong>
                        <small>{prompt.detail}</small>
                      </span>
                    </button>
                  )
                })}
              </div>
            </section>
          </div>
        </div>
        <Composer
          disabled={false}
          running={false}
          provider={fallbackProvider}
          model={fallbackModel}
          onSend={createSessionAndRun}
          onCancel={() => undefined}
          draftPatch={composerDraftPatch}
        />
      </div>
    )
  }

  const handleSlashCommand = (command: string) => {
    void executeWebSlashCommand(command, {
      session,
      createSession,
      resumeSession,
      updateSession,
      openView: setActiveView,
      onSessionMode: setSessionMode,
    })
      .then((result) => {
        if (result.type === 'notice') {
          appendLocalNotice(session.session_id, result.message)
          return
        }
        if (result.type === 'error') {
          appendLocalError(session.session_id, result.message)
          return
        }
        if (result.type === 'clear') {
          clearSession(session.session_id)
          return
        }
        startRun(session.session_id, result.message)
      })
      .catch((error) => {
        appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
      })
  }

  return (
    <div className="chat-surface" aria-label={'Chat session ' + (session.summary || session.session_id)}>
      <span className="sr-only">{session.summary || session.session_id}</span>
      <SessionTaskBar tasks={tasks} onOpenTask={(task) => setSelectedTaskId(task.task_id)} />
      <div className="chat-scroll" onScroll={handleScroll} ref={scrollRef}>
        <ChatTimeline
          blocks={blocks}
          messageActionsDisabled={Boolean(activeRunId)}
          onOpenTask={setSelectedTaskId}
          onRespondPermission={respondPermission}
          onRespondApproval={respondApproval}
          onRetryUserMessage={(block) => startRun(session.session_id, block.content, block.attachments)}
          onEditUserMessage={(block) => patchComposerDraft('replace', block.content, block.attachments)}
          onQuoteUserMessage={(block) => quoteMessage('user', block.content)}
          onQuoteAssistantMessage={(block) => quoteMessage('assistant', block.content)}
        />
        <div ref={bottomRef} />
        {showJumpToLatest ? (
          <button type="button" className="chat-jump-latest" onClick={() => scrollToBottom('smooth')}>
            <ArrowDown size={14} />
            Jump to latest
          </button>
        ) : null}
      </div>
      <ActivityBar
        activeRunId={activeRunId}
        status={activeStatus}
        tokens={usage}
        sessionId={session.session_id}
        model={session.model}
      />
      <Composer
        disabled={!session}
        running={Boolean(activeRunId)}
        sessionId={session.session_id}
        provider={session.provider}
        model={session.model}
        mode={session.mode}
        sessionSummary={session.summary}
        messageCount={session.message_count}
        inputTokens={usage?.input_tokens}
        outputTokens={usage?.output_tokens}
        tokens={usage}
        onSend={(message, attachments) => startRun(session.session_id, message, attachments)}
        onSlashCommand={handleSlashCommand}
        onCancel={() => cancelRun(session.session_id)}
        draftPatch={composerDraftPatch}
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
      {selectedTaskId ? (
        <TaskDetailDialog
          taskId={selectedTaskId}
          initialTask={tasks.find((task) => task.task_id === selectedTaskId)}
          sessionTasks={tasks}
          onOpenTask={setSelectedTaskId}
          onClose={() => setSelectedTaskId(null)}
        />
      ) : null}
    </div>
  )
}

function latestTokenUsageForSession(
  sessionId: string,
  blocks: ChatBlock[],
  statusByRun: Record<string, RunStatusSnapshot>,
): TokenUsage | undefined {
  const seenRunIds = new Set<string>()
  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index]
    if (!block || block.sessionId !== sessionId) continue
    const runId = block.runId || ''
    if (runId && !seenRunIds.has(runId)) {
      seenRunIds.add(runId)
      const statusUsage = statusByRun[runId]?.tokens
      if (tokenUsageTotal(statusUsage) > 0) return statusUsage
    }
    if (block.kind === 'assistant_message' || block.kind === 'tool_result') {
      const metadataUsage = tokenUsageFromRecord(block.metadata ?? {})
      if (tokenUsageTotal(metadataUsage) > 0) return metadataUsage
    }
  }

  const snapshots = Object.values(statusByRun).filter((snapshot) => snapshot.sessionId === sessionId)
  for (let index = snapshots.length - 1; index >= 0; index -= 1) {
    const tokens = snapshots[index]?.tokens
    if (tokenUsageTotal(tokens) > 0) return tokens
  }
  return undefined
}

export function isNearChatBottom(element: Pick<HTMLElement, 'scrollHeight' | 'scrollTop' | 'clientHeight'>): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= AUTO_SCROLL_BOTTOM_THRESHOLD_PX
}

export function restoredChatScrollTop(
  snapshot: ScrollSnapshot,
  element: Pick<HTMLElement, 'scrollHeight' | 'clientHeight'>,
): number {
  return Math.max(0, Math.min(snapshot.scrollTop, element.scrollHeight - element.clientHeight))
}

const starterPrompts = [
  {
    title: 'Inspect project',
    detail: 'Architecture and entry points',
    prompt: 'Inspect this project and summarize the architecture',
    icon: FileSearch,
  },
  {
    title: 'Review UI',
    detail: 'Find high-risk interface issues',
    prompt: 'Find the highest-risk UI issues in the web app',
    icon: ShieldCheck,
  },
  {
    title: 'Plan edit',
    detail: 'Scope the next implementation step',
    prompt: 'Plan the next implementation step before editing files',
    icon: Route,
  },
]

function formatQuotedDraft(role: 'user' | 'assistant', content: string): string {
  const trimmed = content.trim()
  if (!trimmed) return ''
  const label = role === 'user' ? 'User' : 'Assistant'
  return ['> ' + label + ':', ...trimmed.split('\n').map((line) => '> ' + line)].join('\n')
}
