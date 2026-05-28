import { ArrowDown, Bot, FileSearch, Route, ShieldCheck } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { PermissionMode, SessionCheckpointActionBody, SessionCheckpointActionResult, SessionInfo, SessionRewindResult, SessionTurnCheckpoint, TaskSummary } from '../../api/types'
import type { AssistantMessageBlock as AssistantMessage, ChatBlock, RunStatusSnapshot, TokenUsage, UserMessageBlock as UserMessage } from '../../chat-rendering'
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
import { ConfirmDialog } from '../shared/ConfirmDialog'
import type { CurrentTurnFileChangeAction, CurrentTurnUndoAction } from './blocks/CurrentTurnChangeCard'

type Props = {
  session: SessionInfo | null
  workspaceRootVersion?: number
}

const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 48
const WEB_VERBOSE_STORAGE_KEY = 'aether-web-verbose-mode'
const EMPTY_CHAT_BLOCKS: ChatBlock[] = []
const EMPTY_TASKS: TaskSummary[] = []

export type ScrollSnapshot = {
  scrollTop: number
  atBottom: boolean
}

export function ChatView({ session, workspaceRootVersion = 0 }: Props) {
  const sessionId = session?.session_id ?? null
  const loadTranscript = useChatStore((state) => state.loadTranscript)
  const blocks = useChatStore((state) => (sessionId ? state.blocksBySession[sessionId] ?? EMPTY_CHAT_BLOCKS : EMPTY_CHAT_BLOCKS))
  const tokenUsageByRun = useChatStore((state) => state.tokenUsageByRun)
  const statusByRun = useChatStore((state) => state.statusByRun)
  const startRun = useChatStore((state) => state.startRun)
  const cancelRun = useChatStore((state) => state.cancelRun)
  const replaceTranscript = useChatStore((state) => state.replaceTranscript)
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
  const setSessionPermissionMode = useSessionStore((state) => state.setSessionPermissionMode)
  const loadSessions = useSessionStore((state) => state.loadSessions)
  const currentProvider = useProviderStore((state) => state.current)
  const providers = useProviderStore((state) => state.providers)
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const shouldAutoScrollRef = useRef(true)
  const programmaticScrollRef = useRef(false)
  const lastSessionIdRef = useRef<string | null>(null)
  const scrollSnapshotsRef = useRef<Record<string, ScrollSnapshot>>({})
  const [showJumpToLatest, setShowJumpToLatest] = useState(false)
  const [verboseMode, setVerboseModeState] = useState(() => readBooleanSetting(WEB_VERBOSE_STORAGE_KEY, false))
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [composerDraftPatch, setComposerDraftPatch] = useState<ComposerDraftPatch | null>(null)
  const [turnCheckpoints, setTurnCheckpoints] = useState<SessionTurnCheckpoint[]>([])
  const [pendingTurnUndo, setPendingTurnUndo] = useState<CurrentTurnUndoAction | null>(null)
  const [turnUndoRunning, setTurnUndoRunning] = useState(false)
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
    if (!sessionId) {
      setTurnCheckpoints([])
      return
    }
    if (activeRunId) {
      setTurnCheckpoints([])
      return
    }
    let cancelled = false
    api.sessionTurnCheckpoints(sessionId)
      .then((result) => {
        if (!cancelled) setTurnCheckpoints(result.checkpoints ?? [])
      })
      .catch(() => {
        if (!cancelled) setTurnCheckpoints([])
      })
    return () => {
      cancelled = true
    }
  }, [activeRunId, blocks.length, sessionId])

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
    setPendingTurnUndo(null)
    setTurnUndoRunning(false)
  }, [sessionId])

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
  const latestRunMetadata = session ? latestRunMetadataForSession(session.session_id, blocks) : undefined
  const fallbackProvider = currentProvider?.provider_name || providers[0]?.name || 'openai'
  const fallbackModel = currentProvider?.model || 'gpt-5.4'

  const patchComposerDraft = (mode: ComposerDraftPatch['mode'], text: string, attachments?: ComposerDraftPatch['attachments']) => {
    composerDraftPatchIdRef.current += 1
    setComposerDraftPatch({ id: composerDraftPatchIdRef.current, mode, text, ...(attachments ? { attachments } : {}) })
  }

  const setVerboseMode = (enabled: boolean) => {
    setVerboseModeState(enabled)
    writeBooleanSetting(WEB_VERBOSE_STORAGE_KEY, enabled)
  }

  const quoteMessage = (role: 'user' | 'assistant', content: string) => {
    const quoted = formatQuotedDraft(role, content)
    if (quoted) patchComposerDraft('append', quoted)
  }

  const forkMessage = (block: UserMessage | AssistantMessage) => {
    if (!session || typeof block.messageIndex !== 'number') {
      if (session) appendLocalError(session.session_id, 'This message cannot be forked until it is persisted in the transcript.')
      return
    }
    const target = actionTargetForMessage(blocks, block, turnCheckpoints)
    const body = block.kind === 'user_message' && target
      ? target.body
      : { message_index: block.messageIndex }
    void api.sessionActionFork(session.session_id, body)
      .then((forked) => {
        setActiveView('chat')
        void resumeSession(forked.info.session_id)
        void loadSessions()
        void loadTranscript(forked.info.session_id)
      })
      .catch((error) => {
        appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
      })
  }

  const rewindMessage = (block: UserMessage | AssistantMessage) => {
    if (!session || typeof block.messageIndex !== 'number') {
      if (session) appendLocalError(session.session_id, 'This message cannot be rewound until it is persisted in the transcript.')
      return
    }
    const target = block.kind === 'user_message' ? actionTargetForUser(blocks, block, turnCheckpoints) : null
    const body = target?.body ?? { message_index: block.messageIndex }
    void api.sessionActionRewind(session.session_id, body)
      .then((result) => {
        const rewind = result.result
        if (!isSessionRewindResult(rewind)) return
        replaceTranscript(rewind.session_id, rewind.messages)
        void loadSessions()
        appendLocalNotice(rewind.session_id, checkpointActionNotice(result, 'Rewound session to message ' + (rewind.rewound_to_index + 1).toLocaleString() + '.'))
      })
      .catch((error) => {
        appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
      })
  }

  const retryUserMessage = (block: UserMessage) => {
    if (!session) return
    if (block.source !== 'transcript' || typeof block.messageIndex !== 'number') {
      startRun(session.session_id, block.content, block.attachments)
      return
    }
    const target = actionTargetForMessage(blocks, block, turnCheckpoints)
    if (!target) {
      appendLocalError(session.session_id, 'Could not resolve the persisted prompt for retry.')
      return
    }
    void api.sessionActionRetry(session.session_id, target.body)
      .then((result) => {
        const rewind = result.result
        if (!isSessionRewindResult(rewind)) return
        replaceTranscript(rewind.session_id, rewind.messages)
        void loadSessions()
        appendLocalNotice(rewind.session_id, checkpointActionNotice(result, 'Prepared retry from the selected prompt.'))
        startRun(session.session_id, block.content, block.attachments)
      })
      .catch((error) => {
        appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
      })
  }

  const retryAssistantMessage = (block: AssistantMessage) => {
    if (!session || typeof block.messageIndex !== 'number') {
      if (session) appendLocalError(session.session_id, 'This reply cannot be retried until it is persisted in the transcript.')
      return
    }
    const prompt = previousUserBlockForAssistant(blocks, block)
    if (!prompt || typeof prompt.messageIndex !== 'number') {
      appendLocalError(session.session_id, 'Could not find the prompt that produced this reply.')
      return
    }
    const target = actionTargetForUser(blocks, prompt, turnCheckpoints)
    if (!target) {
      appendLocalError(session.session_id, 'Could not resolve the persisted prompt for retry.')
      return
    }
    void api.sessionActionRetry(session.session_id, target.body)
      .then((result) => {
        const rewind = result.result
        if (!isSessionRewindResult(rewind)) return
        replaceTranscript(rewind.session_id, rewind.messages)
        void loadSessions()
        appendLocalNotice(rewind.session_id, checkpointActionNotice(result, 'Prepared retry from the selected reply.'))
        startRun(session.session_id, prompt.content, prompt.attachments)
      })
      .catch((error) => {
        appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
      })
  }

  const revertFileChange = async (change: CurrentTurnFileChangeAction) => {
    if (!session) return
    const expected_hashes = change.currentHash ? { [change.path]: change.currentHash } : undefined
    if (change.checkpointId) {
      const result = await api.rejectWorkspaceChanges({ paths: [change.path], checkpoint_id: change.checkpointId, expected_hashes })
      appendLocalNotice(session.session_id, result.message || 'Rejected `' + change.path + '` from checkpoint `' + change.checkpointId + '`.')
      return
    }
    if (change.oldText == null) return
    const result = await api.rejectWorkspaceChanges({ paths: [change.path], expected_hashes })
    appendLocalNotice(session.session_id, result.message || 'Rejected `' + change.path + '`.')
  }

  const acceptFileChange = async (change: CurrentTurnFileChangeAction) => {
    if (!session) return
    const result = await api.acceptWorkspaceChanges([change.path])
    appendLocalNotice(session.session_id, result.message || 'Accepted `' + change.path + '`.')
  }

  const requestUndoTurn = (action: CurrentTurnUndoAction) => {
    setPendingTurnUndo(action)
  }

  const confirmUndoTurn = () => {
    if (!session || !pendingTurnUndo || turnUndoRunning) return
    const action = pendingTurnUndo
    setTurnUndoRunning(true)
    void api.sessionActionUndoRun(session.session_id, action.body)
      .then((result) => {
        const rewind = result.result
        if (!isSessionRewindResult(rewind)) {
          throw new Error('Unexpected undo response from session service.')
        }
        replaceTranscript(rewind.session_id, rewind.messages)
        void loadSessions()
        setTurnCheckpoints([])
        patchComposerDraft('replace', action.promptContent, action.attachments)
        appendLocalNotice(rewind.session_id, checkpointActionNotice(result, turnUndoFallbackNotice(rewind)))
        setPendingTurnUndo(null)
      })
      .catch((error) => {
        appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
        setPendingTurnUndo(null)
      })
      .finally(() => {
        setTurnUndoRunning(false)
      })
  }

  const stopTaskFromBar = async (task: TaskSummary) => {
    if (!session) return
    try {
      const result = await api.stopTask(task.task_id)
      appendLocalNotice(session.session_id, result.message)
      await loadSessionTasks(session.session_id)
    } catch (error) {
      appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
      throw error
    }
  }

  const changePermissionMode = async (mode: PermissionMode) => {
    if (!session) return
    const updated = await setSessionPermissionMode(session.session_id, mode)
    if (updated.mode) setSessionMode(updated.session_id, updated.mode === 'plan' ? 'plan' : 'agent')
    appendLocalNotice(session.session_id, 'Permission mode set to `' + mode + '`.')
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
        workspaceRootVersion={workspaceRootVersion}
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
      activeRunId,
      runStatus: activeStatus,
      tokens: usage,
      verbose: verboseMode,
      setVerbose: setVerboseMode,
      cancelRun: (targetSessionId) => cancelRun(targetSessionId),
      refreshSession: async (targetSessionId) => {
        await Promise.all([
          loadTranscript(targetSessionId),
          loadSessionTasks(targetSessionId),
          loadSessions(),
        ])
      },
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
      <SessionTaskBar tasks={tasks} onOpenTask={(task) => setSelectedTaskId(task.task_id)} onStopTask={stopTaskFromBar} />
      <div className="chat-scroll" onScroll={handleScroll} ref={scrollRef}>
        <ChatTimeline
          blocks={blocks}
          sessionId={session.session_id}
          turnCheckpoints={turnCheckpoints}
          messageActionsDisabled={Boolean(activeRunId)}
          onOpenTask={setSelectedTaskId}
          onRespondPermission={respondPermission}
          onRespondApproval={respondApproval}
          onAcceptFileChange={acceptFileChange}
          onRevertFileChange={revertFileChange}
          onUndoTurn={requestUndoTurn}
          onRetryAssistantMessage={retryAssistantMessage}
          onRetryUserMessage={retryUserMessage}
          onEditUserMessage={(block) => patchComposerDraft('replace', block.content, block.attachments)}
          onForkMessage={forkMessage}
          onRewindMessage={rewindMessage}
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
        verbose={verboseMode}
      />
      <Composer
        disabled={!session}
        running={Boolean(activeRunId)}
        sessionId={session.session_id}
        provider={session.provider}
        model={session.model}
        mode={session.mode}
        permissionMode={session.mode === 'plan' ? 'plan' : session.permission_mode ?? 'default'}
        onPermissionModeChange={changePermissionMode}
        sessionSummary={session.summary}
        messageCount={session.message_count}
        inputTokens={usage?.input_tokens}
        outputTokens={usage?.output_tokens}
        tokens={usage}
        runMetadata={latestRunMetadata}
        onSend={(message, attachments) => startRun(session.session_id, message, attachments)}
        onSlashCommand={handleSlashCommand}
        onCancel={() => cancelRun(session.session_id)}
        draftPatch={composerDraftPatch}
        workspaceRootVersion={workspaceRootVersion}
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
      {pendingTurnUndo ? (
        <ConfirmDialog
          title="Undo this turn?"
          description={turnUndoDescription(pendingTurnUndo)}
          confirmLabel={turnUndoRunning ? 'Undoing' : 'Undo turn'}
          cancelLabel="Cancel"
          onConfirm={confirmUndoTurn}
          onCancel={() => {
            if (!turnUndoRunning) setPendingTurnUndo(null)
          }}
        />
      ) : null}
    </div>
  )
}

type CheckpointActionTarget = {
  prompt: UserMessage
  body: SessionCheckpointActionBody
}

function actionTargetForMessage(
  blocks: ChatBlock[],
  block: UserMessage | AssistantMessage,
  checkpoints: SessionTurnCheckpoint[],
): CheckpointActionTarget | null {
  if (block.kind === 'user_message') return actionTargetForUser(blocks, block, checkpoints)
  const prompt = previousUserBlockForAssistant(blocks, block)
  return prompt ? actionTargetForUser(blocks, prompt, checkpoints) : null
}

function actionTargetForUser(
  blocks: ChatBlock[],
  block: UserMessage,
  checkpoints: SessionTurnCheckpoint[],
): CheckpointActionTarget | null {
  if (block.source !== 'transcript' || typeof block.messageIndex !== 'number') return null
  const userMessageIndex = userMessageIndexForBlock(blocks, block)
  if (userMessageIndex == null) return null
  const checkpoint = checkpoints.find((candidate) => (
    candidate.target.message_index === block.messageIndex ||
    candidate.target.user_message_index === userMessageIndex
  ))
  const body: SessionCheckpointActionBody = {
    user_message_index: userMessageIndex,
    expected_content: block.content,
  }
  if (checkpoint?.code.checkpoint_id) {
    body.checkpoint_id = checkpoint.code.checkpoint_id
    body.paths = checkpoint.code.files_changed
  }
  return { prompt: block, body }
}

function userMessageIndexForBlock(blocks: ChatBlock[], block: UserMessage): number | null {
  if (typeof block.messageIndex !== 'number') return null
  const persistedUsers = blocks
    .filter((candidate): candidate is UserMessage => (
      candidate.kind === 'user_message' &&
      candidate.source === 'transcript' &&
      typeof candidate.messageIndex === 'number'
    ))
    .sort((left, right) => (left.messageIndex ?? 0) - (right.messageIndex ?? 0))
  const index = persistedUsers.findIndex((candidate) => candidate.messageIndex === block.messageIndex)
  return index >= 0 ? index : null
}

function isSessionRewindResult(value: unknown): value is SessionRewindResult {
  return Boolean(
    value &&
    typeof value === 'object' &&
    'session_id' in value &&
    'messages' in value &&
    'rewound_to_index' in value,
  )
}

function checkpointActionNotice(result: SessionCheckpointActionResult, fallback: string): string {
  const restore = recordFromUnknown(result.restore)
  const checkpointId = stringValue(restore.checkpoint_id)
  const paths = Array.isArray(restore.paths)
    ? restore.paths.filter((path): path is string => typeof path === 'string' && path.trim().length > 0)
    : []
  if (!checkpointId) return fallback
  const fileCount = paths.length > 0 ? paths.length.toLocaleString() + ' file' + (paths.length === 1 ? '' : 's') : 'workspace'
  return fallback + ' Restored ' + fileCount + ' from checkpoint `' + checkpointId + '`.'
}

function turnUndoFallbackNotice(rewind: SessionRewindResult): string {
  const removed = rewind.messages_removed
  return 'Undid selected turn and returned its prompt to the composer. Removed ' + removed.toLocaleString() + ' message' + (removed === 1 ? '' : 's') + '.'
}

function turnUndoDescription(action: CurrentTurnUndoAction): string {
  const fileCount = action.paths.length
  const fileLabel = fileCount.toLocaleString() + ' file' + (fileCount === 1 ? '' : 's')
  return 'Restore ' + fileLabel + ' from checkpoint `' + (action.checkpointId || 'unknown') + '`, remove this turn from the transcript, and return the prompt to the composer.'
}

function recordFromUnknown(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function stringValue(value: unknown): string {
  return typeof value === 'string' && value.trim() ? value.trim() : ''
}

function latestRunMetadataForSession(sessionId: string, blocks: ChatBlock[]): Record<string, unknown> | undefined {
  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index]
    if (!block || block.sessionId !== sessionId) continue
    if ((block.kind === 'assistant_message' || block.kind === 'tool_result') && block.metadata && Object.keys(block.metadata).length > 0) {
      return block.metadata
    }
  }
  return undefined
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

function previousUserBlockForAssistant(blocks: ChatBlock[], assistant: AssistantMessage): UserMessage | null {
  if (typeof assistant.messageIndex !== 'number') return null
  let candidate: UserMessage | null = null
  for (const block of blocks) {
    if (block.kind !== 'user_message' || typeof block.messageIndex !== 'number') continue
    if (block.messageIndex >= assistant.messageIndex) continue
    if (!candidate || block.messageIndex > (candidate.messageIndex ?? -1)) candidate = block
  }
  return candidate
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

function readBooleanSetting(key: string, fallback: boolean): boolean {
  if (typeof window === 'undefined') return fallback
  const value = window.localStorage.getItem(key)
  if (value === null) return fallback
  return value === 'true'
}

function writeBooleanSetting(key: string, value: boolean): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(key, value ? 'true' : 'false')
}

function formatQuotedDraft(role: 'user' | 'assistant', content: string): string {
  const trimmed = content.trim()
  if (!trimmed) return ''
  const label = role === 'user' ? 'User' : 'Assistant'
  return ['> ' + label + ':', ...trimmed.split('\n').map((line) => '> ' + line)].join('\n')
}
