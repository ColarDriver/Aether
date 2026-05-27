import type { SessionTurnCheckpoint } from '../../api/types'
import type { AssistantMessageBlock as AssistantMessage, ChatBlock, DiagnosticsBlock as DiagnosticsChatBlock, DiffBlock as DiffChatBlock, ToolResultBlock as ToolResultChatBlock, UserMessageBlock as UserMessage } from '../../chat-rendering'
import { buildChatRenderModel, type ChatRenderItem } from '../../chat-rendering/renderModel'
import {
  ApprovalRequestBlock,
  AskUserQuestionBlock,
  AssistantMessageBlock,
  CurrentTurnChangeCard,
  DiagnosticsBlock,
  DiffBlock,
  ErrorBlock,
  PermissionRequestBlock,
  StreamingStatusBlock,
  SystemNoticeBlock,
  TaskNotificationBlock,
  ThinkingBlock,
  ToolCallGroup,
  ToolResultBlock,
  UserMessageBlock,
} from './blocks'
import type { CurrentTurnCheckpoint, CurrentTurnFileChangeAction, CurrentTurnUndoAction, CurrentTurnVerification } from './blocks/CurrentTurnChangeCard'

type Props = {
  blocks: ChatBlock[]
  sessionId?: string | null
  turnCheckpoints?: SessionTurnCheckpoint[]
  messageActionsDisabled?: boolean
  onRespondPermission?: (decision: Record<string, unknown>) => void
  onRespondApproval?: (result: Record<string, unknown>) => void
  onOpenTask?: (taskId: string) => void
  onAcceptFileChange?: (change: CurrentTurnFileChangeAction) => Promise<void> | void
  onRevertFileChange?: (change: CurrentTurnFileChangeAction) => Promise<void> | void
  onUndoTurn?: (action: CurrentTurnUndoAction) => void
  onRetryAssistantMessage?: (block: AssistantMessage) => void
  onRetryUserMessage?: (block: UserMessage) => void
  onEditUserMessage?: (block: UserMessage) => void
  onForkMessage?: (block: UserMessage | AssistantMessage) => void
  onRewindMessage?: (block: UserMessage | AssistantMessage) => void
  onQuoteUserMessage?: (block: UserMessage) => void
  onQuoteAssistantMessage?: (block: AssistantMessage) => void
}

export function ChatTimeline({ blocks, sessionId = null, turnCheckpoints = [], messageActionsDisabled = false, onRespondPermission, onRespondApproval, onOpenTask, onAcceptFileChange, onRevertFileChange, onUndoTurn, onRetryAssistantMessage, onRetryUserMessage, onEditUserMessage, onForkMessage, onRewindMessage, onQuoteUserMessage, onQuoteAssistantMessage }: Props) {
  if (blocks.length === 0) {
    return <div className="empty-chat">No messages in this session yet.</div>
  }
  const model = buildChatRenderModel(blocks)
  const turns = groupRenderItemsIntoTurns(model.items)
  return (
    <div className="chat-timeline">
      {turns.map((turn) => {
        const serverCheckpoint = checkpointForTurn(turn, turnCheckpoints)
        return (
          <section className={turn.hasUser ? 'chat-turn' : 'chat-turn chat-turn-orphan'} key={turn.id}>
            {turn.items.map((item) => (
              <ChatRenderItemView
                item={item}
                key={item.kind === 'tool_group' ? item.id : item.block.id}
                messageActionsDisabled={messageActionsDisabled}
                onRespondPermission={onRespondPermission}
                onRespondApproval={onRespondApproval}
                onOpenTask={onOpenTask}
                onRetryAssistantMessage={onRetryAssistantMessage}
                onRetryUserMessage={onRetryUserMessage}
                onEditUserMessage={onEditUserMessage}
                onForkMessage={onForkMessage}
                onRewindMessage={onRewindMessage}
                onQuoteUserMessage={onQuoteUserMessage}
                onQuoteAssistantMessage={onQuoteAssistantMessage}
                results={model.toolResultsByCallId}
                diffs={model.diffsByToolCallId}
              />
            ))}
            <CurrentTurnChangeCard
              diffs={diffsForTurn(turn, model.diffsByToolCallId)}
              diagnostics={diagnosticsForTurn(turn)}
              verifications={verificationsForTurn(turn, model.toolResultsByCallId)}
              checkpoint={workspaceCheckpointForTurn(turn)}
              sessionId={sessionId}
              serverCheckpoint={serverCheckpoint}
              undoAction={undoActionForTurn(turn, serverCheckpoint)}
              undoDisabled={messageActionsDisabled}
              onAcceptFile={onAcceptFileChange}
              onRevertFile={onRevertFileChange}
              onUndoTurn={onUndoTurn}
            />
          </section>
        )
      })}
    </div>
  )
}

function verificationsForTurn(turn: ChatTurn, results: Map<string, ToolResultChatBlock>): CurrentTurnVerification[] {
  const verifications: CurrentTurnVerification[] = []
  const seen = new Set<string>()
  for (const item of turn.items) {
    if (item.kind === 'tool_group') {
      for (const toolCall of item.toolCalls) {
        const result = results.get(toolCall.toolCallId)
        const verification = verificationFromTool(toolCall.toolName, toolCall.toolCallId, toolCall.arguments, result)
        if (!verification || seen.has(verification.id)) continue
        seen.add(verification.id)
        verifications.push(verification)
      }
      continue
    }
    if (item.block.kind === 'tool_result') {
      const verification = verificationFromTool(item.block.toolName || 'tool', item.block.toolCallId, {}, item.block)
      if (!verification || seen.has(verification.id)) continue
      seen.add(verification.id)
      verifications.push(verification)
    }
  }
  return verifications
}

function verificationFromTool(
  toolName: string,
  toolCallId: string,
  args: Record<string, unknown>,
  result?: ToolResultChatBlock,
): CurrentTurnVerification | null {
  if (!result) return null
  const normalized = toolName.toLowerCase()
  const command = (
    stringValue(args.command)
    || stringValue(args.cmd)
    || stringValue(args.script)
    || stringValue(result.metadata.command)
    || stringValue(result.metadata.cmd)
    || stringValue(result.metadata.script)
  )
  if (isTerminalTool(normalized) && !isVerificationCommand(command)) return null
  if (!isTerminalTool(normalized) && !isLspTool(normalized)) return null
  const exitCode = numberValue(result.metadata.exit_code) ?? numberValue(result.metadata.exitCode)
  const durationMs = numberValue(result.metadata.duration_ms) ?? numberValue(result.metadata.durationMs)
  const status = result.isError || (exitCode != null && exitCode !== 0) ? 'failed' : 'passed'
  return {
    id: 'verification-' + toolCallId,
    toolName,
    label: verificationLabel(normalized, command),
    command: command || null,
    status,
    exitCode,
    durationMs,
    summary: verificationSummary(result.content),
  }
}

function isTerminalTool(toolName: string): boolean {
  return ['bash', 'shell', 'exec', 'exec_command', 'terminal', 'run_shell'].includes(toolName)
}

function isLspTool(toolName: string): boolean {
  return ['lsp', 'language_server', 'diagnostics'].includes(toolName)
}

function isVerificationCommand(command: string): boolean {
  if (!command) return false
  return /\b(test|pytest|vitest|jest|mocha|ctest|typecheck|lint|build|tsc|mypy|ruff|eslint|prettier|biome)\b/i.test(command)
    || /\b(go test|go vet|cargo test|cargo check|uv run|npm run|pnpm|yarn)\b/i.test(command)
}

function verificationLabel(toolName: string, command: string): string {
  if (isLspTool(toolName)) return 'Language diagnostics'
  if (!command) return 'Command check'
  if (/\b(typecheck|tsc)\b/i.test(command)) return 'Typecheck'
  if (/\b(lint|eslint|ruff|biome)\b/i.test(command)) return 'Lint'
  if (/\b(build)\b/i.test(command)) return 'Build'
  if (/\b(test|pytest|vitest|jest|mocha|ctest|go test|cargo test)\b/i.test(command)) return 'Tests'
  return 'Command check'
}

function verificationSummary(content: string): string | null {
  const lines = content.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  const line = lines.find((item) => !item.startsWith('[exit ') && !item.startsWith('[stderr]')) || lines[0]
  if (!line) return null
  return line.length > 180 ? line.slice(0, 177) + '...' : line
}

function numberValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

function workspaceCheckpointForTurn(turn: ChatTurn): CurrentTurnCheckpoint | null {
  for (let index = turn.items.length - 1; index >= 0; index -= 1) {
    const item = turn.items[index]
    if (!item || item.kind !== 'block') continue
    if (item.block.kind !== 'assistant_message' && item.block.kind !== 'tool_result') continue
    const checkpoint = checkpointFromMetadata(item.block.metadata)
    if (checkpoint) return checkpoint
  }
  return null
}

function checkpointForTurn(turn: ChatTurn, checkpoints: SessionTurnCheckpoint[]): SessionTurnCheckpoint | null {
  if (checkpoints.length === 0) return null
  const user = userMessageForTurn(turn)
  if (!user || typeof user.messageIndex !== 'number') return null
  return checkpoints.find((checkpoint) => checkpoint.target.message_index === user.messageIndex) ?? null
}

function userMessageForTurn(turn: ChatTurn): UserMessage | null {
  for (const item of turn.items) {
    if (item.kind === 'block' && item.block.kind === 'user_message') return item.block
  }
  return null
}

function undoActionForTurn(turn: ChatTurn, checkpoint: SessionTurnCheckpoint | null): CurrentTurnUndoAction | null {
  const user = userMessageForTurn(turn)
  if (!user || !checkpoint?.code.available || !checkpoint.code.checkpoint_id || checkpoint.code.files_changed.length === 0) return null
  return {
    body: {
      target_user_message_id: checkpoint.target.target_user_message_id,
      user_message_index: checkpoint.target.user_message_index,
      expected_content: user.content,
      checkpoint_id: checkpoint.code.checkpoint_id,
      paths: checkpoint.code.files_changed,
    },
    promptContent: user.content,
    attachments: user.attachments ?? [],
    checkpointId: checkpoint.code.checkpoint_id,
    paths: checkpoint.code.files_changed,
  }
}

function checkpointFromMetadata(metadata: Record<string, unknown> | undefined): CurrentTurnCheckpoint | null {
  const direct = recordFromUnknown(metadata?.workspace_checkpoint)
  const turn = recordFromUnknown(metadata?.turn)
  const nested = recordFromUnknown(turn.workspace_checkpoint)
  const checkpoint = Object.keys(direct).length > 0 ? direct : nested
  const checkpointId = stringValue(checkpoint.checkpoint_id) || stringValue(checkpoint.checkpointId)
  if (!checkpointId) return null
  return {
    checkpointId,
    label: stringValue(checkpoint.label) || null,
    files: checkpointFiles(checkpoint.files),
  }
}

function checkpointFiles(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => recordFromUnknown(item).path)
    .filter((path): path is string => typeof path === 'string' && path.length > 0)
}

function recordFromUnknown(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function stringValue(value: unknown): string {
  return typeof value === 'string' && value.trim() ? value.trim() : ''
}

function diagnosticsForTurn(turn: ChatTurn): DiagnosticsChatBlock[] {
  const collected: DiagnosticsChatBlock[] = []
  for (const item of turn.items) {
    if (item.kind === 'block' && item.block.kind === 'diagnostics') collected.push(item.block)
  }
  return collected
}

function diffsForTurn(turn: ChatTurn, diffs: Map<string, DiffChatBlock[]>): DiffChatBlock[] {
  const collected: DiffChatBlock[] = []
  for (const item of turn.items) {
    if (item.kind === 'tool_group') {
      for (const toolCall of item.toolCalls) collected.push(...(diffs.get(toolCall.toolCallId) ?? []))
      continue
    }
    if (item.block.kind === 'diff') collected.push(item.block)
  }
  const seen = new Set<string>()
  return collected.filter((diff) => {
    if (seen.has(diff.id)) return false
    seen.add(diff.id)
    return true
  })
}

function ChatRenderItemView({
  item,
  messageActionsDisabled,
  onRespondPermission,
  onRespondApproval,
  onOpenTask,
  onRetryAssistantMessage,
  onRetryUserMessage,
  onEditUserMessage,
  onForkMessage,
  onRewindMessage,
  onQuoteUserMessage,
  onQuoteAssistantMessage,
  results,
  diffs,
}: {
  item: ChatRenderItem
  messageActionsDisabled?: boolean
  onRespondPermission?: (decision: Record<string, unknown>) => void
  onRespondApproval?: (result: Record<string, unknown>) => void
  onOpenTask?: (taskId: string) => void
  onRetryAssistantMessage?: (block: AssistantMessage) => void
  onRetryUserMessage?: (block: UserMessage) => void
  onEditUserMessage?: (block: UserMessage) => void
  onForkMessage?: (block: UserMessage | AssistantMessage) => void
  onRewindMessage?: (block: UserMessage | AssistantMessage) => void
  onQuoteUserMessage?: (block: UserMessage) => void
  onQuoteAssistantMessage?: (block: AssistantMessage) => void
  results: Map<string, ToolResultChatBlock>
  diffs: Map<string, DiffChatBlock[]>
}) {
  if (item.kind === 'tool_group') {
    return (
      <ToolCallGroup
        toolCalls={item.toolCalls}
        results={results}
        diffs={diffs}
      />
    )
  }
  return (
    <ChatBlockView
      block={item.block}
      messageActionsDisabled={messageActionsDisabled}
      onRespondPermission={onRespondPermission}
      onRespondApproval={onRespondApproval}
      onOpenTask={onOpenTask}
      onRetryAssistantMessage={onRetryAssistantMessage}
      onRetryUserMessage={onRetryUserMessage}
      onEditUserMessage={onEditUserMessage}
      onForkMessage={onForkMessage}
      onRewindMessage={onRewindMessage}
      onQuoteUserMessage={onQuoteUserMessage}
      onQuoteAssistantMessage={onQuoteAssistantMessage}
    />
  )
}

type ChatTurn = {
  id: string
  hasUser: boolean
  items: ChatRenderItem[]
}

function groupRenderItemsIntoTurns(items: ChatRenderItem[]): ChatTurn[] {
  const turns: ChatTurn[] = []
  let current: ChatTurn | null = null

  for (const item of items) {
    const startsTurn = item.kind === 'block' && item.block.kind === 'user_message'
    if (startsTurn || !current) {
      current = {
        id: startsTurn ? 'turn-' + item.block.id : 'turn-orphan-' + turns.length,
        hasUser: startsTurn,
        items: [],
      }
      turns.push(current)
    }
    current.items.push(item)
  }

  return turns
}

function ChatBlockView({
  block,
  messageActionsDisabled,
  onRespondPermission,
  onRespondApproval,
  onOpenTask,
  onRetryAssistantMessage,
  onRetryUserMessage,
  onEditUserMessage,
  onForkMessage,
  onRewindMessage,
  onQuoteUserMessage,
  onQuoteAssistantMessage,
}: {
  block: ChatBlock
  messageActionsDisabled?: boolean
  onRespondPermission?: (decision: Record<string, unknown>) => void
  onRespondApproval?: (result: Record<string, unknown>) => void
  onOpenTask?: (taskId: string) => void
  onRetryAssistantMessage?: (block: AssistantMessage) => void
  onRetryUserMessage?: (block: UserMessage) => void
  onEditUserMessage?: (block: UserMessage) => void
  onForkMessage?: (block: UserMessage | AssistantMessage) => void
  onRewindMessage?: (block: UserMessage | AssistantMessage) => void
  onQuoteUserMessage?: (block: UserMessage) => void
  onQuoteAssistantMessage?: (block: AssistantMessage) => void
}) {
  switch (block.kind) {
    case 'user_message':
      return (
        <UserMessageBlock
          block={block}
          actionsDisabled={messageActionsDisabled}
          onEdit={onEditUserMessage}
          onFork={onForkMessage}
          onQuote={onQuoteUserMessage}
          onRewind={onRewindMessage}
          onRetry={onRetryUserMessage}
        />
      )
    case 'assistant_message':
      return <AssistantMessageBlock block={block} actionsDisabled={messageActionsDisabled} onFork={onForkMessage} onQuote={onQuoteAssistantMessage} onRetry={onRetryAssistantMessage} onRewind={onRewindMessage} />
    case 'thinking':
      return <ThinkingBlock block={block} />
    case 'tool_result':
      return <ToolResultBlock block={block} />
    case 'diff':
      return <DiffBlock block={block} />
    case 'diagnostics':
      return <DiagnosticsBlock block={block} />
    case 'permission_request':
      return <PermissionRequestBlock block={block} onRespond={onRespondPermission} />
    case 'approval_request':
      return <ApprovalRequestBlock block={block} onRespond={onRespondApproval} />
    case 'ask_user_question':
      return <AskUserQuestionBlock block={block} />
    case 'streaming_status':
      return <StreamingStatusBlock block={block} />
    case 'task_notification':
      return <TaskNotificationBlock block={block} onOpenTask={onOpenTask} />
    case 'system_notice':
      return <SystemNoticeBlock block={block} />
    case 'error':
      return <ErrorBlock block={block} />
    case 'tool_call':
      return (
        <ToolCallGroup
          toolCalls={[block]}
          results={new Map()}
          diffs={new Map()}
        />
      )
  }
}
