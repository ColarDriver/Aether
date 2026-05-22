import type { ChatBlock, DiffBlock as DiffChatBlock, ToolResultBlock as ToolResultChatBlock } from '../../chat-rendering'
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

type Props = {
  blocks: ChatBlock[]
  onRespondPermission?: (decision: Record<string, unknown>) => void
  onRespondApproval?: (result: Record<string, unknown>) => void
  onOpenTask?: (taskId: string) => void
}

export function ChatTimeline({ blocks, onRespondPermission, onRespondApproval, onOpenTask }: Props) {
  if (blocks.length === 0) {
    return <div className="empty-chat">No messages in this session yet.</div>
  }
  const model = buildChatRenderModel(blocks)
  const turns = groupRenderItemsIntoTurns(model.items)
  return (
    <div className="chat-timeline">
      {turns.map((turn) => (
        <section className={turn.hasUser ? 'chat-turn' : 'chat-turn chat-turn-orphan'} key={turn.id}>
          {turn.items.map((item) => (
            <ChatRenderItemView
              item={item}
              key={item.kind === 'tool_group' ? item.id : item.block.id}
              onRespondPermission={onRespondPermission}
              onRespondApproval={onRespondApproval}
              onOpenTask={onOpenTask}
              results={model.toolResultsByCallId}
              diffs={model.diffsByToolCallId}
            />
          ))}
          <CurrentTurnChangeCard diffs={diffsForTurn(turn, model.diffsByToolCallId)} />
        </section>
      ))}
    </div>
  )
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
  onRespondPermission,
  onRespondApproval,
  onOpenTask,
  results,
  diffs,
}: {
  item: ChatRenderItem
  onRespondPermission?: (decision: Record<string, unknown>) => void
  onRespondApproval?: (result: Record<string, unknown>) => void
  onOpenTask?: (taskId: string) => void
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
      onRespondPermission={onRespondPermission}
      onRespondApproval={onRespondApproval}
      onOpenTask={onOpenTask}
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
  onRespondPermission,
  onRespondApproval,
  onOpenTask,
}: {
  block: ChatBlock
  onRespondPermission?: (decision: Record<string, unknown>) => void
  onRespondApproval?: (result: Record<string, unknown>) => void
  onOpenTask?: (taskId: string) => void
}) {
  switch (block.kind) {
    case 'user_message':
      return <UserMessageBlock block={block} />
    case 'assistant_message':
      return <AssistantMessageBlock block={block} />
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
