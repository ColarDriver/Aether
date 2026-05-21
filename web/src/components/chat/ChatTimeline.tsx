import type { ChatBlock } from '../../chat-rendering'
import { buildChatRenderModel } from '../../chat-rendering/renderModel'
import {
  ApprovalRequestBlock,
  AskUserQuestionBlock,
  AssistantMessageBlock,
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
}

export function ChatTimeline({ blocks, onRespondPermission, onRespondApproval }: Props) {
  if (blocks.length === 0) {
    return <div className="empty-chat">No messages in this session yet.</div>
  }
  const model = buildChatRenderModel(blocks)
  return (
    <div className="chat-timeline">
      {model.items.map((item) => {
        if (item.kind === 'tool_group') {
          return (
            <ToolCallGroup
              key={item.id}
              toolCalls={item.toolCalls}
              results={model.toolResultsByCallId}
              diffs={model.diffsByToolCallId}
            />
          )
        }
        return (
          <ChatBlockView
            block={item.block}
            key={item.block.id}
            onRespondPermission={onRespondPermission}
            onRespondApproval={onRespondApproval}
          />
        )
      })}
    </div>
  )
}

function ChatBlockView({
  block,
  onRespondPermission,
  onRespondApproval,
}: {
  block: ChatBlock
  onRespondPermission?: (decision: Record<string, unknown>) => void
  onRespondApproval?: (result: Record<string, unknown>) => void
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
    case 'permission_request':
      return <PermissionRequestBlock block={block} onRespond={onRespondPermission} />
    case 'approval_request':
      return <ApprovalRequestBlock block={block} onRespond={onRespondApproval} />
    case 'ask_user_question':
      return <AskUserQuestionBlock block={block} />
    case 'streaming_status':
      return <StreamingStatusBlock block={block} />
    case 'task_notification':
      return <TaskNotificationBlock block={block} />
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
