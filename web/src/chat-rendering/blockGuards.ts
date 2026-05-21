import type {
  ApprovalRequestBlock,
  AskUserQuestionBlock,
  ChatBlock,
  ChatBlockKind,
  PermissionRequestBlock,
  ToolCallBlock,
  ToolResultBlock,
} from './blocks'

export function isBlockKind<K extends ChatBlockKind>(
  block: ChatBlock,
  kind: K,
): block is Extract<ChatBlock, { kind: K }> {
  return block.kind === kind
}

export function isToolCallBlock(block: ChatBlock): block is ToolCallBlock {
  return block.kind === 'tool_call'
}

export function isToolResultBlock(block: ChatBlock): block is ToolResultBlock {
  return block.kind === 'tool_result'
}

export function isPermissionRequestBlock(block: ChatBlock): block is PermissionRequestBlock {
  return block.kind === 'permission_request'
}

export function isApprovalRequestBlock(block: ChatBlock): block is ApprovalRequestBlock {
  return block.kind === 'approval_request'
}

export function isAskUserQuestionBlock(block: ChatBlock): block is AskUserQuestionBlock {
  return block.kind === 'ask_user_question'
}

export function isPromptBlock(
  block: ChatBlock,
): block is PermissionRequestBlock | ApprovalRequestBlock | AskUserQuestionBlock {
  return (
    block.kind === 'permission_request' ||
    block.kind === 'approval_request' ||
    block.kind === 'ask_user_question'
  )
}
