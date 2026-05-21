export type ChatBlockSource = 'transcript' | 'live' | 'optimistic'

export type ChatBlockBase = {
  id: string
  sessionId: string
  runId?: string | null
  timestamp: number
  source: ChatBlockSource
}

export type ChatAttachment = {
  type: 'file' | 'image' | 'text'
  name?: string
  path?: string
  mimeType?: string
  data?: string
}

export type TokenUsage = {
  input_tokens?: number
  output_tokens?: number
  cache_read_tokens?: number
  cache_write_tokens?: number
  reasoning_tokens?: number
  total_tokens?: number
}

export type ToolStatus = 'pending' | 'running' | 'finished' | 'failed'

export type PermissionPreview = {
  title?: string
  subtitle?: string | null
  body?: string | null
  diff?: string | null
  path?: string | null
  command?: string | null
  metadata?: Record<string, unknown>
}

export type AskUserQuestionOption = {
  id?: string
  label: string
  description?: string
}

export type AskUserQuestion = {
  id?: string
  question: string
  header?: string
  options?: AskUserQuestionOption[]
  multiSelect?: boolean
  freeText?: boolean
}

export type PromptResolution = {
  promptId: string
  state: 'pending' | 'allowed' | 'denied' | 'approved' | 'rejected' | 'answered' | 'expired' | 'aborted'
  answers?: Record<string, string>
}

export type DiffOrigin = 'permission_preview' | 'tool_result' | 'transcript'

export type DiffContent = {
  path?: string | null
  diff?: string | null
  oldText?: string | null
  newText?: string | null
  language?: string | null
}

export type UserMessageBlock = ChatBlockBase & {
  kind: 'user_message'
  content: string
  attachments?: ChatAttachment[]
  pending?: boolean
}

export type AssistantMessageBlock = ChatBlockBase & {
  kind: 'assistant_message'
  content: string
  isStreaming?: boolean
  isError?: boolean
  model?: string | null
}

export type ThinkingBlock = ChatBlockBase & {
  kind: 'thinking'
  content: string
  isActive?: boolean
  sequence?: number
}

export type ToolCallBlock = ChatBlockBase & {
  kind: 'tool_call'
  toolCallId: string
  toolName: string
  arguments: Record<string, unknown>
  status: ToolStatus
  iteration?: number
  parentToolCallId?: string | null
}

export type ToolResultBlock = ChatBlockBase & {
  kind: 'tool_result'
  toolCallId: string
  toolName?: string | null
  content: string
  isError: boolean
  metadata: Record<string, unknown>
}

export type DiffBlock = ChatBlockBase & DiffContent & {
  kind: 'diff'
  origin: DiffOrigin
}

export type PermissionRequestBlock = ChatBlockBase & {
  kind: 'permission_request'
  promptId: string
  toolCallId?: string | null
  toolName: string
  arguments: Record<string, unknown>
  category?: string | null
  risk?: string | null
  reason?: string | null
  preview?: PermissionPreview | null
  allowSession?: boolean
  state: 'pending' | 'allowed' | 'denied' | 'expired' | 'aborted'
}

export type ApprovalRequestBlock = ChatBlockBase & {
  kind: 'approval_request'
  promptId: string
  approvalKind: 'plan' | 'questions' | string
  planText?: string | null
  planPath?: string | null
  questions: AskUserQuestion[]
  state: 'pending' | 'approved' | 'rejected' | 'answered' | 'expired'
}

export type AskUserQuestionBlock = ChatBlockBase & {
  kind: 'ask_user_question'
  promptId?: string | null
  toolCallId?: string | null
  questions: AskUserQuestion[]
  answers?: Record<string, string>
  state: 'pending' | 'answered' | 'cancelled'
}

export type StreamingStatusBlock = ChatBlockBase & {
  kind: 'streaming_status'
  state: 'thinking' | 'responding' | 'tool_use' | 'idle' | string
  detail?: string | null
  elapsedMs?: number
  tokens?: TokenUsage
}

export type SystemNoticeBlock = ChatBlockBase & {
  kind: 'system_notice'
  content: string
}

export type ErrorBlock = ChatBlockBase & {
  kind: 'error'
  message: string
  code?: string | null
}

export type ChatBlock =
  | UserMessageBlock
  | AssistantMessageBlock
  | ThinkingBlock
  | ToolCallBlock
  | ToolResultBlock
  | DiffBlock
  | PermissionRequestBlock
  | ApprovalRequestBlock
  | AskUserQuestionBlock
  | StreamingStatusBlock
  | SystemNoticeBlock
  | ErrorBlock

export type ChatBlockKind = ChatBlock['kind']
