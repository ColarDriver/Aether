export type JsonPrimitive = string | number | boolean | null
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[]
export type JsonObject = { [key: string]: JsonValue | undefined }
export type RpcId = string | number

export interface RpcError {
  code: number
  message: string
  data?: JsonObject | null
}

export interface RpcRequest {
  jsonrpc: '2.0'
  id: RpcId
  method: string
  params?: JsonObject | null
}

export interface RpcNotification {
  jsonrpc: '2.0'
  method: string
  params?: JsonObject | null
}

export interface RpcSuccessResponse<T = unknown> {
  jsonrpc: '2.0'
  id: RpcId | null
  result: T
}

export interface RpcErrorResponse {
  jsonrpc: '2.0'
  id: RpcId | null
  error: RpcError
}

export type RpcResponse<T = unknown> = RpcSuccessResponse<T> | RpcErrorResponse
export type RpcFrame<T = unknown> = RpcRequest | RpcNotification | RpcResponse<T>

export interface GatewayReady {
  type: 'gateway.ready'
  version: string
  capabilities: string[]
  methods?: string[]
}

export interface GatewayErrorEvent {
  type: 'gateway.error'
  message: string
  where?: string | null
}

export interface AgentEventBase {
  session_id: string
  run_id: string
}

export interface TextDelta extends AgentEventBase {
  type: 'text.delta'
  text: string
  sequence: number
}

export interface StreamProgress extends AgentEventBase {
  type: 'stream.progress'
  chars: number
  sequence: number
}

export interface Reasoning extends AgentEventBase {
  type: 'reasoning.delta'
  text: string
  sequence: number
}

export interface ToolCall extends AgentEventBase {
  type: 'tool.call'
  tool_call_id: string
  tool_name: string
  arguments: JsonObject
  iteration: number
}

export interface ToolResultMetadata {
  // file tools (write_file / file_edit)
  path?: string
  bytes_before?: number
  bytes_after?: number
  size_bytes?: number
  lines_added?: number
  lines_removed?: number
  hunks?: number
  diff?: string
  change_count?: number
  replace_all?: boolean
  existed?: boolean
  sha256?: string
  no_op?: boolean
  // shell tool
  exit_code?: number
  duration_ms?: number
  cwd?: string | null
  command?: string
  truncated?: boolean
  timed_out?: boolean
  interrupted?: boolean
  stderr_lines?: number
  // free-form fallback so unknown tools can still ship structured data
  [key: string]: unknown
}

export interface ToolResult extends AgentEventBase {
  type: 'tool.result'
  tool_call_id: string
  tool_name: string
  content: string
  is_error?: boolean
  iteration: number
  metadata?: ToolResultMetadata
}

export interface IterationStart extends AgentEventBase {
  type: 'iteration.start'
  iteration: number
}

export interface IterationEnd extends AgentEventBase {
  type: 'iteration.end'
  iteration: number
}

export interface LoopStateChanged extends AgentEventBase {
  type: 'loop.state'
  state: string
}

export interface Status extends AgentEventBase {
  type: 'status'
  kind: 'thinking' | 'responding' | 'tool_use' | 'idle'
  detail?: string | null
}

export interface TokenUsage extends AgentEventBase {
  type: 'usage'
  input_tokens: number
  output_tokens: number
  cache_read_tokens?: number
  cache_write_tokens?: number
}

export interface Done extends AgentEventBase {
  type: 'done'
  final_text?: string
  exit_reason?: string
}

export interface Cancelled extends AgentEventBase {
  type: 'cancelled'
  reason?: string | null
  partial_text?: string
}

export interface ErrorEvent extends AgentEventBase {
  type: 'error'
  message: string
}

export interface ApprovalQuestion {
  id: string
  text: string
  kind?: 'open' | 'select'
  options?: string[]
}

export interface ApprovalRequestParams {
  kind: 'plan' | 'questions'
  session_id: string
  run_id: string
  tool_call_id?: string | null
  plan_text?: string | null
  plan_path?: string | null
  questions?: ApprovalQuestion[]
  deadline_ms: number
}

export interface PermissionPreview {
  title: string
  subtitle?: string | null
  body?: string | null
  diff?: string | null
  path?: string | null
  command?: string | null
  metadata?: JsonObject
}

export interface PermissionToolRequest {
  tool_call_id: string
  tool_name: string
  arguments: JsonObject
  category: string
  risk: string
  preview?: PermissionPreview | null
  reason?: string | null
  allow_session?: boolean
}

export interface PermissionRequestParams {
  session_id: string
  run_id: string
  request: PermissionToolRequest
  deadline_ms: number
}

export interface GatewayStderr {
  type: 'gateway.stderr'
  line: string
}

export interface GatewayProtocolError {
  type: 'gateway.protocol_error'
  raw: string
  reason: string
}

export interface GatewayServerRequest<TParams = unknown> {
  type: 'gateway.server_request'
  id: string
  method: string
  params: TParams
}

export type ApprovalRequestEvent = GatewayServerRequest<ApprovalRequestParams> & {
  method: 'approval.request'
}

export type PermissionRequestEvent = GatewayServerRequest<PermissionRequestParams> & {
  method: 'permission.request'
}

export type GatewayEvent =
  | GatewayReady
  | GatewayErrorEvent
  | TextDelta
  | StreamProgress
  | Reasoning
  | ToolCall
  | ToolResult
  | IterationStart
  | IterationEnd
  | LoopStateChanged
  | Status
  | TokenUsage
  | Done
  | Cancelled
  | ErrorEvent
  | GatewayStderr
  | GatewayProtocolError
  | GatewayServerRequest

export interface SessionInfo {
  session_id: string
  created_at: number
  updated_at: number
  provider: string
  model: string
  base_url?: string | null
  system_prompt?: string | null
  message_count?: number
  summary?: string | null
  mode?: 'agent' | 'plan' | string | null
}

export interface TranscriptToolCall {
  id: string
  name: string
  arguments: JsonObject
}

export interface TranscriptMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  text?: string | null
  name?: string | null
  tool_call_id?: string | null
  tool_calls?: TranscriptToolCall[]
  is_error?: boolean
  metadata?: JsonObject | null
}

export interface ProviderInfo {
  name: string
  display_name: string
  requires_api_key: boolean
  default_base_url?: string | null
}

export interface ModelInfo {
  id: string
  display_name: string
  context_window?: number | null
}

export interface SlashCommandInfo {
  name: string
  description: string
  category?: string | null
}

export interface GatewayPingResult {
  pong: true
  timestamp: number
  echo?: JsonValue
}
