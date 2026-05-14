import { existsSync } from 'node:fs'

import type { GatewayClient } from '../gatewayClient.js'
import type {
  JsonObject,
  ModelInfo,
  ProviderInfo,
  SessionInfo,
  SlashCommandInfo,
  TranscriptMessage
} from '../gatewayTypes.js'
import type { SessionState } from '../store/sessionStore.js'
import { clearCommand } from './commands/clear.js'
import { exitCommand } from './commands/exit.js'
import { helpCommand } from './commands/help.js'
import { interruptCommand } from './commands/interrupt.js'
import { modelCommand } from './commands/model.js'
import { newCommand } from './commands/new.js'
import { refreshCommand } from './commands/refresh.js'
import { resumeCommand } from './commands/resume.js'
import { sessionCommand, sessionsCommand } from './commands/session.js'
import { statsCommand } from './commands/stats.js'
import { systemCommand } from './commands/system.js'
import { toolsCommand } from './commands/tools.js'
import { verboseCommand } from './commands/verbose.js'

export type SlashCategory = 'local' | 'remote' | 'control'

export type SlashCommand = {
  name: string
  category: SlashCategory
  execute(args: string[], ctx: SlashCtx): Promise<SlashResult>
}

export type SlashResult =
  | { kind: 'note'; text: string; level: 'success' | 'info' | 'warn' | 'error' }
  | { kind: 'replace-history'; messages: TranscriptMessage[]; info?: SessionInfo }
  | { kind: 'session'; info: SessionInfo; note?: string }
  | { kind: 'clear' }
  | { kind: 'refresh' }
  | { kind: 'toggle-verbose'; enabled: boolean }
  | { kind: 'exit' }
  | { kind: 'noop' }

export type SlashCtx = {
  client: Pick<GatewayClient, 'request'>
  catalog: SlashCommandInfo[]
  getSession(): SessionState
  createSession(input?: {
    provider?: string
    model?: string
    system?: string | null
    sessionId?: string | null
    baseUrl?: string | null
  }): Promise<SessionInfo>
  setSystemOverride(text: string | null): void
  toggleVerbose(): boolean
}

export type ParsedSlash = {
  name: string
  args: string[]
}

export const slashCommands: SlashCommand[] = [
  helpCommand,
  exitCommand,
  clearCommand,
  refreshCommand,
  sessionCommand,
  sessionsCommand,
  newCommand,
  systemCommand,
  modelCommand,
  statsCommand,
  resumeCommand,
  verboseCommand,
  interruptCommand,
  toolsCommand
]

const commandMap = new Map(slashCommands.map((command) => [command.name, command]))
const COMMAND_NAME_RE = /^[a-zA-Z0-9:_-]+$/

export async function dispatchSlash(input: string, ctx: SlashCtx): Promise<SlashResult> {
  const parsed = parseSlash(input)
  if (!parsed) {
    return { kind: 'noop' }
  }

  const command = commandMap.get(parsed.name)
  if (!command) {
    return {
      kind: 'note',
      level: 'warn',
      text: `unknown command: ${parsed.name}. Try /help.`
    }
  }

  try {
    return await command.execute(parsed.args, ctx)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return { kind: 'note', level: 'error', text: message }
  }
}

export function parseSlash(input: string): ParsedSlash | null {
  const trimmed = input.trim()
  if (!isSlashCommandLine(trimmed)) {
    return null
  }
  const parts = splitArgs(trimmed)
  const [name, ...args] = parts
  if (!name) {
    return null
  }
  return { name, args }
}

export function isSlashCommandLine(input: string): boolean {
  const trimmed = input.trim()
  if (!trimmed.startsWith('/') || trimmed.length < 2) {
    return false
  }
  const head = trimmed.slice(1).split(/\s+/, 1)[0]
  if (!head || !COMMAND_NAME_RE.test(head)) {
    return false
  }
  try {
    if (existsSync(`/${head}`)) {
      return false
    }
  } catch {
    return true
  }
  return true
}

export function splitArgs(input: string): string[] {
  const args: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaping = false

  for (const char of input) {
    if (escaping) {
      current += char
      escaping = false
      continue
    }
    if (char === '\\') {
      escaping = true
      continue
    }
    if (quote) {
      if (char === quote) {
        quote = null
      } else {
        current += char
      }
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (/\s/.test(char)) {
      if (current) {
        args.push(current)
        current = ''
      }
      continue
    }
    current += char
  }

  if (current) {
    args.push(current)
  }
  return args
}

export function formatCommandCatalog(catalog: SlashCommandInfo[]): string {
  if (catalog.length === 0) {
    return 'No commands advertised by gateway.'
  }
  const width = Math.max(...catalog.map((command) => command.name.length), 5)
  return catalog
    .map((command) => {
      const category = command.category ? ` [${command.category}]` : ''
      return `${command.name.padEnd(width)}  ${command.description}${category}`
    })
    .join('\n')
}

export type CommandsCatalogResult = {
  commands: SlashCommandInfo[]
}

export type ProvidersListResult = {
  providers: ProviderInfo[]
}

export type ModelDiscovery = {
  /** `live` when /v1/models returned a usable list, `static` for the fallback. */
  kind: 'live' | 'static'
  source?: string
  base_url?: string
  base_url_source?: 'param' | 'session' | 'env' | 'default'
  count?: number
  /** URL we successfully fetched model ids from (live path only). */
  url?: string
  reason?: 'no_credentials' | 'list_models_error' | 'empty_response' | string
  error?: string
  /** First 200 chars of the failing response body — shown to help debug. */
  body_preview?: string
  /**
   * When discovery succeeded at `…/v1/models` but the configured base_url
   * does not end with `/v1`, the gateway emits the corrected URL here so
   * the user can fix `OPENAI_BASE_URL` before chat calls 404.
   */
  suggested_base_url?: string
  /** Free-text warning that pairs with `suggested_base_url`. */
  warning?: string
}

export type ProvidersModelsResult = {
  models: ModelInfo[]
  discovery?: ModelDiscovery
}

export type SessionCreateResult = {
  session_id: string
  info: SessionInfo
}

export type SessionUpdateResult = {
  session_id: string
  info: SessionInfo
}

export type SessionCurrentResult = {
  session_id: string | null
  info?: SessionInfo
}

export type SessionListResult = {
  sessions: SessionInfo[]
}

export type SessionResumeResult = {
  info: SessionInfo
  messages: TranscriptMessage[]
}

export type PrefsGetResult<T = unknown> = {
  value: T | null
}

export type AgentRunResult = {
  final_text: string
  exit_reason: string
  usage?: JsonObject
  metadata?: JsonObject
}

export type ToolsListResult = {
  tools: ToolInfo[]
}

export type ToolInfo = {
  name: string
  description: string
  parameters: JsonObject
  required: string[]
}
