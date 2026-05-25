import { api } from '../../api/client'
import type { CommandCatalog, ConsoleView, PlanCurrent, SessionInfo, SlashCommandInfo, ToolGroup } from '../../api/types'

export type ParsedSlashCommand = {
  name: string
  arg: string
}

export type WebSlashResult =
  | { type: 'notice'; message: string }
  | { type: 'error'; message: string }
  | { type: 'send'; message: string }
  | { type: 'clear' }

type SlashExecutionContext = {
  session: SessionInfo
  commands?: SlashCommandInfo[]
  loadCommands?: () => Promise<CommandCatalog>
  loadSessions?: () => Promise<{ sessions: SessionInfo[] }>
  loadToolGroups?: () => Promise<{ groups: ToolGroup[] }>
  loadPlanCurrent?: (sessionId: string) => Promise<PlanCurrent>
  setPlanMode?: (sessionId: string, mode: 'agent' | 'plan') => Promise<PlanCurrent>
  clearPlan?: (sessionId: string) => Promise<PlanCurrent>
  onSessionMode?: (sessionId: string, mode: 'agent' | 'plan') => void
  openView?: (view: ConsoleView) => void
  createSession?: (input: { provider: string; model: string }) => Promise<SessionInfo>
  resumeSession?: (sessionId: string) => Promise<SessionInfo>
  updateSession?: (
    sessionId: string,
    updates: Partial<Pick<SessionInfo, 'provider' | 'model' | 'base_url' | 'system_prompt'>>
  ) => Promise<SessionInfo>
}

export function isSlashCommandInput(value: string): boolean {
  return /^\/[a-zA-Z0-9:_-]+(?:\s|$)/.test(value.trim())
}

export function parseSlashCommand(value: string): ParsedSlashCommand | null {
  const trimmed = value.trim()
  if (!isSlashCommandInput(trimmed)) return null
  const match = trimmed.match(/^\/([a-zA-Z0-9:_-]+)(?:\s+([\s\S]*))?$/)
  if (!match?.[1]) return null
  return {
    name: match[1],
    arg: match[2]?.trim() ?? '',
  }
}

export async function executeWebSlashCommand(
  command: string,
  context: SlashExecutionContext,
): Promise<WebSlashResult> {
  const parsed = parseSlashCommand(command)
  if (!parsed) return { type: 'send', message: command }

  switch (parsed.name) {
    case 'help':
      return { type: 'notice', message: formatHelp(await loadCommandCatalog(context)) }
    case 'new':
      return executeNewCommand(context)
    case 'clear':
      return executeClearCommand(context)
    case 'session':
      return { type: 'notice', message: formatSession(context.session) }
    case 'sessions':
      context.openView?.('sessions')
      return { type: 'notice', message: formatSessions((await loadSessions(context)).sessions) }
    case 'resume':
      return executeResumeCommand(parsed.arg, context)
    case 'system':
      return executeSystemCommand(parsed.arg, context)
    case 'tools':
      context.openView?.('tools')
      return { type: 'notice', message: formatTools((await loadToolGroups(context)).groups) }
    case 'model':
      return executeModelCommand(parsed.arg, context)
    case 'plan':
      return executePlanCommand(parsed.arg, context)
    default: {
      const catalog = await loadCommandCatalog(context)
      const known = catalog.some((item) => item.name === '/' + parsed.name)
      return {
        type: 'error',
        message: known
          ? '/' + parsed.name + ' is not implemented in the web console yet.'
          : 'Unknown slash command /' + parsed.name + '. Type /help for available commands.',
      }
    }
  }
}

async function loadCommandCatalog(context: SlashExecutionContext): Promise<SlashCommandInfo[]> {
  if (context.commands && context.commands.length > 0) return context.commands
  return (await (context.loadCommands ?? api.commands)()).commands
}

async function loadSessions(context: SlashExecutionContext): Promise<{ sessions: SessionInfo[] }> {
  return (context.loadSessions ?? api.sessions)()
}

async function loadToolGroups(context: SlashExecutionContext): Promise<{ groups: ToolGroup[] }> {
  return (context.loadToolGroups ?? api.toolGroups)()
}

async function loadPlanCurrent(context: SlashExecutionContext): Promise<PlanCurrent> {
  return (context.loadPlanCurrent ?? api.planCurrent)(context.session.session_id)
}

async function setPlanMode(context: SlashExecutionContext, mode: 'agent' | 'plan'): Promise<PlanCurrent> {
  const result = await (context.setPlanMode ?? api.setPlanMode)(context.session.session_id, mode)
  context.onSessionMode?.(context.session.session_id, result.mode)
  return result
}

async function clearPlan(context: SlashExecutionContext): Promise<PlanCurrent> {
  const result = await (context.clearPlan ?? api.clearPlan)(context.session.session_id)
  context.onSessionMode?.(context.session.session_id, result.mode)
  return result
}

async function executeClearCommand(context: SlashExecutionContext): Promise<WebSlashResult> {
  await clearPlan(context).catch(() => {
    context.onSessionMode?.(context.session.session_id, 'agent')
  })
  return { type: 'clear' }
}

async function executeNewCommand(context: SlashExecutionContext): Promise<WebSlashResult> {
  if (!context.createSession) {
    return { type: 'error', message: '/new is not available in this web context.' }
  }
  const created = await context.createSession({
    provider: context.session.provider,
    model: context.session.model,
  })
  context.openView?.('chat')
  return {
    type: 'notice',
    message: 'Created session `' + created.session_id + '` with `' + created.provider + '/' + created.model + '`.',
  }
}

async function executeResumeCommand(arg: string, context: SlashExecutionContext): Promise<WebSlashResult> {
  const target = arg.trim()
  if (!target) {
    context.openView?.('sessions')
    return {
      type: 'notice',
      message: 'Opened Sessions. Use `/resume <id-or-prefix>` to resume directly from chat.',
    }
  }
  if (!context.resumeSession) {
    return { type: 'error', message: '/resume is not available in this web context.' }
  }
  const resumed = await context.resumeSession(target)
  context.openView?.('chat')
  return {
    type: 'notice',
    message: 'Resumed session `' + resumed.session_id + '`.',
  }
}

async function executeSystemCommand(arg: string, context: SlashExecutionContext): Promise<WebSlashResult> {
  const nextPrompt = arg.trim()
  if (!nextPrompt) {
    return {
      type: 'notice',
      message: context.session.system_prompt?.trim()
        ? '# System prompt\n\n' + context.session.system_prompt
        : 'No system prompt is configured for this session.',
    }
  }
  if (!context.updateSession) {
    return { type: 'error', message: '/system is not available in this web context.' }
  }
  const updated = await context.updateSession(context.session.session_id, { system_prompt: nextPrompt })
  return {
    type: 'notice',
    message: 'Updated system prompt for session `' + updated.session_id + '`.',
  }
}

async function executeModelCommand(arg: string, context: SlashExecutionContext): Promise<WebSlashResult> {
  const target = arg.trim()
  if (!target) {
    context.openView?.('models')
    return { type: 'notice', message: formatModel(context.session) }
  }
  if (!context.updateSession) {
    return { type: 'error', message: '/model is not available in this web context.' }
  }
  const updated = await context.updateSession(context.session.session_id, { model: target })
  return {
    type: 'notice',
    message: 'Model set to `' + cell(updated.model) + '` for session `' + updated.session_id + '`.',
  }
}

async function executePlanCommand(arg: string, context: SlashExecutionContext): Promise<WebSlashResult> {
  const normalizedArg = arg.trim()
  const current = await loadPlanCurrent(context)

  if (normalizedArg.toLowerCase() === 'open') {
    return { type: 'notice', message: formatPlanCurrent(current) }
  }

  if (normalizedArg) {
    if (current.mode !== 'plan') {
      await setPlanMode(context, 'plan')
    } else {
      context.onSessionMode?.(context.session.session_id, 'plan')
    }
    return { type: 'send', message: normalizedArg }
  }

  if (current.mode !== 'plan') {
    const updated = await setPlanMode(context, 'plan')
    return {
      type: 'notice',
      message: 'Enabled plan mode.\n\n' + formatPlanCurrent(updated),
    }
  }

  context.onSessionMode?.(context.session.session_id, 'plan')
  return { type: 'notice', message: formatPlanCurrent(current) }
}

function formatHelp(commands: SlashCommandInfo[]): string {
  const rows = commands.map((command) => (
    '| `' + command.name + '` | ' + cell(command.category ?? 'local') + ' | ' + cell(command.description) + ' |'
  ))
  return [
    '# Slash commands',
    '',
    '| Command | Category | Description |',
    '| --- | --- | --- |',
    ...rows,
  ].join('\n')
}

function formatSession(session: SessionInfo): string {
  return [
    '# Current session',
    '',
    '| Field | Value |',
    '| --- | --- |',
    '| Session | `' + cell(session.session_id) + '` |',
    '| Provider | ' + cell(session.provider) + ' |',
    '| Model | `' + cell(session.model) + '` |',
    '| Mode | ' + cell(session.mode ?? 'agent') + ' |',
    '| Messages | ' + String(session.message_count) + ' |',
  ].join('\n')
}

function formatSessions(sessions: SessionInfo[]): string {
  const rows = sessions.slice(0, 10).map((session) => (
    '| `' + cell(session.session_id.slice(0, 8)) + '` | ' +
    cell(session.summary || '(no summary)') + ' | `' +
    cell(session.provider + '/' + session.model) + '` |'
  ))
  return [
    '# Recent sessions',
    '',
    '| Session | Summary | Runtime |',
    '| --- | --- | --- |',
    ...(rows.length > 0 ? rows : ['| - | No sessions found. | - |']),
  ].join('\n')
}

function formatTools(groups: ToolGroup[]): string {
  const rows = groups.map((group) => (
    '| ' + cell(group.name) + ' | ' + String(group.tools.length) + ' | ' +
    cell(group.tools.slice(0, 8).map((tool) => tool.name).join(', ') || '-') + ' |'
  ))
  return [
    '# Tool catalog',
    '',
    '| Group | Tools | Examples |',
    '| --- | ---: | --- |',
    ...(rows.length > 0 ? rows : ['| - | 0 | No tools available. |']),
  ].join('\n')
}

function formatModel(session: SessionInfo): string {
  return [
    '# Current model',
    '',
    '- Provider: `' + session.provider + '`',
    '- Model: `' + session.model + '`',
    session.base_url ? '- Base URL: `' + session.base_url + '`' : '',
  ].filter(Boolean).join('\n')
}

function formatPlanCurrent(plan: PlanCurrent): string {
  if (plan.has_plan && plan.plan_content?.trim()) {
    return [
      '# Current plan',
      '',
      plan.plan_path ? '_Artifact: `' + cell(plan.plan_path) + '`_' : '',
      '',
      plan.plan_content,
    ].filter(Boolean).join('\n')
  }
  return [
    '# Current plan',
    '',
    '- Mode: `' + plan.mode + '`',
    plan.plan_path ? '- Artifact: `' + cell(plan.plan_path) + '`' : '',
    '- No plan written yet.',
  ].filter(Boolean).join('\n')
}

function cell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\n/g, ' ')
}
