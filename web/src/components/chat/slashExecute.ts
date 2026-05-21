import { api } from '../../api/client'
import type { CommandCatalog, PlanCurrent, SessionInfo, SlashCommandInfo, ToolGroup } from '../../api/types'

export type ParsedSlashCommand = {
  name: string
  arg: string
}

export type WebSlashResult =
  | { type: 'notice'; message: string }
  | { type: 'error'; message: string }
  | { type: 'send'; message: string }

type SlashExecutionContext = {
  session: SessionInfo
  commands?: SlashCommandInfo[]
  loadCommands?: () => Promise<CommandCatalog>
  loadSessions?: () => Promise<{ sessions: SessionInfo[] }>
  loadToolGroups?: () => Promise<{ groups: ToolGroup[] }>
  loadPlanCurrent?: (sessionId: string) => Promise<PlanCurrent>
  setPlanMode?: (sessionId: string, mode: 'agent' | 'plan') => Promise<PlanCurrent>
  onSessionMode?: (sessionId: string, mode: 'agent' | 'plan') => void
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
    case 'session':
      return { type: 'notice', message: formatSession(context.session) }
    case 'sessions':
      return { type: 'notice', message: formatSessions((await loadSessions(context)).sessions) }
    case 'tools':
      return { type: 'notice', message: formatTools((await loadToolGroups(context)).groups) }
    case 'model':
      return { type: 'notice', message: formatModel(context.session) }
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
