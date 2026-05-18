/**
 * Port of `aether/cli/tool_groups.py`'s category resolver, hint formatter and
 * verb tables — kept dependency-free so the store + tests can import without
 * pulling in Ink. Comments deliberately mirror the Python originals so the
 * two stay readable as a pair.
 */

import type { JsonObject } from '../gatewayTypes.js'
import { summarizeTodos, todosFromArgs } from './todos.js'

export type ToolCategory =
  | 'search'
  | 'read'
  | 'list'
  | 'write'
  | 'edit'
  | 'bash'
  | 'web'
  | 'subagent'
  | 'mcp'
  | 'other'

// Categories that get COALESCED under a single `● Explored` umbrella.
// Bash / Web / Subagent / MCP / Other stay per-call because their output IS
// the user-relevant signal.
export const EXPLORE_CATEGORIES: ReadonlySet<ToolCategory> = new Set([
  'read',
  'list',
  'search',
  'write',
  'edit'
])

// Order drives how categories appear in the rolling headline
// ("Searching for X, reading Y, listing Z").
export const CATEGORY_ORDER: readonly ToolCategory[] = [
  'search',
  'read',
  'list',
  'write',
  'edit',
  'bash',
  'web',
  'subagent',
  'mcp',
  'other'
]

const NAME_TO_CATEGORY: Record<string, ToolCategory> = {
  // read
  read_file: 'read',
  read: 'read',
  Read: 'read',
  view_file: 'read',
  ViewFile: 'read',
  cat: 'read',
  // search
  search: 'search',
  grep: 'search',
  Grep: 'search',
  search_code: 'search',
  find: 'search',
  Glob: 'search',
  glob: 'search',
  rg: 'search',
  // list
  list_directory: 'list',
  ListDirectory: 'list',
  ls: 'list',
  list: 'list',
  tree: 'list',
  // write
  write_file: 'write',
  WriteFile: 'write',
  Write: 'write',
  create_file: 'write',
  save_file: 'write',
  // edit
  todo_write: 'edit',
  edit_file: 'edit',
  file_edit: 'edit',
  Edit: 'edit',
  EditFile: 'edit',
  patch: 'edit',
  apply_patch: 'edit',
  delete_file: 'edit',
  DeleteFile: 'edit',
  str_replace: 'edit',
  StrReplace: 'edit',
  notebook_edit: 'edit',
  // bash
  run_bash: 'bash',
  Bash: 'bash',
  bash: 'bash',
  shell: 'bash',
  execute: 'bash',
  Exec: 'bash',
  execute_command: 'bash',
  run_command: 'bash',
  // web
  WebFetch: 'web',
  fetch_url: 'web',
  WebSearch: 'web',
  // subagent
  Task: 'subagent',
  spawn_agent: 'subagent'
}

const PREFIX_TO_CATEGORY: Array<readonly [string, ToolCategory]> = [
  ['read_', 'read'],
  ['get_', 'read'],
  ['view_', 'read'],
  ['write_', 'write'],
  ['create_', 'write'],
  ['save_', 'write'],
  ['edit_', 'edit'],
  ['update_', 'edit'],
  ['patch_', 'edit'],
  ['delete_', 'edit'],
  ['remove_', 'edit'],
  ['list_', 'list'],
  ['show_', 'list'],
  ['run_', 'bash'],
  ['exec_', 'bash'],
  ['execute_', 'bash'],
  ['search_', 'search'],
  ['find_', 'search'],
  ['grep_', 'search'],
  ['fetch_', 'web'],
  ['download_', 'web']
]

function canonicalToolName(name: string): string {
  let normalized = (name ?? '').trim()
  if (!normalized) {
    return ''
  }
  if (normalized.includes('__')) {
    const parts = normalized.split('__')
    normalized = parts[parts.length - 1] ?? normalized
  }
  if (normalized.includes('.')) {
    const parts = normalized.split('.')
    normalized = parts[parts.length - 1] ?? normalized
  }
  if (normalized.includes(':')) {
    const parts = normalized.split(':')
    normalized = parts[parts.length - 1] ?? normalized
  }
  return normalized
}

export function categoryFor(name: string): ToolCategory {
  const canonical = canonicalToolName(name)
  if (canonical in NAME_TO_CATEGORY) {
    return NAME_TO_CATEGORY[canonical] ?? 'other'
  }
  // Names that originate from an MCP server are bucketed as MCP only after
  // canonical lookup misses — `mcp__filesystem__read_file` still reads as
  // READ thanks to the canonicalisation above.
  if (name && name.toLowerCase().includes('mcp__')) {
    return 'mcp'
  }
  const lname = canonical.toLowerCase()
  for (const [prefix, category] of PREFIX_TO_CATEGORY) {
    if (lname.startsWith(prefix)) {
      return category
    }
  }
  return 'other'
}

// (first-position present, follow-on present)
const VERBS_PRESENT: Record<ToolCategory, [string, string]> = {
  search: ['Searching for', 'searching for'],
  read: ['Reading', 'reading'],
  list: ['Listing', 'listing'],
  write: ['Writing', 'writing'],
  edit: ['Editing', 'editing'],
  bash: ['Running', 'running'],
  web: ['Fetching', 'fetching'],
  subagent: ['Spawning', 'spawning'],
  mcp: ['Querying', 'querying'],
  other: ['Calling', 'calling']
}

const VERBS_PAST: Record<ToolCategory, [string, string]> = {
  search: ['Searched for', 'searched for'],
  read: ['Read', 'read'],
  list: ['Listed', 'listed'],
  write: ['Wrote', 'wrote'],
  edit: ['Edited', 'edited'],
  bash: ['Ran', 'ran'],
  web: ['Fetched', 'fetched'],
  subagent: ['Spawned', 'spawned'],
  mcp: ['Queried', 'queried'],
  other: ['Called', 'called']
}

const NOUNS: Record<ToolCategory, [string, string]> = {
  search: ['pattern', 'patterns'],
  read: ['file', 'files'],
  list: ['directory', 'directories'],
  write: ['file', 'files'],
  edit: ['file', 'files'],
  bash: ['command', 'commands'],
  web: ['URL', 'URLs'],
  subagent: ['subagent', 'subagents'],
  mcp: ['call', 'calls'],
  other: ['call', 'calls']
}

export function verbForCategory(category: ToolCategory, past: boolean): [string, string] {
  return past ? VERBS_PAST[category] : VERBS_PRESENT[category]
}

export function nounForCategory(category: ToolCategory): [string, string] {
  return NOUNS[category]
}

/**
 * Friendly verb for a single tool call (Python's `_verb_for_tool`). Bash uses
 * "Ran"/"Running" etc. depending on tense; for parity with the explore tree
 * we always return the past-tense singular here.
 */
export function verbForTool(name: string): string {
  const category = categoryFor(name)
  return VERBS_PAST[category][0]
}

// Friendlier display names for tools whose snake_case name reads poorly in
// the transcript header. Mirrors Claude Code's `userFacingName` overrides
// (open-claude-code/src/tools/{Enter,Exit}PlanModeTool, AskUserQuestionTool).
const FRIENDLY_TOOL_NAMES: Record<string, string> = {
  enter_plan_mode: 'Plan mode',
  exit_plan_mode: 'Plan ready',
  ask_user_question: 'Ask user',
  todo_write: 'Todos'
}

export function displayToolName(name: string): string {
  const canonical = canonicalToolName(name)
  return FRIENDLY_TOOL_NAMES[canonical] ?? name
}

function truncate(value: string, limit: number = 80): string {
  const cleaned = (value || '').replace(/[\r\n]+/g, ' ').trim()
  if (cleaned.length <= limit) {
    return cleaned
  }
  return `${cleaned.slice(0, limit - 1)}…`
}

/**
 * Indented hint shown beneath an active group line. Mirrors `hint_for_call`
 * in tool_groups.py — same per-category preferences so users see the same
 * shape as the Python TUI.
 */
export function hintForCall(name: string, args: JsonObject | undefined): string {
  if (!args) {
    return ''
  }
  const canonical = canonicalToolName(name)
  if (canonical === 'todo_write') {
    return summarizeTodos(todosFromArgs(args))
  }
  if (canonical === 'ask_user_question') {
    return hintForAskUserQuestion(args)
  }
  if (canonical === 'exit_plan_mode') {
    return hintForExitPlanMode(args)
  }
  if (canonical === 'enter_plan_mode') {
    return 'entering'
  }
  const category = categoryFor(name)
  const path =
    strOrNull(args['path']) ??
    strOrNull(args['file_path']) ??
    strOrNull(args['filename']) ??
    strOrNull(args['filepath']) ??
    strOrNull(args['relative_workspace_path']) ??
    strOrNull(args['target_file']) ??
    strOrNull(args['directory'])

  if (category === 'bash') {
    const cmd = strOrNull(args['command']) ?? strOrNull(args['cmd']) ?? strOrNull(args['script'])
    if (cmd) {
      return `$ ${truncate(cmd)}`
    }
  }

  if (category === 'search') {
    const pattern =
      strOrNull(args['pattern']) ??
      strOrNull(args['query']) ??
      strOrNull(args['search_term'])
    const searchPath = path ?? strOrNull(args['file'])
    if (pattern && searchPath) {
      return `"${truncate(pattern, 40)}" in ${truncate(searchPath, 40)}`
    }
    if (pattern) {
      return `"${truncate(pattern)}"`
    }
    if (searchPath) {
      return truncate(searchPath)
    }
  }

  if (category === 'web') {
    const url = strOrNull(args['url'])
    if (url) {
      return truncate(url)
    }
  }

  if (category === 'subagent') {
    const desc =
      strOrNull(args['description']) ??
      strOrNull(args['prompt']) ??
      strOrNull(args['name'])
    if (desc) {
      return truncate(desc)
    }
  }

  if (category === 'read' && path) {
    const range = formatReadRange(args)
    if (range) {
      return truncate(`${path} · ${range}`)
    }
    return truncate(path)
  }

  if (path) {
    return truncate(path)
  }

  for (const key of ['command', 'cmd', 'script', 'pattern', 'query']) {
    const value = strOrNull(args[key])
    if (value) {
      return truncate(value)
    }
  }

  for (const value of Object.values(args)) {
    if (typeof value === 'string' && value.length > 0 && value.length <= 80) {
      return truncate(value)
    }
  }

  return ''
}

function hintForExitPlanMode(args: JsonObject): string {
  const plan = strOrNull(args['plan'])
  if (!plan) {
    return 'plan ready for approval'
  }
  const firstLine = plan.split('\n').find((line) => line.trim().length > 0) ?? ''
  const cleaned = firstLine.replace(/^#+\s*/, '').trim()
  return truncate(cleaned || 'plan ready for approval', 80)
}

function hintForAskUserQuestion(args: JsonObject): string {
  const raw = args['questions']
  if (!Array.isArray(raw) || raw.length === 0) {
    return '(asking the user)'
  }
  const firstPrompt = pickQuestionText(raw[0])
  if (raw.length === 1) {
    return firstPrompt ? truncate(firstPrompt, 80) : '1 question'
  }
  if (firstPrompt) {
    return `${raw.length} questions · ${truncate(firstPrompt, 60)}`
  }
  return `${raw.length} questions`
}

function pickQuestionText(question: unknown): string {
  if (!question || typeof question !== 'object') {
    return ''
  }
  const record = question as Record<string, unknown>
  return (
    strOrNull(record['prompt']) ??
    strOrNull(record['question']) ??
    strOrNull(record['text']) ??
    ''
  )
}

function strOrNull(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function intOrNull(value: unknown): number | null {
  if (typeof value === 'number' && Number.isInteger(value)) {
    return value
  }
  if (typeof value !== 'string') {
    return null
  }
  const trimmed = value.trim()
  if (!/^-?\d+$/.test(trimmed)) {
    return null
  }
  return Number.parseInt(trimmed, 10)
}

function formatReadRange(args: JsonObject): string | null {
  const offset = intOrNull(args['offset'])
  const limit = intOrNull(args['limit'])
  if (offset === null && limit === null) {
    return null
  }
  if (offset !== null && offset < 0) {
    if (limit !== null && limit > 0) {
      return `${limit} lines from ${offset}`
    }
    return `from ${offset}`
  }
  const start = offset ?? 1
  if (limit !== null && limit > 0) {
    return `lines ${start}-${start + limit - 1}`
  }
  return `from line ${start}`
}
