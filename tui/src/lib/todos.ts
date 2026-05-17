import stringWidth from 'string-width'

import type { JsonObject } from '../gatewayTypes.js'

export type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled'

export interface TodoItem {
  id: string
  content: string
  status: TodoStatus
  activeForm?: string
}

export interface TodoPreviewOptions {
  limit?: number
  ascii?: boolean
  width?: number
}

const VALID_STATUSES: ReadonlySet<string> = new Set([
  'pending',
  'in_progress',
  'completed',
  'cancelled'
])

export function normalizeTodos(value: unknown): TodoItem[] {
  if (!Array.isArray(value)) {
    return []
  }
  const todos: TodoItem[] = []
  for (const item of value) {
    if (!isObject(item)) {
      continue
    }
    const id = stringValue(item.id)
    const content = stringValue(item.content)
    const status = stringValue(item.status)
    if (!id || !content || !status || !VALID_STATUSES.has(status)) {
      continue
    }
    const activeForm = stringValue(item.activeForm)
    todos.push({
      id,
      content,
      status: status as TodoStatus,
      ...(activeForm ? { activeForm } : {})
    })
  }
  return todos
}

export function todosFromArgs(args: JsonObject | undefined): TodoItem[] {
  if (!args) {
    return []
  }
  return normalizeTodos(args.todos)
}

export function shouldClearTodos(todos: TodoItem[]): boolean {
  return todos.length > 0 && todos.every((todo) => isTerminalStatus(todo.status))
}

export function activeTodoTitle(todos: TodoItem[]): string | null {
  const active =
    todos.find((todo) => todo.status === 'in_progress') ??
    todos.find((todo) => todo.status === 'pending')
  return active?.activeForm || active?.content || null
}

export function summarizeTodos(todos: TodoItem[]): string {
  if (todos.length === 0) {
    return ''
  }
  const completed = todos.filter((todo) => todo.status === 'completed').length
  const total = todos.length
  return `${total} ${total === 1 ? 'todo' : 'todos'} · ${completed}/${total} done`
}

export function formatTodoPreviewLines(
  todos: TodoItem[],
  optionsOrLimit: TodoPreviewOptions | number = {}
): string[] {
  const options =
    typeof optionsOrLimit === 'number' ? { limit: optionsOrLimit } : optionsOrLimit
  const limit = Math.max(0, Math.floor(options.limit ?? 8))
  const ascii = options.ascii ?? false
  const width = options.width ?? 0
  const lines = todos.slice(0, limit).map((todo, idx) => {
    const prefix = idx === 0 ? (ascii ? '- ' : '└ ') : '  '
    const icon = statusIcon(todo.status, ascii)
    const available = width > 0
      ? Math.max(8, width - stringWidth(prefix) - stringWidth(icon) - 1)
      : 0
    const content = available > 0
      ? truncateToWidth(todo.content, available, ascii)
      : todo.content
    return `${prefix}${icon} ${content}`
  })
  if (todos.length > limit) {
    const hidden = todos.slice(limit)
    lines.push(`  ${ascii ? '...' : '…'} +${summarizeHiddenTodos(hidden)}`)
  }
  return lines
}

function statusIcon(status: TodoStatus, ascii: boolean): string {
  if (ascii) {
    switch (status) {
      case 'completed':
        return '[x]'
      case 'in_progress':
        return '[*]'
      case 'cancelled':
        return '[-]'
      case 'pending':
        return '[ ]'
    }
  }
  switch (status) {
    case 'completed':
      return '☑'
    case 'in_progress':
      return '■'
    case 'cancelled':
      return '⊘'
    case 'pending':
      return '□'
  }
}

function summarizeHiddenTodos(todos: TodoItem[]): string {
  const parts: string[] = []
  const inProgress = todos.filter((todo) => todo.status === 'in_progress').length
  const pending = todos.filter((todo) => todo.status === 'pending').length
  const completed = todos.filter((todo) => todo.status === 'completed').length
  const cancelled = todos.filter((todo) => todo.status === 'cancelled').length
  if (inProgress > 0) {
    parts.push(`${inProgress} in progress`)
  }
  if (pending > 0) {
    parts.push(`${pending} pending`)
  }
  if (completed > 0) {
    parts.push(`${completed} completed`)
  }
  if (cancelled > 0) {
    parts.push(`${cancelled} cancelled`)
  }
  return parts.length > 0 ? parts.join(', ') : `${todos.length} more`
}

function truncateToWidth(value: string, width: number, ascii: boolean): string {
  if (stringWidth(value) <= width) {
    return value
  }
  const suffix = ascii ? '...' : '…'
  const suffixWidth = stringWidth(suffix)
  const target = Math.max(1, width - suffixWidth)
  let result = ''
  let used = 0
  for (const char of Array.from(value)) {
    const charWidth = stringWidth(char)
    if (used + charWidth > target) {
      break
    }
    result += char
    used += charWidth
  }
  return `${result}${suffix}`
}

function isTerminalStatus(status: TodoStatus): boolean {
  return status === 'completed' || status === 'cancelled'
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function stringValue(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}
