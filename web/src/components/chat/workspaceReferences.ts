import type { ChatAttachment } from '../../chat-rendering'
import type { WorkspaceEntry } from '../../api/types'

export type WorkspaceReferenceTrigger = {
  atPos: number
  filter: string
}

export function findWorkspaceReferenceTrigger(value: string, cursorPosition: number): WorkspaceReferenceTrigger | null {
  const cursor = clampCursor(value, cursorPosition)
  const beforeCursor = value.slice(0, cursor)
  let atPos = -1

  for (let index = beforeCursor.length - 1; index >= 0; index -= 1) {
    const char = beforeCursor[index]
    if (char === '@') {
      if (index === 0 || /\s/.test(beforeCursor[index - 1] ?? '')) {
        atPos = index
      }
      break
    }
    if (/\s/.test(char ?? '')) break
  }

  if (atPos < 0) return null
  return { atPos, filter: beforeCursor.slice(atPos + 1) }
}

export function replaceWorkspaceReferenceToken(
  value: string,
  cursorPosition: number,
  path: string,
): { value: string; cursorPosition: number } {
  const cursor = clampCursor(value, cursorPosition)
  const trigger = findWorkspaceReferenceTrigger(value, cursor)
  const token = '@' + path + ' '

  if (!trigger) {
    const prefix = value && !/\s$/.test(value) ? value + ' ' : value
    const nextValue = prefix + token
    return { value: nextValue, cursorPosition: nextValue.length }
  }

  const before = value.slice(0, trigger.atPos)
  const after = value.slice(cursor).replace(/^[ \t]+/, '')
  const nextValue = before + token + after
  return {
    value: nextValue,
    cursorPosition: before.length + token.length,
  }
}

export function replaceWorkspaceReferenceBrowseToken(
  value: string,
  cursorPosition: number,
  path: string,
): { value: string; cursorPosition: number } {
  const cursor = clampCursor(value, cursorPosition)
  const trigger = findWorkspaceReferenceTrigger(value, cursor)
  const normalizedPath = path.trim().replace(/^\/+/, '').replace(/\/+$/g, '')
  const token = '@' + (normalizedPath ? normalizedPath + '/' : '')

  if (!trigger) {
    const prefix = value && !/\s$/.test(value) ? value + ' ' : value
    const nextValue = prefix + token
    return { value: nextValue, cursorPosition: nextValue.length }
  }

  const before = value.slice(0, trigger.atPos)
  const afterText = value.slice(cursor).replace(/^[ \t]+/, '')
  const separator = afterText ? ' ' : ''
  const nextValue = before + token + separator + afterText
  return {
    value: nextValue,
    cursorPosition: before.length + token.length,
  }
}

export function workspaceEntryToAttachment(entry: WorkspaceEntry): ChatAttachment {
  return {
    type: entry.kind === 'directory' ? 'file' : fileAttachmentType(entry.path),
    name: entry.kind === 'directory' ? entry.name + '/' : entry.name,
    path: entry.path,
    ...(entry.kind === 'directory' ? { isDirectory: true } : {}),
    note: 'workspace reference',
  }
}

export function mergeWorkspaceAttachment(
  attachments: ChatAttachment[],
  entry: WorkspaceEntry,
): ChatAttachment[] {
  if (attachments.some((attachment) => attachment.path === entry.path)) return attachments
  return [...attachments, workspaceEntryToAttachment(entry)]
}

export function workspaceReferenceTokenExists(value: string, path: string): boolean {
  const normalizedPath = path.trim().replace(/^@+/, "").replace(/\/+$/g, "")
  if (!normalizedPath) return false
  const escaped = escapeRegExp(normalizedPath)
  return new RegExp("(^|\\s)@" + escaped + "/?(?=\\s|$)").test(value)
}

export function syncWorkspaceReferenceAttachmentsForValue(
  attachments: ChatAttachment[],
  value: string,
): ChatAttachment[] {
  let changed = false
  const next = attachments.filter((attachment) => {
    if (attachment.note !== "workspace reference") return true
    if (!attachment.path) return true
    const keep = workspaceReferenceTokenExists(value, attachment.path)
    if (!keep) changed = true
    return keep
  })
  return changed ? next : attachments
}

function fileAttachmentType(path: string): ChatAttachment['type'] {
  return /\.(md|markdown|txt|json|jsonl|yaml|yml|toml|csv|ts|tsx|js|jsx|py|rs|go|java|c|cc|cpp|h|hpp|css|html|xml|sh|bash|zsh)$/i.test(path)
    ? 'text'
    : 'file'
}

function escapeRegExp(value: string): string {
  const special = new Set(["\\", ".", "*", "+", "?", "^", "$", "{", "}", "(", ")", "|", "[", "]"])
  let escaped = ""
  for (const char of value) escaped += special.has(char) ? "\\" + char : char
  return escaped
}

function clampCursor(value: string, cursorPosition: number): number {
  if (!Number.isFinite(cursorPosition)) return value.length
  return Math.max(0, Math.min(value.length, cursorPosition))
}
