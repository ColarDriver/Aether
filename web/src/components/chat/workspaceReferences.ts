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

function fileAttachmentType(path: string): ChatAttachment['type'] {
  return /\.(md|markdown|txt|json|jsonl|yaml|yml|toml|csv|ts|tsx|js|jsx|py|rs|go|java|c|cc|cpp|h|hpp|css|html|xml|sh|bash|zsh)$/i.test(path)
    ? 'text'
    : 'file'
}

function clampCursor(value: string, cursorPosition: number): number {
  if (!Number.isFinite(cursorPosition)) return value.length
  return Math.max(0, Math.min(value.length, cursorPosition))
}
