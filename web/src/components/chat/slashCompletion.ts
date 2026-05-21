import type { SlashCommandInfo } from '../../api/types'

export type SlashTrigger = {
  slashPos: number
  filter: string
}

export function findSlashTrigger(value: string, cursorPosition: number): SlashTrigger | null {
  const cursor = clampCursor(value, cursorPosition)
  const beforeCursor = value.slice(0, cursor)
  let slashPos = -1

  for (let index = beforeCursor.length - 1; index >= 0; index -= 1) {
    const char = beforeCursor[index]
    if (char === '/') {
      if (index === 0 || /\s/.test(beforeCursor[index - 1] ?? '')) {
        slashPos = index
      }
      break
    }
    if (/\s/.test(char ?? '')) break
  }

  if (slashPos < 0) return null
  const filter = beforeCursor.slice(slashPos + 1)
  if (/\s|\//.test(filter)) return null
  return { slashPos, filter }
}

export function filterSlashCommands(
  commands: ReadonlyArray<SlashCommandInfo>,
  filter: string,
  limit = 8,
): SlashCommandInfo[] {
  const normalized = filter.toLowerCase()
  const unique = new Map<string, SlashCommandInfo>()
  for (const command of commands) {
    if (!command.name?.startsWith('/')) continue
    if (!unique.has(command.name)) unique.set(command.name, command)
  }

  return [...unique.values()]
    .filter((command) => {
      const name = command.name.slice(1).toLowerCase()
      return !normalized || name.startsWith(normalized) || name.includes(normalized)
    })
    .sort((left, right) => {
      const leftName = left.name.slice(1).toLowerCase()
      const rightName = right.name.slice(1).toLowerCase()
      const leftPrefix = leftName.startsWith(normalized) ? 0 : 1
      const rightPrefix = rightName.startsWith(normalized) ? 0 : 1
      return leftPrefix - rightPrefix || leftName.localeCompare(rightName)
    })
    .slice(0, limit)
}

export function replaceSlashToken(
  value: string,
  cursorPosition: number,
  commandName: string,
): { value: string; cursorPosition: number } {
  const cursor = clampCursor(value, cursorPosition)
  const normalizedCommand = commandName.startsWith('/') ? commandName : '/' + commandName
  const trigger = findSlashTrigger(value, cursor)
  const token = normalizedCommand + ' '

  if (!trigger) {
    const prefix = value && !/\s$/.test(value) ? value + ' ' : value
    const nextValue = prefix + token
    return { value: nextValue, cursorPosition: nextValue.length }
  }

  const before = value.slice(0, trigger.slashPos)
  const after = value.slice(cursor)
  const nextValue = before + token + after
  return {
    value: nextValue,
    cursorPosition: before.length + token.length,
  }
}

function clampCursor(value: string, cursorPosition: number): number {
  if (!Number.isFinite(cursorPosition)) return value.length
  return Math.max(0, Math.min(value.length, cursorPosition))
}
