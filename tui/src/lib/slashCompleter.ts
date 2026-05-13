import type { SlashCommandInfo } from '../gatewayTypes.js'

export interface SlashCompletionMatch {
  command: SlashCommandInfo
  rank: number
}

/**
 * Detect whether the current draft starts a slash command and return the
 * leading token (the part the user has typed so far, e.g. "/he").
 *
 * Returns null when the draft is multi-line or does not begin with `/` —
 * those cases should fall through to other Tab handling (focus hand-off,
 * indentation, etc.).
 */
export function leadingSlashToken(draft: string): string | null {
  if (!draft.startsWith('/')) {
    return null
  }
  if (draft.includes('\n')) {
    return null
  }
  const space = draft.indexOf(' ')
  return space === -1 ? draft : draft.slice(0, space)
}

/**
 * Filter the slash catalog by prefix-match against the leading token. The
 * empty token (`/`) returns every command so Tab on a bare `/` shows the
 * full list to flip through.
 */
export function rankSlashMatches(
  token: string,
  catalog: readonly SlashCommandInfo[]
): SlashCommandInfo[] {
  if (!token.startsWith('/')) {
    return []
  }
  const lower = token.toLowerCase()
  if (lower === '/') {
    return [...catalog]
  }
  const matches: SlashCompletionMatch[] = []
  for (const command of catalog) {
    if (command.name.toLowerCase().startsWith(lower)) {
      matches.push({ command, rank: 0 })
      continue
    }
    if (command.name.toLowerCase().includes(lower.slice(1))) {
      matches.push({ command, rank: 1 })
    }
  }
  matches.sort((a, b) => {
    if (a.rank !== b.rank) {
      return a.rank - b.rank
    }
    return a.command.name.localeCompare(b.command.name)
  })
  return matches.map((match) => match.command)
}

export interface SlashCompleterState {
  /** Active when the draft starts with `/` and we have at least one match. */
  active: boolean
  matches: SlashCommandInfo[]
  /** Index of the current candidate (incremented by Tab, decremented by Shift+Tab). */
  index: number
}

export function buildCompleterState(
  draft: string,
  catalog: readonly SlashCommandInfo[]
): SlashCompleterState {
  const token = leadingSlashToken(draft)
  if (token === null) {
    return { active: false, matches: [], index: 0 }
  }
  const matches = rankSlashMatches(token, catalog)
  return { active: matches.length > 0, matches, index: 0 }
}

export function applyCompletion(draft: string, command: SlashCommandInfo): string {
  // Replace only the leading slash token; preserve any trailing arguments
  // the user already typed.
  const space = draft.indexOf(' ')
  if (space === -1) {
    return command.name
  }
  return command.name + draft.slice(space)
}
