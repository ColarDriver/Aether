import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, resolve } from 'node:path'

const DEFAULT_PATH = resolve(homedir(), '.aether', 'repl_history')
const MAX_LINES = 500

/**
 * NDJSON-line history file mirroring Python's `prompt_toolkit` `FileHistory`
 * shape, kept intentionally minimal so a corrupt file never breaks startup.
 *
 * Each line is a JSON-encoded string. We pick JSON over raw text because
 * users frequently submit multi-line prompts; storing them as a single
 * JSON string per row keeps loading robust against embedded newlines.
 */
export interface HistoryFile {
  path: string
  load(): string[]
  append(entry: string): void
}

export function createHistoryFile(path: string = DEFAULT_PATH): HistoryFile {
  return {
    path,
    load() {
      try {
        if (!existsSync(path)) {
          return []
        }
        const raw = readFileSync(path, 'utf8')
        const lines = raw.split(/\r?\n/).filter(Boolean)
        const parsed: string[] = []
        for (const line of lines) {
          const decoded = tryDecode(line)
          if (decoded !== null && decoded !== '') {
            parsed.push(decoded)
          }
        }
        // Trim to the most recent MAX_LINES entries — keeps the in-memory
        // ring bounded without us having to compact the file every time.
        return parsed.slice(-MAX_LINES)
      } catch {
        return []
      }
    },
    append(entry) {
      try {
        const trimmed = entry.trim()
        if (!trimmed) {
          return
        }
        ensureDirectory(path)
        appendFileSync(path, `${JSON.stringify(trimmed)}\n`, 'utf8')
      } catch {
        // History persistence is a nice-to-have; never fail the turn.
      }
    }
  }
}

function ensureDirectory(filePath: string): void {
  const dir = dirname(filePath)
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true })
  }
}

function tryDecode(line: string): string | null {
  try {
    const parsed = JSON.parse(line)
    if (typeof parsed === 'string') {
      return parsed
    }
    return null
  } catch {
    return null
  }
}

export const DEFAULT_HISTORY_PATH = DEFAULT_PATH
