import { atom } from 'nanostores'

import { createHistoryFile, type HistoryFile } from '../lib/fileHistory.js'

const HISTORY_LIMIT = 100

export type ComposerState = {
  draft: string
  /** Cursor offset into `draft` (0-based, inclusive of end position). */
  cursor: number
  multiline: boolean
  history: string[]
  historyIndex: number | null
  /** Messages submitted while a turn was running, kept in FIFO order. */
  queued: string[]
}

const initialState: ComposerState = {
  draft: '',
  cursor: 0,
  multiline: false,
  history: [],
  historyIndex: null,
  queued: []
}

export const composerState = atom<ComposerState>(initialState)

let historyFile: HistoryFile | null = null

function clampCursor(value: number, draftLength: number): number {
  if (!Number.isFinite(value) || value < 0) {
    return 0
  }
  return Math.min(draftLength, Math.floor(value))
}

function isWordChar(char: string | undefined): boolean {
  if (!char) {
    return false
  }
  return /[\p{L}\p{N}_]/u.test(char)
}

export const composerActions = {
  resetForTests(): void {
    composerState.set(initialState)
    historyFile = null
  },

  /**
   * Wire up a persistent on-disk history. Safe to call multiple times — the
   * latest call wins. Existing in-memory history is kept; the file's entries
   * are merged in (deduplicated, preserving order).
   *
   * Tests can pass an in-memory shim implementing the HistoryFile contract.
   */
  attachHistoryFile(file: HistoryFile = createHistoryFile()): void {
    historyFile = file
    const onDisk = file.load()
    if (onDisk.length === 0) {
      return
    }
    const current = composerState.get()
    const merged: string[] = []
    const seen = new Set<string>()
    for (const entry of [...onDisk, ...current.history]) {
      if (!seen.has(entry)) {
        seen.add(entry)
        merged.push(entry)
      }
    }
    composerState.set({
      ...current,
      history: merged.slice(-HISTORY_LIMIT)
    })
  },

  setDraft(draft: string, cursor?: number): void {
    const current = composerState.get()
    composerState.set({
      ...current,
      draft,
      cursor: cursor === undefined ? draft.length : clampCursor(cursor, draft.length),
      multiline: current.multiline || draft.includes('\n') || draft.endsWith('\\')
    })
  },

  insert(input: string): void {
    if (!input) {
      return
    }
    const current = composerState.get()
    const before = current.draft.slice(0, current.cursor)
    const after = current.draft.slice(current.cursor)
    const draft = `${before}${input}${after}`
    composerState.set({
      ...current,
      draft,
      cursor: current.cursor + input.length,
      multiline: current.multiline || draft.includes('\n') || draft.endsWith('\\')
    })
  },

  newline(): void {
    const current = composerState.get()
    const before = current.draft.slice(0, current.cursor)
    const after = current.draft.slice(current.cursor)
    // Backslash-at-cursor → eat the slash and emit a newline (matches Python
    // `Esc+Enter` muscle-memory; in TS we also accept Shift+Enter / Ctrl+J).
    if (before.endsWith('\\')) {
      const draft = `${before.slice(0, -1)}\n${after}`
      composerState.set({
        ...current,
        draft,
        cursor: before.length,
        multiline: true
      })
      return
    }
    const draft = `${before}\n${after}`
    composerState.set({
      ...current,
      draft,
      cursor: current.cursor + 1,
      multiline: true
    })
  },

  backspace(): void {
    const current = composerState.get()
    if (current.cursor === 0) {
      return
    }
    const before = current.draft.slice(0, current.cursor - 1)
    const after = current.draft.slice(current.cursor)
    const draft = `${before}${after}`
    composerState.set({
      ...current,
      draft,
      cursor: current.cursor - 1,
      multiline: draft.includes('\n') || draft.endsWith('\\')
    })
  },

  deleteForward(): void {
    const current = composerState.get()
    if (current.cursor >= current.draft.length) {
      return
    }
    const before = current.draft.slice(0, current.cursor)
    const after = current.draft.slice(current.cursor + 1)
    const draft = `${before}${after}`
    composerState.set({
      ...current,
      draft,
      multiline: draft.includes('\n') || draft.endsWith('\\')
    })
  },

  // ── Cursor movement ──────────────────────────────────────────────────
  moveLeft(): void {
    const current = composerState.get()
    if (current.cursor === 0) {
      return
    }
    composerState.set({ ...current, cursor: current.cursor - 1 })
  },

  moveRight(): void {
    const current = composerState.get()
    if (current.cursor >= current.draft.length) {
      return
    }
    composerState.set({ ...current, cursor: current.cursor + 1 })
  },

  moveWordLeft(): void {
    const current = composerState.get()
    if (current.cursor === 0) {
      return
    }
    let cursor = current.cursor
    // Skip the run of non-word chars first, then the run of word chars —
    // matches the readline / VS Code behaviour users expect.
    while (cursor > 0 && !isWordChar(current.draft[cursor - 1])) {
      cursor--
    }
    while (cursor > 0 && isWordChar(current.draft[cursor - 1])) {
      cursor--
    }
    composerState.set({ ...current, cursor })
  },

  moveWordRight(): void {
    const current = composerState.get()
    if (current.cursor >= current.draft.length) {
      return
    }
    let cursor = current.cursor
    while (cursor < current.draft.length && !isWordChar(current.draft[cursor])) {
      cursor++
    }
    while (cursor < current.draft.length && isWordChar(current.draft[cursor])) {
      cursor++
    }
    composerState.set({ ...current, cursor })
  },

  moveLineStart(): void {
    const current = composerState.get()
    const before = current.draft.slice(0, current.cursor)
    const lastNl = before.lastIndexOf('\n')
    composerState.set({ ...current, cursor: lastNl === -1 ? 0 : lastNl + 1 })
  },

  moveLineEnd(): void {
    const current = composerState.get()
    const after = current.draft.slice(current.cursor)
    const nextNl = after.indexOf('\n')
    const target = nextNl === -1 ? current.draft.length : current.cursor + nextNl
    composerState.set({ ...current, cursor: target })
  },

  moveLineUp(): boolean {
    const current = composerState.get()
    const before = current.draft.slice(0, current.cursor)
    const lastNl = before.lastIndexOf('\n')
    if (lastNl === -1) {
      return false // no line above — caller may fall back to history nav.
    }
    const column = current.cursor - lastNl - 1
    const previousLineStart = before.slice(0, lastNl).lastIndexOf('\n') + 1
    const previousLineEnd = lastNl
    const previousLineLen = previousLineEnd - previousLineStart
    const target = previousLineStart + Math.min(column, previousLineLen)
    composerState.set({ ...current, cursor: target })
    return true
  },

  moveLineDown(): boolean {
    const current = composerState.get()
    const after = current.draft.slice(current.cursor)
    const nextNl = after.indexOf('\n')
    if (nextNl === -1) {
      return false
    }
    const before = current.draft.slice(0, current.cursor)
    const lastNl = before.lastIndexOf('\n')
    const column = current.cursor - lastNl - 1
    const nextLineStart = current.cursor + nextNl + 1
    const remaining = current.draft.slice(nextLineStart)
    const followingNl = remaining.indexOf('\n')
    const nextLineLen = followingNl === -1 ? remaining.length : followingNl
    const target = nextLineStart + Math.min(column, nextLineLen)
    composerState.set({ ...current, cursor: target })
    return true
  },

  // ── Word / line kill operations ─────────────────────────────────────
  deleteWordBackward(): void {
    const current = composerState.get()
    if (current.cursor === 0) {
      return
    }
    let cursor = current.cursor
    while (cursor > 0 && !isWordChar(current.draft[cursor - 1])) {
      cursor--
    }
    while (cursor > 0 && isWordChar(current.draft[cursor - 1])) {
      cursor--
    }
    const draft = `${current.draft.slice(0, cursor)}${current.draft.slice(current.cursor)}`
    composerState.set({
      ...current,
      draft,
      cursor,
      multiline: draft.includes('\n') || draft.endsWith('\\')
    })
  },

  killToLineStart(): void {
    const current = composerState.get()
    const before = current.draft.slice(0, current.cursor)
    const lastNl = before.lastIndexOf('\n')
    const lineStart = lastNl === -1 ? 0 : lastNl + 1
    if (lineStart === current.cursor) {
      return
    }
    const draft = `${current.draft.slice(0, lineStart)}${current.draft.slice(current.cursor)}`
    composerState.set({
      ...current,
      draft,
      cursor: lineStart,
      multiline: draft.includes('\n') || draft.endsWith('\\')
    })
  },

  killToLineEnd(): void {
    const current = composerState.get()
    const after = current.draft.slice(current.cursor)
    const nextNl = after.indexOf('\n')
    const lineEnd = nextNl === -1 ? current.draft.length : current.cursor + nextNl
    if (lineEnd === current.cursor) {
      return
    }
    const draft = `${current.draft.slice(0, current.cursor)}${current.draft.slice(lineEnd)}`
    composerState.set({
      ...current,
      draft,
      multiline: draft.includes('\n') || draft.endsWith('\\')
    })
  },

  clear(): void {
    const current = composerState.get()
    composerState.set({
      ...current,
      draft: '',
      cursor: 0,
      multiline: false,
      historyIndex: null
    })
  },

  commit(text: string): void {
    const trimmed = text.trim()
    const current = composerState.get()
    const history = trimmed
      ? [...current.history.filter((item) => item !== trimmed), trimmed].slice(-HISTORY_LIMIT)
      : current.history
    composerState.set({
      ...current,
      draft: '',
      cursor: 0,
      multiline: false,
      history,
      historyIndex: null
    })
    if (trimmed && historyFile) {
      historyFile.append(trimmed)
    }
  },

  /** Push the current draft (or a literal text) onto the queued backlog. */
  enqueue(text: string): void {
    const trimmed = text.trim()
    if (!trimmed) {
      return
    }
    const current = composerState.get()
    composerState.set({
      ...current,
      draft: '',
      cursor: 0,
      multiline: false,
      historyIndex: null,
      queued: [...current.queued, trimmed]
    })
  },

  /** Pop the most-recently-queued message back into the draft. */
  popQueued(): string | null {
    const current = composerState.get()
    if (current.queued.length === 0) {
      return null
    }
    const popped = current.queued[current.queued.length - 1] ?? null
    const draft = popped ?? current.draft
    composerState.set({
      ...current,
      queued: current.queued.slice(0, -1),
      draft,
      cursor: draft.length,
      multiline: draft.includes('\n') || draft.endsWith('\\')
    })
    return popped
  },

  /** Pop the FRONT of the queued list — used to replay backlog after a turn. */
  shiftQueued(): string | null {
    const current = composerState.get()
    if (current.queued.length === 0) {
      return null
    }
    const head = current.queued[0] ?? null
    composerState.set({ ...current, queued: current.queued.slice(1) })
    return head
  },

  clearQueued(): void {
    const current = composerState.get()
    if (current.queued.length === 0) {
      return
    }
    composerState.set({ ...current, queued: [] })
  },

  previousHistory(): void {
    const current = composerState.get()
    if (
      current.multiline ||
      current.history.length === 0 ||
      (current.draft && current.historyIndex === null)
    ) {
      return
    }
    const index =
      current.historyIndex === null
        ? current.history.length - 1
        : Math.max(0, current.historyIndex - 1)
    const draft = current.history[index] ?? ''
    composerState.set({ ...current, draft, cursor: draft.length, historyIndex: index })
  },

  nextHistory(): void {
    const current = composerState.get()
    if (current.historyIndex === null) {
      return
    }
    const index = current.historyIndex + 1
    if (index >= current.history.length) {
      composerState.set({ ...current, draft: '', cursor: 0, historyIndex: null })
      return
    }
    const draft = current.history[index] ?? ''
    composerState.set({ ...current, draft, cursor: draft.length, historyIndex: index })
  }
}
