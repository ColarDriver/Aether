import { atom } from 'nanostores'

import type {
  JsonObject,
  ToolResultMetadata,
  TranscriptMessage,
  TranscriptToolCall
} from '../gatewayTypes.js'
import {
  EXPLORE_CATEGORIES,
  categoryFor,
  hintForCall,
  verbForTool,
  type ToolCategory
} from '../lib/toolCategory.js'
import type { ToolGroupEntry, ToolGroupRecord } from './toolGroupStore.js'

export interface ToolCallSummary {
  path: string
  linesAdded: number
  linesRemoved: number
  hunks?: number
  diff?: string
  noOp?: boolean
}

export type ChatItem =
  | { kind: 'user'; id: string; text: string; ts: number }
  | { kind: 'assistant'; id: string; runId: string; text: string; streaming: boolean; ts: number }
  | {
      kind: 'tool-call'
      id: string
      toolCallId: string
      toolName: string
      args: JsonObject
      argsPreview: string
      ts: number
      iteration: number
      // Set by useGatewayEvents from `categoryFor`. Lets the renderer know
      // whether this row belongs to an explore burst (collapsed under
      // ExploredTree) or stands alone (rendered as ToolCallPanel).
      coalesce: boolean
      durationMs: number | null
      result?: { text: string; isError: boolean; metadata?: ToolResultMetadata }
      // Set when the matching tool.result arrives with edit/write
      // metadata. Drives the EditSummary chat row ("● Edited X (+N −M)").
      summary?: ToolCallSummary
    }
  | {
      kind: 'tool-result'
      id: string
      toolCallId: string
      text: string
      isError: boolean
      ts: number
    }
  | {
      kind: 'tool-group'
      id: string
      group: ToolGroupRecord
      ts: number
    }
  | {
      kind: 'note'
      id: string
      text: string
      // Mirrors Python `ui.py` `success/info/warn/error` helpers — each
      // level has its own colour + icon glyph. 'success' is kept distinct
      // from 'info' so positive confirmations ("started session", "model
      // set") get the ✓ glyph rather than the ℹ generic info marker.
      level: 'success' | 'info' | 'warn' | 'error'
      ts: number
    }

let nextId = 1

export const chatItems = atom<ChatItem[]>([])
export const verboseMode = atom(false)

export const chatActions = {
  reset(): void {
    chatItems.set([])
  },

  resetForTests(): void {
    nextId = 1
    chatItems.set([])
    verboseMode.set(false)
  },

  addUserMessage(text: string): ChatItem {
    const item: ChatItem = {
      kind: 'user',
      id: makeId('user'),
      text,
      ts: Date.now()
    }
    chatItems.set([...chatItems.get(), item])
    return item
  },

  appendAssistant(id: string, text: string): void {
    const items = chatItems.get()
    const index = lastIndex(
      items,
      (item) => item.kind === 'assistant' && item.runId === id && item.streaming
    )
    if (index === -1) {
      chatItems.set([
        ...items,
        {
          kind: 'assistant',
          id: makeId('assistant'),
          runId: id,
          text,
          streaming: true,
          ts: Date.now()
        }
      ])
      return
    }

    const next = [...items]
    const existing = next[index]
    if (existing?.kind === 'assistant') {
      if (index !== items.length - 1) {
        next[index] = { ...existing, streaming: false }
        next.push({
          kind: 'assistant',
          id: makeId('assistant'),
          runId: id,
          text,
          streaming: true,
          ts: Date.now()
        })
        chatItems.set(next)
        return
      }
      next[index] = { ...existing, text: `${existing.text}${text}`, streaming: true }
      chatItems.set(next)
    }
  },

  finishAssistant(id: string, finalText?: string): void {
    const items = chatItems.get()
    const indexes = items.flatMap((item, index) =>
      item.kind === 'assistant' && item.runId === id ? [index] : []
    )
    const lastIndexForRun = indexes[indexes.length - 1] ?? -1
    if (lastIndexForRun === -1) {
      if (finalText) {
        chatItems.set([
          ...items,
          {
            kind: 'assistant',
            id: makeId('assistant'),
            runId: id,
            text: finalText,
            streaming: false,
            ts: Date.now()
          }
        ])
      }
      return
    }

    const next = [...items]
    const existing = next[lastIndexForRun]
    if (existing?.kind === 'assistant') {
      const rendered = indexes
        .map((index) => next[index])
        .filter((item): item is Extract<ChatItem, { kind: 'assistant' }> => item?.kind === 'assistant')
        .map((item) => item.text)
        .join('')
      const suffix =
        finalText && finalText.startsWith(rendered)
          ? finalText.slice(rendered.length)
          : ''

      if (suffix) {
        if (lastIndexForRun !== next.length - 1) {
          next[lastIndexForRun] = { ...existing, streaming: false }
          next.push({
            kind: 'assistant',
            id: makeId('assistant'),
            runId: id,
            text: suffix,
            streaming: false,
            ts: Date.now()
          })
          chatItems.set(next)
          return
        }
        next[lastIndexForRun] = {
          ...existing,
          text: `${existing.text}${suffix}`,
          streaming: false
        }
        chatItems.set(next)
        return
      }

      const text = existing.text || finalText || ''
      next[lastIndexForRun] = { ...existing, text, streaming: false }
      chatItems.set(next)
    }
  },

  pushToolCall(input: {
    id: string
    toolName: string
    args: JsonObject
    iteration: number
    coalesce: boolean
  }): void {
    chatItems.set([
      ...chatItems.get(),
      {
        kind: 'tool-call',
        id: makeId('tool-call'),
        toolCallId: input.id,
        toolName: input.toolName,
        args: input.args,
        argsPreview: previewArgs(input.args),
        iteration: input.iteration,
        coalesce: input.coalesce,
        durationMs: null,
        ts: Date.now()
      }
    ])
  },

  pushToolResult(input: {
    toolCallId: string
    toolName?: string
    text: string
    isError: boolean
    metadata?: ToolResultMetadata
  }): void {
    const items = chatItems.get()
    // Attach the result to the matching tool-call item so ToolCallPanel can
    // render the call + result inside one collapsible block.
    const matchIdx = lastIndex(items, (item) =>
      item.kind === 'tool-call' && item.toolCallId === input.toolCallId
    )
    if (matchIdx === -1) {
      chatItems.set([
        ...items,
        {
          kind: 'tool-result',
          id: makeId('tool-result'),
          toolCallId: input.toolCallId,
          text: input.text,
          isError: input.isError,
          ts: Date.now()
        }
      ])
      return
    }
    const next = [...items]
    const existing = next[matchIdx]
    if (existing?.kind === 'tool-call') {
      const durationMs = Date.now() - existing.ts
      const summary = input.isError
        ? undefined
        : buildEditSummary(existing.toolName, input.metadata)
      next[matchIdx] = {
        ...existing,
        durationMs,
        result: {
          text: input.text,
          isError: input.isError,
          ...(input.metadata ? { metadata: input.metadata } : {})
        },
        ...(summary ? { summary } : {})
      }
      chatItems.set(next)
    }
  },

  pushToolGroup(group: ToolGroupRecord): void {
    chatItems.set([
      ...chatItems.get(),
      {
        kind: 'tool-group',
        id: makeId('tool-group'),
        group,
        ts: Date.now()
      }
    ])
  },

  pushNote(
    text: string,
    level: 'success' | 'info' | 'warn' | 'error' = 'info'
  ): ChatItem {
    const item: ChatItem = {
      kind: 'note',
      id: makeId('note'),
      text,
      level,
      ts: Date.now()
    }
    chatItems.set([...chatItems.get(), item])
    return item
  },

  replaceTranscript(messages: TranscriptMessage[]): void {
    chatItems.set(rebuildChatItemsFromTranscript(messages))
  },

  toggleVerbose(): boolean {
    const next = !verboseMode.get()
    verboseMode.set(next)
    return next
  }
}

export function previewArgs(args: JsonObject): string {
  const raw = JSON.stringify(args)
  if (!raw) {
    return '{}'
  }
  return raw.length > 120 ? `${raw.slice(0, 117)}...` : raw
}

/**
 * Build a ToolCallSummary from a tool.result event's metadata, but only
 * when the tool was an edit/write and the metadata carries a usable path.
 * Returns undefined for shell results, lookups, and failed calls — the
 * EditSummary chat row only renders when there is something concrete to
 * report (file path + line counts).
 */
export function buildEditSummary(
  toolName: string,
  metadata: ToolResultMetadata | undefined
): ToolCallSummary | undefined {
  if (!metadata) return undefined
  const category = categoryFor(toolName)
  if (category !== 'edit' && category !== 'write') return undefined
  const path = typeof metadata.path === 'string' ? metadata.path : undefined
  if (!path) return undefined
  const summary: ToolCallSummary = {
    path,
    linesAdded: numberOr(metadata.lines_added, 0),
    linesRemoved: numberOr(metadata.lines_removed, 0)
  }
  if (typeof metadata.hunks === 'number') {
    summary.hunks = metadata.hunks
  }
  if (typeof metadata.diff === 'string') {
    summary.diff = metadata.diff
  }
  if (metadata.no_op === true) {
    summary.noOp = true
  }
  return summary
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

/**
 * Replay a saved session transcript into the ChatItem list shape the
 * renderer expects. Mirrors what `useGatewayEvents` does live for the
 * `tool.call` / `tool.result` / `iteration.end` events:
 *
 *   * Assistant text becomes a non-streaming `assistant` ChatItem.
 *   * Each entry in `assistant.tool_calls` becomes a `tool-call`
 *     ChatItem (coalesce flag set by `categoryFor`).
 *   * Tool results back-fill the matching `tool-call` (so the
 *     ToolCallPanel can render call + result together) and, when the
 *     metadata includes `path`/`lines_added`/`lines_removed`, attach a
 *     `summary` so `EditSummary` shows `● Update(path)` in chat.
 *   * Explore-category bursts (read/search/list and failed write/edit)
 *     get flushed into a `tool-group` ChatItem at the next user message
 *     or end-of-transcript — same filter live `flushActive` uses.
 *
 * System messages are dropped; they carry the runtime system prompt
 * (often >1 KB of XML) and the Banner already shows session metadata.
 */
export function rebuildChatItemsFromTranscript(
  messages: TranscriptMessage[]
): ChatItem[] {
  const items: ChatItem[] = []
  const toolCallIndex: Map<string, number> = new Map()
  // Held in a single-slot holder so the closures below can read/write
  // the current pending group without tripping TS's narrowing checks
  // around let-bound nullable closures.
  const groupHolder: { current: PendingToolGroup | null } = { current: null }

  const flushPending = (): void => {
    const pending = groupHolder.current
    if (!pending) return
    const visible = pending.entries.filter(
      (e) => !((e.category === 'write' || e.category === 'edit') && !e.isError)
    )
    if (visible.length > 0) {
      const counts = emptyCategoryCounts()
      for (const entry of visible) {
        counts[entry.category] = (counts[entry.category] ?? 0) + 1
      }
      items.push({
        kind: 'tool-group',
        id: makeId('tool-group'),
        ts: Date.now(),
        group: {
          id: pending.id,
          iteration: pending.iteration,
          entries: visible,
          counts,
          totalCalls: visible.length,
          hasError: visible.some((e) => e.isError)
        }
      })
    }
    groupHolder.current = null
  }

  let iteration = 0
  for (const message of messages) {
    if (message.role === 'system') {
      // Drop — the system prompt is internal plumbing.
      continue
    }
    const text = message.text ?? ''

    if (message.role === 'user') {
      flushPending()
      if (text) {
        items.push({
          kind: 'user',
          id: makeId('user'),
          text,
          ts: Date.now()
        })
      }
      iteration += 1
      continue
    }

    if (message.role === 'assistant') {
      // Assistant text always flushes the prior explore burst so the
      // tree lands above the new narrative text (live wiring does the
      // same on iteration.end / done).
      if (text) {
        flushPending()
        items.push({
          kind: 'assistant',
          id: makeId('assistant'),
          runId: makeId('assistant-run'),
          text,
          streaming: false,
          ts: Date.now()
        })
      }
      const toolCalls = message.tool_calls ?? []
      for (const call of toolCalls) {
        pushReplayedToolCall({
          call,
          iteration,
          items,
          toolCallIndex,
          ensureGroup: () => {
            if (!groupHolder.current) {
              groupHolder.current = {
                id: `tg_replay_${items.length}`,
                iteration,
                entries: []
              }
            }
            return groupHolder.current
          }
        })
      }
      continue
    }

    if (message.role === 'tool') {
      const toolCallId = message.tool_call_id ?? ''
      const matchIdx = toolCallId ? toolCallIndex.get(toolCallId) ?? -1 : -1
      const metadata = (message.metadata ?? undefined) as
        | ToolResultMetadata
        | undefined
      const isError = Boolean(message.is_error)
      if (matchIdx >= 0) {
        const existing = items[matchIdx]
        if (existing?.kind === 'tool-call') {
          const summary = isError
            ? undefined
            : buildEditSummary(existing.toolName, metadata)
          items[matchIdx] = {
            ...existing,
            durationMs: 0,
            result: {
              text,
              isError,
              ...(metadata ? { metadata } : {})
            },
            ...(summary ? { summary } : {})
          }
          // Propagate error flag to the matching group entry so a
          // failed read/search still shows `(failed)` in Explored.
          const groupSnapshot = groupHolder.current
          if (groupSnapshot) {
            for (const entry of groupSnapshot.entries) {
              if (entry.toolCallId === toolCallId) {
                entry.isError = entry.isError || isError
                break
              }
            }
          }
        }
      } else if (text || isError) {
        // Orphan result (no matching tool-call seen) — emit a thin
        // standalone tool-result so the user is not missing data.
        items.push({
          kind: 'tool-result',
          id: makeId('tool-result'),
          toolCallId,
          text,
          isError,
          ts: Date.now()
        })
      }
      continue
    }
  }

  flushPending()
  return items
}

interface PendingToolGroup {
  id: string
  iteration: number
  entries: ToolGroupEntry[]
}

interface PushReplayedToolCallInput {
  call: TranscriptToolCall
  iteration: number
  items: ChatItem[]
  toolCallIndex: Map<string, number>
  ensureGroup: () => PendingToolGroup
}

function pushReplayedToolCall(input: PushReplayedToolCallInput): void {
  const category = categoryFor(input.call.name)
  const isExplore = EXPLORE_CATEGORIES.has(category)
  const args = (input.call.arguments ?? {}) as JsonObject
  const item: ChatItem = {
    kind: 'tool-call',
    id: makeId('tool-call'),
    toolCallId: input.call.id,
    toolName: input.call.name,
    args,
    argsPreview: previewArgs(args),
    ts: Date.now(),
    iteration: input.iteration,
    coalesce: isExplore,
    durationMs: null
  }
  input.items.push(item)
  input.toolCallIndex.set(input.call.id, input.items.length - 1)
  if (isExplore) {
    const group = input.ensureGroup()
    group.entries.push({
      toolCallId: input.call.id,
      toolName: input.call.name,
      category,
      verb: verbForTool(input.call.name),
      detail: hintForCall(input.call.name, args),
      isError: false
    })
  }
}

function emptyCategoryCounts(): Record<ToolCategory, number> {
  return {
    search: 0,
    read: 0,
    list: 0,
    write: 0,
    edit: 0,
    bash: 0,
    web: 0,
    subagent: 0,
    mcp: 0,
    other: 0
  }
}

function makeId(prefix: string): string {
  const id = `${prefix}_${nextId}`
  nextId += 1
  return id
}

function lastIndex<T>(arr: readonly T[], predicate: (item: T) => boolean): number {
  for (let i = arr.length - 1; i >= 0; i--) {
    const value = arr[i]
    if (value !== undefined && predicate(value)) {
      return i
    }
  }
  return -1
}
