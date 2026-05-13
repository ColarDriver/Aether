import { atom } from 'nanostores'

import type { JsonObject, TranscriptMessage } from '../gatewayTypes.js'
import type { ToolGroupRecord } from './toolGroupStore.js'

export type ChatItem =
  | { kind: 'user'; id: string; text: string; ts: number }
  | { kind: 'assistant'; id: string; text: string; streaming: boolean; ts: number }
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
      result?: { text: string; isError: boolean }
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
  | { kind: 'note'; id: string; text: string; level: 'info' | 'warn' | 'error'; ts: number }

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
    const index = items.findIndex((item) => item.kind === 'assistant' && item.id === id)
    if (index === -1) {
      chatItems.set([
        ...items,
        {
          kind: 'assistant',
          id,
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
      next[index] = { ...existing, text: `${existing.text}${text}`, streaming: true }
      chatItems.set(next)
    }
  },

  finishAssistant(id: string, finalText?: string): void {
    const items = chatItems.get()
    const index = items.findIndex((item) => item.kind === 'assistant' && item.id === id)
    if (index === -1) {
      if (finalText) {
        chatItems.set([
          ...items,
          {
            kind: 'assistant',
            id,
            text: finalText,
            streaming: false,
            ts: Date.now()
          }
        ])
      }
      return
    }

    const next = [...items]
    const existing = next[index]
    if (existing?.kind === 'assistant') {
      const text = existing.text || finalText || ''
      next[index] = { ...existing, text, streaming: false }
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
      next[matchIdx] = {
        ...existing,
        durationMs,
        result: { text: input.text, isError: input.isError }
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

  pushNote(text: string, level: 'info' | 'warn' | 'error' = 'info'): ChatItem {
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
    const items: ChatItem[] = []
    for (const message of messages) {
      const text = message.text ?? ''
      if (!text) {
        continue
      }
      if (message.role === 'user') {
        items.push({ kind: 'user', id: makeId('user'), text, ts: Date.now() })
      } else if (message.role === 'assistant') {
        items.push({
          kind: 'assistant',
          id: makeId('assistant'),
          text,
          streaming: false,
          ts: Date.now()
        })
      } else if (message.role === 'system') {
        items.push({ kind: 'note', id: makeId('note'), text, level: 'info', ts: Date.now() })
      }
    }
    chatItems.set(items)
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
