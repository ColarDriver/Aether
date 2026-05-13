import { atom } from 'nanostores'

import type { JsonObject } from '../gatewayTypes.js'
import {
  EXPLORE_CATEGORIES,
  categoryFor,
  hintForCall,
  verbForCategory,
  verbForTool,
  type ToolCategory,
  CATEGORY_ORDER,
  nounForCategory
} from '../lib/toolCategory.js'

export interface ToolGroupEntry {
  toolCallId: string
  toolName: string
  category: ToolCategory
  verb: string
  detail: string
  isError: boolean
}

export interface ToolGroupRecord {
  // Stable id used by chatStore to anchor the rendered Explored block.
  id: string
  iteration: number
  entries: ToolGroupEntry[]
  counts: Record<ToolCategory, number>
  totalCalls: number
  hasError: boolean
}

export interface ActiveToolGroup extends ToolGroupRecord {
  inFlight: number
  lastHint: string
}

export interface ToolGroupState {
  active: ActiveToolGroup | null
  /** Most recently flushed group — UI may peek at this to display a fade-out. */
  lastFlushed: ToolGroupRecord | null
}

const initialState: ToolGroupState = { active: null, lastFlushed: null }

export const toolGroupState = atom<ToolGroupState>(initialState)

let nextGroupId = 1

function emptyCounts(): Record<ToolCategory, number> {
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

export const toolGroupActions = {
  resetForTests(): void {
    toolGroupState.set({ active: null, lastFlushed: null })
    nextGroupId = 1
  },

  /**
   * Mirror of Python `ToolGroupTracker.start_call`. Always called for every
   * tool dispatch, including standalone (bash/web/etc.). The `track` flag is
   * used by the consumer to decide whether to also render an inline
   * ToolCallPanel (standalone) or wait for the group flush (explore).
   */
  startCall(input: {
    toolCallId: string
    toolName: string
    args: JsonObject
    iteration: number
  }): { isExplore: boolean; category: ToolCategory } {
    const category = categoryFor(input.toolName)
    const isExplore = EXPLORE_CATEGORIES.has(category)
    const detail = hintForCall(input.toolName, input.args)
    const verb = verbForTool(input.toolName)

    if (!isExplore) {
      // Standalone calls flush the active explore burst first so its tree
      // renders above the standalone call header in the transcript.
      toolGroupActions.flushActive()
      return { isExplore, category }
    }

    const current = toolGroupState.get().active ?? createActive(input.iteration)
    current.counts[category] = (current.counts[category] ?? 0) + 1
    current.totalCalls++
    current.inFlight++
    if (detail) {
      current.lastHint = detail
    }
    current.entries.push({
      toolCallId: input.toolCallId,
      toolName: input.toolName,
      category,
      verb,
      detail,
      isError: false
    })
    toolGroupState.set({ ...toolGroupState.get(), active: current })
    return { isExplore, category }
  },

  finishCall(input: { toolCallId: string; isError: boolean }): void {
    const state = toolGroupState.get()
    const active = state.active
    if (!active) {
      return
    }
    const entry = [...active.entries].reverse().find((e) => e.toolCallId === input.toolCallId)
    if (entry) {
      entry.isError = entry.isError || input.isError
    } else if (input.isError) {
      // Fallback: best-effort like the Python heuristic — mark the most
      // recent unmarked entry as errored if the id does not line up.
      for (let i = active.entries.length - 1; i >= 0; i--) {
        const candidate = active.entries[i]
        if (candidate && !candidate.isError) {
          candidate.isError = true
          break
        }
      }
    }
    if (input.isError) {
      active.hasError = true
    }
    if (active.inFlight > 0) {
      active.inFlight--
    }
    toolGroupState.set({ ...state, active: { ...active } })
  },

  /**
   * Flush the active group into `lastFlushed`. Returns the flushed record
   * so the caller (useGatewayEvents) can also push it into chatStore as a
   * `tool-group` chat item.
   */
  flushActive(): ToolGroupRecord | null {
    const state = toolGroupState.get()
    const active = state.active
    if (!active || active.totalCalls === 0 || active.entries.length === 0) {
      if (active) {
        toolGroupState.set({ ...state, active: null })
      }
      return null
    }
    const flushed: ToolGroupRecord = {
      id: active.id,
      iteration: active.iteration,
      entries: active.entries,
      counts: active.counts,
      totalCalls: active.totalCalls,
      hasError: active.hasError
    }
    toolGroupState.set({ active: null, lastFlushed: flushed })
    return flushed
  },

  /** Drop the active group without producing a flush (fatal during turn). */
  discardActive(): void {
    const state = toolGroupState.get()
    if (state.active) {
      toolGroupState.set({ ...state, active: null })
    }
  },

  beginIteration(): ToolGroupRecord | null {
    return toolGroupActions.flushActive()
  }
}

function createActive(iteration: number): ActiveToolGroup {
  return {
    id: `tg_${nextGroupId++}`,
    iteration,
    entries: [],
    counts: emptyCounts(),
    totalCalls: 0,
    hasError: false,
    inFlight: 0,
    lastHint: ''
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Headline summarisation — pure helpers for the UI layer.
// ──────────────────────────────────────────────────────────────────────────

export interface HeadlineSegment {
  category: ToolCategory
  verb: string
  count: number
  noun: string
}

export function buildHeadline(group: ToolGroupRecord, active: boolean): HeadlineSegment[] {
  const result: HeadlineSegment[] = []
  let isFirst = true
  for (const category of CATEGORY_ORDER) {
    const count = group.counts[category] ?? 0
    if (count === 0) {
      continue
    }
    const [first, more] = verbForCategory(category, !active)
    const verb = isFirst ? first : more
    const [singular, plural] = nounForCategory(category)
    const noun = count === 1 ? singular : plural
    result.push({ category, verb, count, noun })
    isFirst = false
  }
  return result
}
