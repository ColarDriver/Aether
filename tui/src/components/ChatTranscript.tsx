import { Box, Static, Text, useInput, useStdin } from 'ink'
import { useStore } from '@nanostores/react'
import stringWidth from 'string-width'
import { useEffect, useMemo, useRef, useState, type ReactElement, type ReactNode } from 'react'

import { stripToolBlocks } from '../lib/phantomTool.js'
import { mouseButtonCodes } from '../lib/terminalMouse.js'
import { isTurnFooterText } from '../lib/turnFooter.js'
import { chatEpoch, chatItems, type ChatItem } from '../store/chatStore.js'
import { focusActions, focusOwner } from '../store/focusStore.js'
import { overlayStack } from '../store/overlayStore.js'

import { ChatMessage } from './ChatMessage.js'

const MAX_VISIBLE = 80
const DEFAULT_WIDTH = 80
const WHEEL_SCROLL_ROWS = 3
const LIVE_CONTEXT_ITEMS = 10

export interface ChatTranscriptProps {
  viewportRows?: number
  width?: number
  leading?: ReactNode
  leadingRows?: number
  items?: ChatItem[]
  staticScrollback?: boolean
}

/**
 * Tab/Shift+Tab cycle the focus between standalone tool-call items in the
 * transcript; Enter on a focused row toggles its expanded state.
 *
 * Focus arbitration is via `focusOwner`: the composer hands focus over when
 * the user presses Tab; this hook hands focus back to the composer when the
 * user Tabs past the last row (or has nothing focusable to begin with).
 */
export function ChatTranscript({
  viewportRows,
  width = terminalWidth(),
  leading,
  leadingRows = 0,
  items: inputItems,
  staticScrollback = true
}: ChatTranscriptProps = {}): ReactElement | null {
  const storeItems = useStore(chatItems)
  const staticEpoch = useStore(chatEpoch)
  const items = inputItems ?? storeItems
  const overlays = useStore(overlayStack)
  const owner = useStore(focusOwner)
  const { stdin } = useStdin()
  const [focusedToolCallId, setFocusedToolCallId] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  const [scrollOffset, setScrollOffset] = useState(0)
  const previousLastIdRef = useRef<string | null>(null)

  const rowBudget = normaliseRows(viewportRows)
  const usesStaticScrollback = rowBudget === null && staticScrollback
  const renderableItems = useMemo(
    () => items.filter((item) => renderKind(item) !== null),
    [items]
  )
  const allVisible = useMemo(
    () => (usesStaticScrollback ? renderableItems : renderableItems.slice(-MAX_VISIBLE)),
    [renderableItems, usesStaticScrollback]
  )
  const split = useMemo(
    () => splitStaticPrefix(allVisible, usesStaticScrollback),
    [allVisible, usesStaticScrollback]
  )
  const staticItems = usesStaticScrollback ? split.staticItems : []
  const visible = usesStaticScrollback ? split.liveItems : allVisible
  const leadingBudget = leading && !usesStaticScrollback ? normaliseRows(leadingRows) ?? 0 : 0
  const lastStaticItem = staticItems[staticItems.length - 1] ?? null
  const viewport = useMemo(
    () =>
      rowBudget === null
        ? null
        : buildViewport(visible, rowBudget, scrollOffset, width, leadingBudget),
    [leadingBudget, rowBudget, scrollOffset, visible, width]
  )
  const renderStartIndex = viewport?.startIndex ?? 0
  const renderItems = useMemo(
    () =>
      viewport
        ? viewport.items.map((entry) =>
            limitItemRows(entry.item, Math.max(1, entry.rows), width, entry.startRow)
          )
        : visible,
    [viewport, visible, width]
  )

  const focusableTools = useMemo(
    () =>
      renderItems.filter(
        (item): item is Extract<ChatItem, { kind: 'tool-call' }> =>
          item.kind === 'tool-call' && !item.coalesce
      ),
    [renderItems]
  )
  const topOverlay = overlays[overlays.length - 1] ?? null
  const scrollKeysActive = rowBudget !== null && (overlays.length === 0 || topOverlay?.kind === 'permission')

  useEffect(() => {
    const last = visible[visible.length - 1] ?? null
    const previousLastId = previousLastIdRef.current
    previousLastIdRef.current = last?.id ?? null
    if (last?.kind === 'user' && last.id !== previousLastId) {
      setScrollOffset(0)
    }
  }, [visible])

  useEffect(() => {
    if (!viewport) {
      if (scrollOffset !== 0) {
        setScrollOffset(0)
      }
      return
    }
    if (scrollOffset > viewport.maxOffset) {
      setScrollOffset(viewport.maxOffset)
    }
  }, [scrollOffset, viewport])

  // Keep focus pointing at a still-visible tool when the transcript shifts
  // (older items fall off the bottom-N window).
  useEffect(() => {
    if (
      focusedToolCallId &&
      !focusableTools.find((tool) => tool.toolCallId === focusedToolCallId)
    ) {
      const last = focusableTools[focusableTools.length - 1] ?? null
      setFocusedToolCallId(last?.toolCallId ?? null)
    }
  }, [focusableTools, focusedToolCallId])

  // If the composer-side handed focus to us but we have nothing focusable,
  // hand it back so the user keeps typing without a dead Tab.
  useEffect(() => {
    if (owner === 'transcript' && focusableTools.length === 0) {
      focusActions.set('composer')
    }
  }, [owner, focusableTools])

  useInput(
    (input, key) => {
      if (key.tab && !key.ctrl && !key.meta) {
        if (focusableTools.length === 0) {
          focusActions.set('composer')
          return
        }
        const direction = key.shift ? -1 : 1
        const index = focusedToolCallId
          ? focusableTools.findIndex((tool) => tool.toolCallId === focusedToolCallId)
          : -1
        // Cycle through tools; an extra step at each end hands focus back to
        // the composer so Tab on the last row escapes the focus loop cleanly.
        const total = focusableTools.length + 1 // +1 for the composer slot
        const startIndex = index === -1 ? (direction === 1 ? -1 : focusableTools.length) : index
        const nextIndex = (startIndex + direction + total) % total
        if (nextIndex === focusableTools.length) {
          setFocusedToolCallId(null)
          focusActions.set('composer')
          return
        }
        const nextTool = focusableTools[nextIndex]
        if (nextTool) {
          setFocusedToolCallId(nextTool.toolCallId)
        }
        return
      }
      if (key.escape) {
        setFocusedToolCallId(null)
        focusActions.set('composer')
        return
      }
      if (viewport) {
        const page = Math.max(1, viewport.contentRows)
        if (key.upArrow) {
          setScrollOffset((current) => clamp(current + 1, 0, viewport.maxOffset))
          return
        }
        if (key.downArrow) {
          setScrollOffset((current) => clamp(current - 1, 0, viewport.maxOffset))
          return
        }
        if (key.pageUp) {
          setScrollOffset((current) => clamp(current + page, 0, viewport.maxOffset))
          return
        }
        if (key.pageDown) {
          setScrollOffset((current) => clamp(current - page, 0, viewport.maxOffset))
          return
        }
        if (key.home) {
          setScrollOffset(viewport.maxOffset)
          return
        }
        if (key.end) {
          setScrollOffset(0)
          return
        }
      }
      if (key.return && focusedToolCallId) {
        setExpanded((current) => {
          const next = new Set(current)
          if (next.has(focusedToolCallId)) {
            next.delete(focusedToolCallId)
          } else {
            next.add(focusedToolCallId)
          }
          return next
        })
        return
      }
    },
    { isActive: overlays.length === 0 && owner === 'transcript' }
  )

  useInput(
    (_input, key) => {
      if (!viewport) {
        return
      }
      const page = Math.max(1, viewport.contentRows)
      if (key.pageUp) {
        setScrollOffset((current) => clamp(current + page, 0, viewport.maxOffset))
        return
      }
      if (key.pageDown) {
        setScrollOffset((current) => clamp(current - page, 0, viewport.maxOffset))
        return
      }
      if ((key.ctrl && key.home) || (owner === 'transcript' && key.home)) {
        setScrollOffset(viewport.maxOffset)
        return
      }
      if ((key.ctrl && key.end) || (owner === 'transcript' && key.end)) {
        setScrollOffset(0)
      }
    },
    { isActive: scrollKeysActive }
  )

  useEffect(() => {
    if (!scrollKeysActive || !viewport) {
      return
    }
    const handleData = (data: Buffer | string) => {
      const text = Buffer.isBuffer(data) ? data.toString('utf8') : data
      for (const button of mouseButtonCodes(text)) {
        if (button === 64) {
          setScrollOffset((current) => clamp(current + WHEEL_SCROLL_ROWS, 0, viewport.maxOffset))
        } else if (button === 65) {
          setScrollOffset((current) => clamp(current - WHEEL_SCROLL_ROWS, 0, viewport.maxOffset))
        }
      }
    }
    stdin.on('data', handleData)
    return () => {
      stdin.off('data', handleData)
    }
  }, [scrollKeysActive, stdin, viewport])

  const staticNode = usesStaticScrollback ? (
    <Static key={staticEpoch} items={buildStaticEntries(staticItems, leading)}>
      {(entry, index) => {
        if (entry.kind === 'leading') {
          return (
            <Box key={entry.id} flexDirection="column">
              {entry.node}
            </Box>
          )
        }
        return (
          <TranscriptItemRow
            key={entry.item.id}
            item={entry.item}
            previous={previousStaticEntryItem(staticItems, index - (leading ? 1 : 0))}
            focused={false}
            expanded={false}
          />
        )
      }}
    </Static>
  ) : null

  if (rowBudget === 0) {
    return null
  }

  const placeholder = (
    <Box marginTop={1}>
      <Text color="gray">Type a message or /help for commands.</Text>
    </Box>
  )
  const containerProps =
    rowBudget !== null ? { height: rowBudget, overflow: 'hidden' as const } : {}
  const staticOnly = usesStaticScrollback && visible.length === 0
  const showPlaceholderBeforeTranscript = allVisible.length === 0 && (!leading || staticOnly)

  // No outer marginY — Python's prompt_toolkit transcript runs flush against
  // the banner above and the activity bar below; the per-item ChatMessage
  // marginTop already supplies visual breathing between turns.
  return (
    <Box flexDirection="column" {...containerProps}>
      {staticNode}
      {showPlaceholderBeforeTranscript ? (
        <>
          {rowBudget !== null ? <Box flexGrow={1} /> : null}
          {placeholder}
        </>
      ) : null}
      {staticOnly ? null : (
        <>
          {viewport && !viewport.hiddenAbove && !viewport.hiddenBelow ? <Box flexGrow={1} /> : null}
          {viewport?.hiddenAbove ? (
            <Text color="gray">↑ older transcript · PageUp/Home</Text>
          ) : null}
          {leading && (!viewport || viewport.leadingVisible) ? (
            <Box flexDirection="column">{leading}</Box>
          ) : null}
          {visible.length === 0 && leading && !showPlaceholderBeforeTranscript ? (
            <Box marginTop={1}>
              <Text color="gray">Type a message or /help for commands.</Text>
            </Box>
          ) : null}
          {renderItems.map((item, index) => {
            const viewportEntry = viewport?.items[index] ?? null
            const previousIndex = renderStartIndex + index - 1
            const previous = previousIndex >= 0 ? visible[previousIndex] ?? null : lastStaticItem
            return (
              <TranscriptItemRow
                key={item.id}
                item={item}
                previous={previous}
                focused={item.kind === 'tool-call' && item.toolCallId === focusedToolCallId}
                expanded={item.kind === 'tool-call' && expanded.has(item.toolCallId)}
                startRow={viewportEntry?.startRow ?? 0}
              />
            )
          })}
          {viewport?.hiddenBelow ? (
            <Text color="gray">↓ newer transcript · PageDown/End</Text>
          ) : null}
        </>
      )}
    </Box>
  )
}

interface StaticLeadingEntry {
  kind: 'leading'
  id: 'leading'
  node: ReactNode
}

interface StaticItemEntry {
  kind: 'item'
  item: ChatItem
}

type StaticEntry = StaticLeadingEntry | StaticItemEntry

function buildStaticEntries(items: ChatItem[], leading: ReactNode | undefined): StaticEntry[] {
  return [
    ...(leading ? [{ kind: 'leading' as const, id: 'leading' as const, node: leading }] : []),
    ...items.map((item) => ({ kind: 'item' as const, item }))
  ]
}

function previousStaticEntryItem(staticItems: ChatItem[], staticItemIndex: number): ChatItem | null {
  if (staticItemIndex <= 0) {
    return null
  }
  return staticItems[staticItemIndex - 1] ?? null
}

function TranscriptItemRow({
  item,
  previous,
  focused,
  expanded,
  startRow = 0
}: {
  item: ChatItem
  previous: ChatItem | null
  focused: boolean
  expanded: boolean
  startRow?: number
}): ReactElement {
  const spacedAbove = startRow === 0 && shouldInsertSpacer(previous, item)
  return (
    <Box key={item.id} marginTop={spacedAbove ? 1 : 0}>
      <ChatMessage item={item} focused={focused} expanded={expanded} />
    </Box>
  )
}

function splitStaticPrefix(
  items: ChatItem[],
  enabled: boolean
): { staticItems: ChatItem[]; liveItems: ChatItem[] } {
  if (!enabled) {
    return { staticItems: [], liveItems: items }
  }
  let stablePrefixEnd = 0
  while (stablePrefixEnd < items.length && isStaticTranscriptItem(items[stablePrefixEnd])) {
    stablePrefixEnd += 1
  }
  // Keep the recent stable tail live so the user can still see the previous
  // turn and the just-submitted prompt while the next assistant response
  // streams. Older stable rows are printed once into terminal scrollback.
  const staticEnd = Math.max(0, stablePrefixEnd - LIVE_CONTEXT_ITEMS)
  return {
    staticItems: items.slice(0, staticEnd),
    liveItems: items.slice(staticEnd)
  }
}

function isStaticTranscriptItem(item: ChatItem | undefined): boolean {
  if (!item) {
    return false
  }
  switch (item.kind) {
    case 'user':
    case 'tool-group':
    case 'tool-result':
    case 'note':
      return true
    case 'assistant':
      return !item.streaming
    case 'tool-call':
      return item.durationMs !== null && Boolean(item.result)
  }
}

interface MeasuredItem {
  item: ChatItem
  index: number
  start: number
  contentStart: number
  contentEnd: number
  end: number
}

interface TranscriptViewportItem {
  item: ChatItem
  rows: number
  startRow: number
}

interface TranscriptViewport {
  items: TranscriptViewportItem[]
  startIndex: number
  hiddenAbove: boolean
  hiddenBelow: boolean
  maxOffset: number
  contentRows: number
  leadingVisible: boolean
}

function buildViewport(
  items: ChatItem[],
  maxRows: number,
  scrollOffset: number,
  width: number,
  leadingRows = 0
): TranscriptViewport {
  const firstPass = measureViewport(items, maxRows, scrollOffset, width, leadingRows)
  const hintRows = (firstPass.hiddenAbove ? 1 : 0) + (firstPass.hiddenBelow ? 1 : 0)
  if (hintRows === 0) {
    return { ...firstPass, contentRows: maxRows }
  }
  const contentRows = Math.max(1, maxRows - hintRows)
  return { ...measureViewport(items, contentRows, scrollOffset, width, leadingRows), contentRows }
}

function measureViewport(
  items: ChatItem[],
  rows: number,
  scrollOffset: number,
  width: number,
  leadingRows = 0
): Omit<TranscriptViewport, 'contentRows'> {
  const measured: MeasuredItem[] = []
  let cursor = Math.max(0, leadingRows)
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index]
    if (!item) {
      continue
    }
    const previous = index > 0 ? items[index - 1] ?? null : null
    const spacer = shouldInsertSpacer(previous, item) ? 1 : 0
    const itemRows = Math.max(1, estimateItemRows(item, width))
    measured.push({
      item,
      index,
      start: cursor,
      contentStart: cursor + spacer,
      contentEnd: cursor + spacer + itemRows,
      end: cursor + spacer + itemRows
    })
    cursor += spacer + itemRows
  }
  const totalRows = cursor
  const maxOffset = Math.max(0, totalRows - rows)
  const offset = clamp(scrollOffset, 0, maxOffset)
  const targetBottom = totalRows - offset
  const targetTop = Math.max(0, targetBottom - rows)
  const startIndex = Math.max(0, measured.findIndex((entry) => entry.end > targetTop))
  const endIndex = measured.findIndex((entry) => entry.start >= targetBottom)
  const sliceEnd = endIndex === -1 ? measured.length : endIndex
  return {
    items: measured.slice(startIndex, sliceEnd).map((entry) => {
      const visibleStart = Math.max(entry.contentStart, targetTop)
      const visibleEnd = Math.min(entry.contentEnd, targetBottom)
      return {
        item: entry.item,
        rows: Math.max(1, visibleEnd - visibleStart),
        startRow: Math.max(0, visibleStart - entry.contentStart)
      }
    }),
    startIndex,
    hiddenAbove: targetTop > 0,
    hiddenBelow: targetBottom < totalRows,
    maxOffset,
    leadingVisible: leadingRows > 0 && targetTop < leadingRows && targetBottom > 0
  }
}

type RenderKind =
  | 'user'
  | 'assistant'
  | 'tool-group'
  | 'tool-call'
  | 'status'
  | 'footer'
  | 'other'

const SPACED_KINDS: ReadonlySet<RenderKind> = new Set([
  'user',
  'assistant',
  'tool-group',
  'tool-call',
  'status',
  'footer'
])

function renderKind(item: ChatItem): RenderKind | null {
  switch (item.kind) {
    case 'user':
      return 'user'
    case 'assistant':
      return stripToolBlocks(item.text).trim() ? 'assistant' : null
    case 'tool-call':
      return item.coalesce ? null : 'tool-call'
    case 'tool-group':
      return 'tool-group'
    case 'tool-result':
      return 'other'
    case 'note':
      if (isTurnFooterText(item.text) || item.text.startsWith('●') || item.text.startsWith('*')) {
        return 'footer'
      }
      if (isInterruptStatusText(item.text)) {
        return 'status'
      }
      return 'other'
  }
}

function estimateItemRows(item: ChatItem, width: number): number {
  const textWidth = Math.max(10, width - 4)
  switch (item.kind) {
    case 'user':
      return estimateTextRows(item.text, textWidth)
    case 'assistant':
      return estimateTextRows(stripToolBlocks(item.text).trim(), textWidth)
    case 'tool-call': {
      if (item.coalesce) {
        return item.summary ? 3 : 0
      }
      const resultLines = item.result?.text ? Math.min(9, item.result.text.split('\n').length) : 0
      return 1 + resultLines
    }
    case 'tool-group':
      return 1 + item.group.entries.length
    case 'tool-result':
      return estimateTextRows(item.text, textWidth)
    case 'note':
      return estimateTextRows(item.text, textWidth)
  }
}

function limitItemRows(item: ChatItem, maxRows: number, width: number, startRow: number): ChatItem {
  const textWidth = Math.max(10, width - 4)
  switch (item.kind) {
    case 'assistant': {
      const cleaned = stripToolBlocks(item.text).trim()
      const text = sliceTextRows(cleaned, startRow, maxRows, textWidth)
      return text === cleaned ? item : { ...item, text }
    }
    case 'user': {
      const text = sliceTextRows(item.text, startRow, maxRows, textWidth)
      return text === item.text ? item : { ...item, text }
    }
    case 'note': {
      const text = sliceTextRows(item.text, startRow, maxRows, textWidth)
      return text === item.text ? item : { ...item, text }
    }
    case 'tool-result': {
      const text = sliceTextRows(item.text, startRow, maxRows, textWidth)
      return text === item.text ? item : { ...item, text }
    }
    case 'tool-group': {
      const availableEntries = Math.max(1, maxRows - 1)
      if (item.group.entries.length <= availableEntries) {
        return item
      }
      const entries = startRow > 0
        ? item.group.entries.slice(-availableEntries)
        : item.group.entries.slice(0, availableEntries)
      return {
        ...item,
        group: {
          ...item.group,
          entries
        }
      }
    }
    case 'tool-call': {
      if (!item.result?.text) {
        return item
      }
      const text = sliceTextRows(item.result.text, startRow, Math.max(1, maxRows - 1), textWidth)
      return text === item.result.text
        ? item
        : {
            ...item,
            result: {
              ...item.result,
              text
            }
          }
    }
  }
}

function sliceTextRows(text: string, startRow: number, maxRows: number, width: number): string {
  const totalRows = estimateTextRows(text, width)
  if (startRow <= 0 && totalRows <= maxRows) {
    return text
  }

  const normalizedStart = clamp(startRow, 0, Math.max(0, totalRows - 1))
  const normalizedEnd = clamp(normalizedStart + maxRows, normalizedStart + 1, totalRows)
  const prefixRows = normalizedStart > 0 ? 1 : 0
  const suffixRows = normalizedEnd < totalRows ? 1 : 0
  const rowsForContent = Math.max(1, maxRows - prefixRows - suffixRows)
  const lines = text.split('\n')
  const selected: string[] = []

  if (prefixRows && !suffixRows) {
    let used = 0
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      const line = lines[index] ?? ''
      const rows = estimateLineRows(line, width)
      if (used > 0 && used + rows > rowsForContent) {
        break
      }
      const remaining = Math.max(1, rowsForContent - used)
      selected.unshift(limitLineToRows(line, remaining, width, true))
      used += Math.min(rows, remaining)
      if (used >= rowsForContent) {
        break
      }
    }
    return ['... earlier content hidden ...', ...selected].join('\n')
  }

  let cursor = 0
  let used = 0
  for (const line of lines) {
    const rows = estimateLineRows(line, width)
    const lineStart = cursor
    const lineEnd = cursor + rows
    cursor = lineEnd
    if (lineEnd <= normalizedStart) {
      continue
    }
    if (lineStart >= normalizedEnd || used >= rowsForContent) {
      break
    }
    const remaining = Math.max(1, rowsForContent - used)
    selected.push(limitLineToRows(line, remaining, width, false))
    used += Math.min(rows, remaining)
  }

  return [
    ...(prefixRows ? ['... earlier content hidden ...'] : []),
    ...selected,
    ...(suffixRows ? ['... newer content hidden ...'] : [])
  ].join('\n')
}

function estimateTextRows(text: string, width: number): number {
  if (!text) {
    return 1
  }
  return text
    .split('\n')
    .map((line) => estimateLineRows(line, width))
    .reduce((total, rows) => total + rows, 0)
}

function estimateLineRows(line: string, width: number): number {
  return Math.max(1, Math.ceil(stringWidth(line || ' ') / Math.max(1, width)))
}

function limitLineToRows(line: string, maxRows: number, width: number, preferTail: boolean): string {
  const limit = Math.max(8, width * maxRows)
  if (stringWidth(line) <= limit) {
    return line
  }
  return preferTail ? `...${line.slice(-Math.max(1, limit - 3))}` : `${line.slice(0, limit - 3)}...`
}

function normaliseRows(rows: number | undefined): number | null {
  if (rows === undefined) {
    return null
  }
  if (!Number.isFinite(rows)) {
    return null
  }
  return Math.max(0, Math.floor(rows))
}

function terminalWidth(): number {
  const columns = process.stdout?.columns
  return typeof columns === 'number' && columns > 0 ? columns : DEFAULT_WIDTH
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function shouldInsertSpacer(previous: ChatItem | null, current: ChatItem): boolean {
  const prevKind = previous ? renderKind(previous) : null
  const nextKind = renderKind(current)
  if (!prevKind || !nextKind) {
    return false
  }
  if (!SPACED_KINDS.has(prevKind) || !SPACED_KINDS.has(nextKind)) {
    return false
  }
  if (prevKind === 'status' && nextKind === 'footer') {
    return false
  }
  if (prevKind === 'tool-call' && nextKind === 'tool-call') {
    return false
  }
  return true
}

function isInterruptStatusText(text: string): boolean {
  return text.trim().toLowerCase() === 'interrupt'
}
