import { Box, Text, useInput } from 'ink'
import { useEffect, useMemo, useState, type ReactElement } from 'react'

import { theme } from '../lib/theme.js'
import { overlayActions, type OverlayState } from '../store/overlayStore.js'

export interface PickerItem<T> {
  /** Stable identifier — used as React key and for `initialId` matching. */
  id: string
  /** The underlying value handed to `onSelect`. */
  value: T
  /**
   * Optional text used by the `/`-filter to match against. Defaults to the
   * primary label rendered by `renderRow` — but rows often render rich
   * markup so the caller can supply a flat string here for the matcher.
   */
  searchKey?: string
}

export interface PickerPayload<T> {
  title: string
  items: PickerItem<T>[]
  initialId?: string
  /** Enable `/` filtering. Off by default to match the Python picker. */
  filterable?: boolean
  /**
   * When set, the row whose `id` matches `currentId` gets a check-mark
   * appended by the picker chrome (Python TUI's `→ kimi-k2.6 ✓` pattern).
   * The cursor still defaults to the current selection unless `initialId`
   * overrides it.
   */
  currentId?: string
  /**
   * Render one row. `focused` is true for the currently highlighted item,
   * `current` marks the already-selected value.
   * The component is responsible for layout (icon, columns, dim metadata)
   * — the picker chrome only paints the title, count, pager hints and the
   * footer keymap.
   */
  renderRow(item: PickerItem<T>, focused: boolean, current: boolean): ReactElement
  /**
   * Called when the user hits Enter. Return a promise to keep the picker
   * mounted while the action resolves (e.g. RPC call); the picker shows a
   * "resolving…" indicator until the promise settles. Reject to display an
   * inline error and let the user retry without losing the picker state.
   */
  onSelect(item: PickerItem<T>): void | Promise<void>
  /** Optional cleanup hook on ESC / dismissAll. */
  onCancel?(): void
  /**
   * Verb used in the "resolving…" indicator while `onSelect` settles
   * (e.g. "resuming", "switching model"). Defaults to "resolving".
   */
  pendingVerb?: string
  /** "no items" hint copy. Defaults to "No entries available.". */
  emptyMessage?: string
}

const PAGE_SIZE = 10

export function Picker<T>({
  overlay
}: {
  overlay: OverlayState<PickerPayload<T>>
}): ReactElement {
  const {
    title,
    items,
    initialId,
    currentId,
    renderRow,
    onSelect,
    pendingVerb,
    emptyMessage,
    filterable = false
  } =
    overlay.payload
  const effectiveInitialId = initialId ?? currentId

  const [filter, setFilter] = useState('')
  const [filterMode, setFilterMode] = useState(false)

  const filtered = useMemo(() => {
    if (!filter) {
      return items
    }
    const needle = filter.toLowerCase()
    return items.filter((item) => {
      const haystack = (item.searchKey ?? item.id).toLowerCase()
      return haystack.includes(needle)
    })
  }, [items, filter])

  const initialIndex = useMemo(() => {
    if (!effectiveInitialId) {
      return 0
    }
    const idx = filtered.findIndex((entry) => entry.id === effectiveInitialId)
    return idx === -1 ? 0 : idx
  }, [filtered, effectiveInitialId])

  const [cursor, setCursor] = useState(initialIndex)
  const [pageStart, setPageStart] = useState(0)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Clamp the cursor whenever the filtered set changes.
    setCursor((value) => Math.min(value, Math.max(0, filtered.length - 1)))
  }, [filtered.length])

  useEffect(() => {
    if (cursor < pageStart) {
      setPageStart(cursor)
      return
    }
    if (cursor >= pageStart + PAGE_SIZE) {
      setPageStart(cursor - PAGE_SIZE + 1)
    }
  }, [cursor, pageStart])

  useInput(
    (input, key) => {
      if (pending) {
        return
      }
      if (filterMode) {
        if (key.escape || key.return) {
          setFilterMode(false)
          return
        }
        if (key.backspace || key.delete) {
          setFilter((value) => value.slice(0, -1))
          return
        }
        if (input && !key.ctrl && !key.meta) {
          setFilter((value) => value + input)
          return
        }
        return
      }
      if (key.escape) {
        if (filter) {
          setFilter('')
          return
        }
        overlayActions.dismiss(overlay.id, 'cancel')
        return
      }
      if (filtered.length === 0) {
        if (filterable && input === '/') {
          setFilterMode(true)
        }
        return
      }
      if (key.upArrow) {
        setCursor((value) => (value <= 0 ? 0 : value - 1))
        return
      }
      if (key.downArrow) {
        setCursor((value) => Math.min(filtered.length - 1, value + 1))
        return
      }
      if (key.pageUp) {
        setCursor((value) => Math.max(0, value - PAGE_SIZE))
        return
      }
      if (key.pageDown) {
        setCursor((value) => Math.min(filtered.length - 1, value + PAGE_SIZE))
        return
      }
      if (input === 'g') {
        setCursor(0)
        return
      }
      if (input === 'G') {
        setCursor(filtered.length - 1)
        return
      }
      if (filterable && input === '/') {
        setFilterMode(true)
        return
      }
      if (key.return) {
        const item = filtered[cursor]
        if (!item) {
          return
        }
        const result = onSelect(item)
        if (result && typeof (result as Promise<void>).then === 'function') {
          setPending(true)
          ;(result as Promise<void>)
            .then(() => {
              overlayActions.dismiss(overlay.id, 'commit', { id: item.id })
            })
            .catch((err: unknown) => {
              const message = err instanceof Error ? err.message : String(err)
              setError(message)
              setPending(false)
            })
        } else {
          overlayActions.dismiss(overlay.id, 'commit', { id: item.id })
        }
      }
    },
    { isActive: true }
  )

  const border = theme.color('accent')
  const brand = theme.colorProps('brand')
  const accent = theme.colorProps('accent')
  const dim = theme.colorProps('dim')
  const borderColor = theme.color('border')
  const cursorBg = theme.color('brand')
  const cursorFg = '#FFFFFF'
  const arrow = theme.icon('arrow') || '→'
  const logo = theme.icon('logo') || '*'
  const success = theme.icon('success') || '✓'

  if (items.length === 0) {
    return (
      <Box
        flexDirection="column"
        width="100%"
        borderStyle="single"
        {...(borderColor ? { borderColor } : {})}
        paddingX={1}
      >
        <Text bold {...brand}>
          {logo} {title}
        </Text>
        <Box marginTop={1}>
          <Text>{emptyMessage ?? 'No entries available.'}</Text>
        </Box>
        <Box marginTop={1}>
          <Text {...dim}>
            <Text {...accent}>Esc</Text> cancel
          </Text>
        </Box>
      </Box>
    )
  }

  const visible = filtered.slice(pageStart, pageStart + PAGE_SIZE)
  const hasAbove = pageStart > 0
  const hasBelow = pageStart + PAGE_SIZE < filtered.length
  const verb = pendingVerb ?? 'resolving'

  return (
    <Box
      flexDirection="column"
      width="100%"
      borderStyle="single"
      {...(borderColor ? { borderColor } : {})}
      paddingX={1}
    >
      <Text bold {...brand}>
        {logo} {title}
      </Text>

      <Box marginTop={1} flexDirection="column">
        <Text {...dim} italic>
          {hasAbove ? `   ${arrow} more above` : ' '}
        </Text>
        {visible.length === 0 ? (
          <Text {...dim}>no matches — backspace or ESC to clear</Text>
        ) : (
          visible.map((item, idx) => {
            const realIdx = pageStart + idx
            const focused = realIdx === cursor
            const isCurrent = currentId !== undefined && item.id === currentId
            return (
              <Box key={item.id} width="100%">
                <Box width={3} flexShrink={0}>
                  {focused ? (
                    <Text
                      {...(cursorBg ? { backgroundColor: cursorBg } : {})}
                      {...(cursorFg ? { color: cursorFg } : {})}
                    >
                      {' '}{arrow}{' '}
                    </Text>
                  ) : (
                    <Text>{'   '}</Text>
                  )}
                </Box>
                <Box flexGrow={1}>
                  {renderRow(item, focused, isCurrent)}
                </Box>
                {isCurrent ? (
                  <Text {...accent}>{'  ' + success}</Text>
                ) : null}
              </Box>
            )
          })
        )}
        <Text {...dim} italic>
          {hasBelow ? `   ${arrow} more below` : ' '}
        </Text>
      </Box>

      <Box marginTop={1} flexDirection="column">
        {pending ? (
          <Text color="yellow">
            {verb}…
          </Text>
        ) : null}
        {error ? <Text color="red">error: {error}</Text> : null}
        <Text {...dim}>
          {'  '}
          <Text {...accent}>↑↓</Text> navigate
          {'  ·  '}
          <Text {...accent}>Enter</Text> select
          {'  ·  '}
          <Text {...accent}>Esc</Text> cancel
          {filterable ? (
            <>
              {'  ·  '}
              <Text {...accent}>/</Text> filter
              {filter || filterMode ? (
                <>
                  {' · /'}
                  {filter}
                  {filterMode ? <Text {...accent}>_</Text> : null}
                </>
              ) : null}
            </>
          ) : null}
        </Text>
      </Box>
    </Box>
  )
}
