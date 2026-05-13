import { Box, Text, useInput } from 'ink'
import { useMemo, useState, type ReactElement } from 'react'

import { theme } from '../lib/theme.js'
import type { SlashCommandInfo } from '../gatewayTypes.js'
import { overlayActions, type OverlayState } from '../store/overlayStore.js'

export interface HelpOverlayPayload {
  catalog: SlashCommandInfo[]
}

const PAGE_SIZE = 14

export function HelpOverlay({
  overlay
}: {
  overlay: OverlayState<HelpOverlayPayload>
}): ReactElement {
  const [scroll, setScroll] = useState(0)

  // Mirror Python `_cmd_help`: a flat alphabetical table of every command
  // with its description. No category grouping — that was a TS-only
  // embellishment that drifted from Python's UX.
  const rows = useMemo(
    () => [...overlay.payload.catalog].sort((a, b) => a.name.localeCompare(b.name)),
    [overlay.payload.catalog]
  )

  useInput((input, key) => {
    if (key.escape || input === 'q' || input === '?') {
      overlayActions.dismiss(overlay.id, 'cancel')
      return
    }
    if (key.upArrow) {
      setScroll((value) => Math.max(0, value - 1))
      return
    }
    if (key.downArrow) {
      setScroll((value) => Math.min(Math.max(0, rows.length - PAGE_SIZE), value + 1))
      return
    }
    if (key.pageUp) {
      setScroll((value) => Math.max(0, value - PAGE_SIZE))
      return
    }
    if (key.pageDown) {
      setScroll((value) => Math.min(Math.max(0, rows.length - PAGE_SIZE), value + PAGE_SIZE))
    }
  })

  const visible = rows.slice(scroll, scroll + PAGE_SIZE)
  const more = rows.length > scroll + PAGE_SIZE
  const accent = theme.colorProps('accent')
  const dim = theme.colorProps('dim')
  const brand = theme.colorProps('brand')
  const nameWidth = Math.max(...rows.map((r) => r.name.length), 8)
  const border = theme.color('border')

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      {...(border ? { borderColor: border } : {})}
      paddingX={1}
    >
      <Box>
        <Text bold {...brand}>
          Slash commands
        </Text>
        <Text {...dim}>
          {' · '}
          {rows.length} command{rows.length === 1 ? '' : 's'}
        </Text>
      </Box>
      <Box flexDirection="column" marginTop={1}>
        {visible.map((entry) => (
          <Box key={entry.name}>
            <Text bold {...accent}>
              {entry.name.padEnd(nameWidth)}
            </Text>
            <Text {...dim}>{'  ' + entry.description}</Text>
          </Box>
        ))}
      </Box>
      <Box marginTop={1}>
        <Text {...dim}>
          ↑/↓ scroll · PgUp/PgDn page · ESC close{more ? ' · more below' : ''}
        </Text>
      </Box>
    </Box>
  )
}
