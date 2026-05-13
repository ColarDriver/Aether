import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import { theme } from '../lib/theme.js'
import { type ToolGroupRecord } from '../store/toolGroupStore.js'

export function ExploredTree({ group }: { group: ToolGroupRecord }): ReactElement {
  // Mirrors Python `ToolGroup.render_explore_tree`:
  //   ● Explored
  //     ⎿ Read file.py
  //       List dir/
  //       Search "pattern" in dir/
  // No headline count summary — the per-entry tree already shows what ran.
  const brand = theme.colorProps('brand')
  const warning = theme.colorProps('warning')
  return (
    <Box marginTop={1} flexDirection="column">
      <Box>
        <Text bold {...brand}>
          {theme.icon('assistant')}{' '}
        </Text>
        <Text bold>Explored</Text>
        {group.hasError ? (
          <>
            <Text bold {...warning}>
              {'  '}
              {theme.icon('warn')}{' '}
            </Text>
            <Text {...warning}>with errors</Text>
          </>
        ) : null}
      </Box>
      <Box flexDirection="column" marginLeft={2}>
        {group.entries.map((entry, idx) => (
          <Box key={`${entry.toolCallId}_${idx}`}>
            <Text dimColor>{idx === 0 ? '⎿ ' : '  '}</Text>
            <Text bold>{entry.verb}</Text>
            {entry.detail ? (
              <>
                <Text> </Text>
                <Text dimColor>{entry.detail}</Text>
              </>
            ) : null}
            {entry.isError ? (
              <>
                <Text> </Text>
                <Text color="red" dimColor>
                  (failed)
                </Text>
              </>
            ) : null}
          </Box>
        ))}
      </Box>
    </Box>
  )
}
