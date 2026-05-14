import { Box, Text, useInput } from 'ink'
import type { ReactElement } from 'react'

import type { JsonObject } from '../gatewayTypes.js'
import { overlayActions, type OverlayState } from '../store/overlayStore.js'

/**
 * Transitional modal used while the real Approval/Permission components are
 * unavailable. It surfaces the raw method + params so a developer can see a
 * reverse RPC arrived, and it lets Y / N short-circuit the answer.
 *
 * Y commits an empty `{ confirmed: true }` for approval requests and
 * `{ type: 'allow_once' }` for permission requests; N (and ESC, handled in
 * OverlayFrame) sends the conservative deny payload from defaultDeny().
 */
export function PlaceholderModal({ overlay }: { overlay: OverlayState }): ReactElement {
  useInput(
    (input, key) => {
      if (key.escape) {
        overlayActions.dismiss(overlay.id, 'cancel')
        return
      }
      if (input === 'y' || input === 'Y') {
        overlayActions.dismiss(overlay.id, 'commit', commitPayload(overlay))
        return
      }
      if (input === 'n' || input === 'N') {
        overlayActions.dismiss(overlay.id, 'cancel')
        return
      }
    },
    { isActive: true }
  )

  const summary = summarisePayload(overlay.payload)
  const title = labelForKind(overlay)

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={1}>
      <Box>
        <Text bold color="yellow">
          {title}
        </Text>
        <Text dimColor> · {overlay.id}</Text>
      </Box>
      <Box marginTop={1}>
        <Text>{summary}</Text>
      </Box>
      <Box marginTop={1}>
        <Text dimColor>(placeholder UI) </Text>
        <Text>
          [<Text color="green">Y</Text>] approve · [<Text color="red">N</Text>] deny · ESC
          deny
        </Text>
      </Box>
    </Box>
  )
}

function labelForKind(overlay: OverlayState): string {
  switch (overlay.kind) {
    case 'approval':
      return 'Approval requested'
    case 'permission':
      return 'Tool permission requested'
    default:
      return `Overlay · ${overlay.kind}`
  }
}

function commitPayload(overlay: OverlayState): JsonObject {
  if (overlay.kind === 'approval') {
    return { confirmed: true, answers: [] }
  }
  if (overlay.kind === 'permission') {
    return { type: 'allow_once' }
  }
  return {}
}

function summarisePayload(payload: unknown): string {
  if (!payload || typeof payload !== 'object') {
    return '(no payload)'
  }
  const raw = JSON.stringify(payload, null, 2)
  if (raw.length <= 600) {
    return raw
  }
  return `${raw.slice(0, 597)}...`
}
