import { Box } from 'ink'
import { useStore } from '@nanostores/react'
import type { ReactElement } from 'react'

import type { ApprovalRequestParams, PermissionRequestParams } from '../gatewayTypes.js'
import { overlayStack, type OverlayState } from '../store/overlayStore.js'

import { ApprovalModal } from './ApprovalModal.js'
import { HelpOverlay, type HelpOverlayPayload } from './HelpOverlay.js'
import { PermissionModal } from './PermissionModal.js'
import { Picker, type PickerPayload } from './Picker.js'
import { PlaceholderModal } from './PlaceholderModal.js'
import { SessionPicker, type SessionPickerPayload } from './SessionPicker.js'

/**
 * Single global overlay renderer. Only the top-of-stack overlay is mounted;
 * lower overlays remain in the store but receive no keyboard input. Each
 * overlay component owns its own `useInput` (active when it renders) which
 * keeps the focus story trivial: there is at most one active key consumer
 * besides the composer at any time.
 *
 * The composer must observe `overlayStack` (or `hasOverlay()`) and pass
 * `isActive: false` to its own `useInput` so two consumers never compete.
 */
export function OverlayFrame(): ReactElement | null {
  const stack = useStore(overlayStack)
  const top = stack[stack.length - 1] ?? null
  if (!top) {
    return null
  }
  return (
    <Box flexDirection="column" marginTop={1}>
      {renderOverlay(top)}
    </Box>
  )
}

function renderOverlay(overlay: OverlayState): ReactElement {
  switch (overlay.kind) {
    case 'approval':
      return <ApprovalModal overlay={overlay as OverlayState<ApprovalRequestParams>} />
    case 'permission':
      return <PermissionModal overlay={overlay as OverlayState<PermissionRequestParams>} />
    case 'session-picker':
      return <SessionPicker overlay={overlay as OverlayState<SessionPickerPayload>} />
    case 'picker':
      return <Picker overlay={overlay as OverlayState<PickerPayload<unknown>>} />
    case 'help':
      return <HelpOverlay overlay={overlay as OverlayState<HelpOverlayPayload>} />
    case 'placeholder':
      return <PlaceholderModal overlay={overlay} />
  }
}
