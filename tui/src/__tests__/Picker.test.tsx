import { Text } from 'ink'
import { render } from 'ink-testing-library'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { Picker, type PickerItem, type PickerPayload } from '../overlays/Picker.js'
import { OVERLAY_PRIORITY, overlayActions, type OverlayState } from '../store/overlayStore.js'

interface Model {
  id: string
  display: string
}

let nextCreatedAt = 1

function makeItems(ids: string[]): PickerItem<Model>[] {
  return ids.map((id) => ({ id, value: { id, display: `model ${id}` } }))
}

function makeOverlay(
  payload: PickerPayload<Model>
): OverlayState<PickerPayload<Model>> {
  const overlay: OverlayState<PickerPayload<Model>> = {
    kind: 'picker',
    id: `picker_${nextCreatedAt}`,
    payload,
    createdAt: nextCreatedAt++,
    priority: OVERLAY_PRIORITY.picker,
    onDismiss: vi.fn()
  }
  overlayActions.push(overlay)
  return overlay
}

describe('Picker (generic overlay)', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
  })
  afterEach(() => {
    overlayActions.resetForTests()
  })

  it('renders the title, count, and item rows', () => {
    const overlay = makeOverlay({
      title: 'Select model',
      items: makeItems(['gpt-4o', 'gpt-4.1', 'claude-sonnet-4-6']),
      renderRow: (item) => <Text>{item.value.display}</Text>,
      onSelect: vi.fn()
    })
    const { lastFrame, unmount } = render(<Picker overlay={overlay} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Select model')
    expect(frame).toContain('3 entries')
    expect(frame).toContain('model gpt-4o')
    expect(frame).toContain('model claude-sonnet-4-6')
    unmount()
  })

  it('renders the empty-state hint when no items provided', () => {
    const overlay = makeOverlay({
      title: 'Select model',
      items: [],
      renderRow: () => <Text>placeholder</Text>,
      onSelect: vi.fn(),
      emptyMessage: 'No models found.'
    })
    const { lastFrame, unmount } = render(<Picker overlay={overlay} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('No models found')
    expect(frame).toContain('ESC close')
    unmount()
  })

  it('shows the resolving indicator while onSelect is in-flight', async () => {
    let release: () => void = () => {}
    const pending = new Promise<void>((resolve) => {
      release = resolve
    })
    const onSelect = vi.fn(() => pending)
    const overlay = makeOverlay({
      title: 'Pick',
      items: makeItems(['a', 'b']),
      pendingVerb: 'switching',
      renderRow: (item) => <Text>{item.id}</Text>,
      onSelect
    })
    const { lastFrame, stdin, unmount } = render(<Picker overlay={overlay} />)
    stdin.write('\r')
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(lastFrame() ?? '').toContain('switching')
    expect(onSelect).toHaveBeenCalledTimes(1)
    release()
    unmount()
  })

  it('initialId selects the matching row', () => {
    const overlay = makeOverlay({
      title: 'Pick',
      items: makeItems(['gpt-4o', 'gpt-4.1', 'claude-sonnet-4-6']),
      initialId: 'gpt-4.1',
      renderRow: (item, focused) => (
        <Text>
          {focused ? '*' : ' '}
          {item.id}
        </Text>
      ),
      onSelect: vi.fn()
    })
    const { lastFrame, unmount } = render(<Picker overlay={overlay} />)
    const frame = lastFrame() ?? ''
    // The focused row uses the `*` prefix from renderRow.
    expect(frame).toContain('*gpt-4.1')
    expect(frame).not.toContain('*gpt-4o')
    unmount()
  })

  it('currentId both highlights the row by default and appends the success glyph', () => {
    const overlay = makeOverlay({
      title: 'Pick',
      items: makeItems(['kimi-k2', 'kimi-k2.5', 'kimi-k2.6', 'kimi-k2-thinking']),
      currentId: 'kimi-k2.6',
      renderRow: (item, focused) => (
        <Text>
          {focused ? '*' : ' '}
          {item.id}
        </Text>
      ),
      onSelect: vi.fn()
    })
    const { lastFrame, unmount } = render(<Picker overlay={overlay} />)
    const frame = lastFrame() ?? ''
    // Cursor jumps to the current row.
    expect(frame).toContain('*kimi-k2.6')
    // The success glyph (✓ in Unicode mode) marks the current row in the
    // gutter — Python TUI's `→ kimi-k2.6 ✓` parity.
    expect(frame).toMatch(/kimi-k2\.6.*[✓v*]/)
    unmount()
  })
})
