import { describe, expect, it } from 'vitest'
import { render } from 'ink-testing-library'

import { EditSummary } from '../components/EditSummary.js'
import type { ToolCallSummary } from '../store/chatStore.js'

const DIFF = [
  '--- src/foo.ts',
  '+++ src/foo.ts',
  '@@ -1,2 +1,3 @@',
  ' const a = 1',
  '+const b = 2',
  ' const c = 3'
].join('\n')

function summary(overrides: Partial<ToolCallSummary> = {}): ToolCallSummary {
  return {
    path: 'src/foo.ts',
    linesAdded: 1,
    linesRemoved: 0,
    diff: DIFF,
    ...overrides
  }
}

describe('EditSummary — post-execution', () => {
  it('renders the header + counts subline + no diff when folded', () => {
    const { lastFrame, unmount } = render(
      <EditSummary
        summary={summary()}
        toolName="file_edit"
        expanded={false}
        focused={false}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Update')
    expect(frame).toContain('src/foo.ts')
    expect(frame).toContain('Added 1 line')
    // Folded: the diff body should not be present.
    expect(frame).not.toContain('const b = 2')
    unmount()
  })

  it('shows the diff body when expanded={true}', () => {
    const { lastFrame, unmount } = render(
      <EditSummary
        summary={summary()}
        toolName="file_edit"
        expanded={true}
        focused={false}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('const b = 2')
    unmount()
  })

  it("uses 'Create' for write_file with zero removed lines", () => {
    const { lastFrame, unmount } = render(
      <EditSummary
        summary={summary({ linesRemoved: 0 })}
        toolName="write_file"
        expanded={false}
        focused={false}
      />
    )
    expect(lastFrame()).toContain('Create')
    unmount()
  })
})

describe('EditSummary — previewStatus', () => {
  it("'pending' forces the diff visible and tags the subline", () => {
    const { lastFrame, unmount } = render(
      <EditSummary
        summary={summary()}
        toolName="file_edit"
        expanded={false}
        focused={false}
        previewStatus="pending"
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Will add 1 line')
    expect(frame).toContain('(pending approval)')
    // Diff visible even though `expanded={false}`.
    expect(frame).toContain('const b = 2')
    unmount()
  })

  it("'rejected' tags the subline and keeps the diff visible", () => {
    const { lastFrame, unmount } = render(
      <EditSummary
        summary={summary()}
        toolName="file_edit"
        expanded={false}
        focused={false}
        previewStatus="rejected"
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Added 1 line')
    expect(frame).toContain('(rejected)')
    expect(frame).toContain('const b = 2')
    unmount()
  })

  it('no-op row hides the diff regardless of previewStatus', () => {
    const { lastFrame, unmount } = render(
      <EditSummary
        summary={summary({ noOp: true })}
        toolName="file_edit"
        expanded={true}
        focused={false}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('(no-op)')
    expect(frame).not.toContain('const b = 2')
    unmount()
  })
})
