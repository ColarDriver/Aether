// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApprovalDialog } from './ApprovalDialog'

const prompt = {
  promptId: 'prompt-1',
  kind: 'plan',
  sessionId: 'session-1',
  runId: 'run-1',
  planPath: '/tmp/plan.md',
  planText: '# Plan\n\n| Step | Action |\n| --- | --- |\n| 1 | Inspect |',
  questions: [],
}

afterEach(cleanup)

describe('ApprovalDialog', () => {
  it('renders plan markdown and path', () => {
    render(<ApprovalDialog prompt={prompt} onApprove={() => undefined} onReject={() => undefined} />)

    expect(screen.getByRole('dialog', { name: 'Approval request' })).toBeTruthy()
    expect(screen.getByText('/tmp/plan.md')).toBeTruthy()
    expect(screen.getByRole('table')).toBeTruthy()
    expect(screen.getByRole('cell', { name: 'Inspect' })).toBeTruthy()
  })

  it('emits approve and reject actions', () => {
    const onApprove = vi.fn()
    const onReject = vi.fn()
    render(<ApprovalDialog prompt={prompt} onApprove={onApprove} onReject={onReject} />)

    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reject' }))

    expect(onApprove).toHaveBeenCalledOnce()
    expect(onReject).toHaveBeenCalledOnce()
  })

  it('submits question answers with the approval', () => {
    const onApprove = vi.fn()
    render(
      <ApprovalDialog
        prompt={{
          ...prompt,
          kind: 'questions',
          planText: null,
          questions: [
            {
              id: 'mode',
              prompt: 'Which mode?',
              options: [
                { id: 'fast', label: 'Fast', description: 'Less detail' },
                { id: 'careful', label: 'Careful', description: 'More detail' },
              ],
            },
          ],
        }}
        onApprove={onApprove}
        onReject={() => undefined}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /Careful/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit answers' }))

    expect(onApprove).toHaveBeenCalledWith({ mode: 'Careful' })
  })
})
