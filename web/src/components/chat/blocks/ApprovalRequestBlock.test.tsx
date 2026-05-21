// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ApprovalRequestBlock as ApprovalRequest } from '../../../chat-rendering'
import { ApprovalRequestBlock } from './ApprovalRequestBlock'

const base = {
  id: 'approval-1',
  sessionId: 'session-1',
  runId: 'run-1',
  timestamp: 1,
  source: 'live' as const,
  kind: 'approval_request' as const,
  promptId: 'approval-1',
  state: 'pending' as const,
}

afterEach(cleanup)

describe('ApprovalRequestBlock', () => {
  it('submits structured question answers by stable question id', () => {
    const onRespond = vi.fn()
    const block: ApprovalRequest = {
      ...base,
      approvalKind: 'questions',
      questions: [
        {
          id: 'mode',
          header: 'Mode',
          question: 'Which mode?',
          options: [
            { id: 'fast', label: 'Fast', description: 'Less detail' },
            { id: 'careful', label: 'Careful', description: 'More detail' },
          ],
        },
      ],
    }

    render(<ApprovalRequestBlock block={block} onRespond={onRespond} />)

    fireEvent.click(screen.getByRole('button', { name: /Careful/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit answers' }))

    expect(onRespond).toHaveBeenCalledWith({
      confirmed: true,
      answers: { mode: 'Careful' },
    })
  })

  it('supports free-text question answers', () => {
    const onRespond = vi.fn()
    const block: ApprovalRequest = {
      ...base,
      approvalKind: 'questions',
      questions: [{ id: 'name', question: 'Project name?' }],
    }

    render(<ApprovalRequestBlock block={block} onRespond={onRespond} />)

    fireEvent.change(screen.getByLabelText('Answer 1'), { target: { value: 'Aether' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit answers' }))

    expect(onRespond).toHaveBeenCalledWith({
      confirmed: true,
      answers: { name: 'Aether' },
    })
  })
})
