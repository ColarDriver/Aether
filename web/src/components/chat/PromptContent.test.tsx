// @vitest-environment jsdom

import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ApprovalContent, PermissionPreviewContent } from './PromptContent'

describe('PromptContent', () => {
  it('renders permission preview command, diff, and arguments', () => {
    render(
      <PermissionPreviewContent
        args={{ path: 'app.py' }}
        preview={{ command: 'python app.py', diff: '@@ -1 +1 @@\n-old\n+new' }}
        reason="Needs write access"
      />,
    )

    expect(screen.getByText('Needs write access')).toBeTruthy()
    expect(screen.getByText('python app.py')).toBeTruthy()
    expect(screen.getByRole('table', { name: 'Code diff' })).toBeTruthy()
    expect(screen.getAllByText(/app.py/).length).toBeGreaterThan(0)
  })

  it('renders approval questions and submits answers', () => {
    const onSubmit = vi.fn()
    render(
      <ApprovalContent
        approvalKind="questions"
        questions={[{
          id: 'mode',
          question: 'Which mode?',
          options: [{ label: 'Fast' }, { label: 'Careful' }],
        }]}
        onSubmitAnswers={onSubmit}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Careful' }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit answers' }))

    expect(onSubmit).toHaveBeenCalledWith({ mode: 'Careful' })
  })
})
