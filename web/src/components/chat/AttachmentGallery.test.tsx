// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { AttachmentGallery } from './AttachmentGallery'

afterEach(cleanup)

describe('AttachmentGallery', () => {
  it('renders file chips and image thumbnails', () => {
    render(
      <AttachmentGallery
        attachments={[
          { type: 'file', name: 'app.ts', path: 'src/app.ts', lineStart: 4, lineEnd: 8 },
          { type: 'image', name: 'chart.png', data: 'data:image/png;base64,abc' },
        ]}
        align="end"
      />,
    )

    expect(screen.getByText('app.ts')).toBeTruthy()
    expect(screen.getByText('L4-L8')).toBeTruthy()
    expect(screen.getByRole('img', { name: 'chart.png' })).toBeTruthy()
    expect(document.querySelector('.attachment-gallery-end')).toBeTruthy()
  })

  it('opens and closes the image preview dialog', () => {
    render(<AttachmentGallery attachments={[{ type: 'image', name: 'plot.png', data: 'data:image/png;base64,abc' }]} />)

    fireEvent.click(screen.getByRole('button', { name: /plot.png/ }))
    expect(screen.getByRole('dialog', { name: 'plot.png' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Close image preview' }))
    expect(screen.queryByRole('dialog', { name: 'plot.png' })).toBeNull()
  })
})
