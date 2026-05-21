// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { DocsView } from './DocsView'

const index = {
  root: '/workspace/Aether/docs',
  default_path: 'README.md',
  documents: [
    { path: 'README.md', title: 'Aether Docs', size_bytes: 1200, updated_at: 1 },
    { path: 'sprint-20/00_overview.md', title: 'Sprint 20', size_bytes: 2400, updated_at: 2 },
  ],
}

const readme = {
  path: 'README.md',
  title: 'Aether Docs',
  content: '# Aether Docs\n\nStart here.',
  size_bytes: 1200,
  updated_at: 1,
}

const sprint = {
  path: 'sprint-20/00_overview.md',
  title: 'Sprint 20',
  content: '# Sprint 20\n\n| Area | Status |\n|---|---|\n| Web | Done |',
  size_bytes: 2400,
  updated_at: 2,
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('DocsView', () => {
  it('loads the docs index and renders selected markdown content', async () => {
    vi.spyOn(api, 'docs').mockResolvedValue(index)
    const doc = vi.spyOn(api, 'doc').mockImplementation(async (path: string) => (path === sprint.path ? sprint : readme))

    render(<DocsView />)

    expect((await screen.findAllByText('Aether Docs')).length).toBeGreaterThan(1)
    expect(screen.getByText('Start here.')).toBeTruthy()

    fireEvent.click(screen.getByText('Sprint 20'))

    await waitFor(() => expect(doc).toHaveBeenCalledWith('sprint-20/00_overview.md'))
    expect(await screen.findByText('Web')).toBeTruthy()
    expect(screen.getByText('Done')).toBeTruthy()
  })
})
