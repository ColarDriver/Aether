// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { ToolsView } from './ToolsView'

const groups = [
  {
    name: 'filesystem',
    tools: [
      {
        name: 'read_file',
        description: 'Read a file from disk',
        enabled: true,
        required: ['path'],
        parameters: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'File path to read' },
            offset: { type: 'integer' },
          },
        },
      },
    ],
  },
  {
    name: 'web',
    tools: [
      {
        name: 'web_search',
        description: 'Search the web',
        enabled: true,
        required: ['query'],
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
          },
        },
      },
    ],
  },
]

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ToolsView', () => {
  it('renders searchable tool details and parameter schema', async () => {
    vi.spyOn(api, 'toolGroups').mockResolvedValue({ groups })

    render(<ToolsView />)

    expect((await screen.findAllByText('read_file')).length).toBeGreaterThan(1)
    expect(screen.getByText('File path to read')).toBeTruthy()
    expect(screen.getByText('yes')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('Search tools'), { target: { value: 'web' } })
    expect((screen.getAllByText('web_search')).length).toBeGreaterThan(1)
    expect(screen.queryByText('read_file')).toBeNull()

    fireEvent.click(screen.getAllByText('web_search')[0])
    expect(screen.getByText('Search query')).toBeTruthy()
  })
})
