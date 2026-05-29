// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { Composer } from './Composer'

const slashCommands = [
  { name: '/help', description: 'Show help', category: 'local' },
  { name: '/plan', description: 'Plan mode', category: 'session' },
]

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('Composer', () => {
  it('applies slash completion instead of sending while popover is open', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/pl', selectionStart: 3 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).not.toHaveBeenCalled()
    expect(textbox.value).toBe('/plan ')
  })

  it('sends normal messages with Enter and preserves Shift+Enter', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: 'hello', selectionStart: 5 } })
    fireEvent.keyDown(textbox, { key: 'Enter', shiftKey: true })
    expect(onSend).not.toHaveBeenCalled()

    fireEvent.keyDown(textbox, { key: 'Enter' })
    expect(onSend).toHaveBeenCalledWith('hello')
  })

  it('applies external draft patches for edit and quote flows', async () => {
    const onSend = vi.fn()
    const attachment = { type: 'file' as const, name: 'notes.md', path: 'notes.md' }
    const { rerender } = render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        slashCommands={slashCommands}
        draftPatch={{ id: 1, mode: 'replace', text: 'old prompt', attachments: [attachment] }}
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    await waitFor(() => expect(textbox.value).toBe('old prompt'))
    expect(screen.getAllByText('notes.md').length).toBeGreaterThanOrEqual(1)

    rerender(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        slashCommands={slashCommands}
        draftPatch={{ id: 2, mode: 'append', text: '> Assistant:\n> answer' }}
      />,
    )

    await waitFor(() => expect(textbox.value).toBe('old prompt\n\n> Assistant:\n> answer'))
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).toHaveBeenCalledWith('old prompt\n\n> Assistant:\n> answer', [attachment])
  })

  it('routes complete slash commands separately from agent prompts', () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        onSlashCommand={onSlashCommand}
        slashCommands={slashCommands}
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/help', selectionStart: 5 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSlashCommand).toHaveBeenCalledWith('/help')
    expect(onSend).not.toHaveBeenCalled()
  })

  it('opens local inspector commands without sending to the agent', async () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    vi.spyOn(api, 'status').mockResolvedValue({ ok: true, name: 'Aether', version: '1.1.0', web: { enabled: true } })
    vi.spyOn(api, 'health').mockResolvedValue({
      status: 'ok',
      runtime: { python_version: '3.12', platform: 'linux', implementation: 'CPython' },
      services: [{ name: 'sessions', available: true, status: 'ok' }],
    })
    vi.spyOn(api, 'currentProvider').mockResolvedValue({
      family: 'openai',
      provider_name: 'codex',
      model: 'gpt-5.4',
      api_key_env_names: [],
      model_env_names: [],
      base_url_env_names: [],
      source: 'test',
    })
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        onSlashCommand={onSlashCommand}
        slashCommands={slashCommands}
        sessionId="session-123456"
        sessionSummary="Test session"
        messageCount={4}
        provider="codex"
        model="gpt-5.4"
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/status', selectionStart: 7 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).not.toHaveBeenCalled()
    expect(onSlashCommand).not.toHaveBeenCalled()
    expect(screen.getByLabelText('Status inspector')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('Runtime')).toBeTruthy())
  })

  it('renders context inspector with provider window and latest compression metadata', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'providerModels').mockResolvedValue({
      models: [{ id: 'gpt-5.4', display_name: 'GPT-5.4', context_window: 128000 }],
      discovery: { kind: 'static', source: 'test' },
    })
    vi.spyOn(api, 'estimateContext').mockResolvedValue({
      session_id: 'session-ctx',
      context_engine: 'default',
      compression_count: 0,
      message_count: 1,
      token_estimate: 1200,
      prompt_tokens: 1200,
      pressure_level: 'low',
      next_action: 'none',
      breakdown: [],
    })
    vi.spyOn(api, 'contextStatus').mockResolvedValue({
      session_id: 'session-ctx',
      context_engine: 'default',
      compression_count: 1,
      last_compression: { status: 'compressed', source_message_count: 42, result_message_count: 18, source_tokens: 90000, result_tokens: 60000 },
      message_count: 18,
      token_estimate: 60000,
      status: 'compressed',
      error: null,
    })
    vi.spyOn(api, 'compressContext').mockResolvedValue({
      session_id: 'session-ctx',
      context_engine: 'default',
      compression_count: 2,
      last_compression: { status: 'skipped', reason: 'not_needed', source_tokens: 60000, result_tokens: 60000 },
      message_count: 18,
      token_estimate: 60000,
      status: 'skipped',
      error: null,
    })
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        slashCommands={slashCommands}
        sessionId="session-ctx"
        provider="codex"
        model="gpt-5.4"
        messageCount={8}
        tokens={{ input_tokens: 1000, output_tokens: 500, total_tokens: 1500 }}
        runMetadata={{
          usage: { input_tokens: 1000, output_tokens: 500, total_tokens: 1500 },
          context_engine: {
            name: 'default',
            compression: {
              status: 'compressed',
              trigger_reason: 'preflight',
              source_message_count: 42,
              result_message_count: 18,
              source_tokens: 90000,
              result_tokens: 60000,
              engine: { tiers_run: ['tier2_snip', 'tier4_collapse'] },
            },
          },
          compaction: { tier2_snipped_count: 2, tier4_collapse_segments: 1 },
          compression_lineage: { generation: 1, trigger_reason: 'preflight' },
        }}
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/context', selectionStart: 8 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).not.toHaveBeenCalled()
    expect(screen.getByLabelText('Context inspector')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('128,000')).toBeTruthy())
    expect(screen.getAllByText('90,000 -> 60,000').length).toBeGreaterThan(0)
    expect(screen.getAllByText('30,000 tokens freed').length).toBeGreaterThan(0)
    expect(screen.getByText('tier2_snip, tier4_collapse')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('optional compression focus'), { target: { value: 'auth' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compress context' }))
    await waitFor(() => expect(api.compressContext).toHaveBeenCalledWith('session-ctx', { focus: 'auth', force: true }))
  })

  it('renders skill inspector data from the skills API', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'skills').mockResolvedValue({
      skills: [
        {
          name: 'web-audit',
          description: 'Audit web UI parity',
          when_to_use: 'When checking web migration gaps',
          source: { source: 'local', path: '/workspace/Aether/skills/web-audit/SKILL.md' },
        },
      ],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/skills', selectionStart: 7 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    await waitFor(() => expect(screen.getByText('web-audit')).toBeTruthy())
    expect(onSend).not.toHaveBeenCalled()
  })

  it('shows service-backed MCP integration status and closes with Escape', async () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    vi.spyOn(api, 'mcpStatus').mockResolvedValue({
      enabled: false,
      status: 'not_configured',
      message: 'No MCP servers are configured for this Aether runtime.',
      servers: [],
      imported_tools: [],
    })
    vi.spyOn(api, 'mcpResources').mockResolvedValue({
      enabled: false,
      status: 'not_configured',
      message: 'No MCP resources are available because no MCP servers are configured.',
      resources: [],
    })
    vi.spyOn(api, 'mcpConfig').mockResolvedValue({
      config_path: '/tmp/aether/mcp_servers.json',
      exists: false,
      servers: [],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} onSlashCommand={onSlashCommand} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/mcp', selectionStart: 4 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(await screen.findByText('No MCP servers are configured for this Aether runtime.')).toBeTruthy()
    expect(screen.queryByText(/MCP management is not wired/)).toBeNull()
    expect(onSend).not.toHaveBeenCalled()
    expect(onSlashCommand).not.toHaveBeenCalled()

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByLabelText('MCP inspector')).toBeNull()
  })

  it('includes local inspector commands in slash completion', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/stat', selectionStart: 5 } })

    expect(screen.getByRole('option', { name: /\/status/ })).toBeTruthy()
  })

  it('opens inspector panels from the composer control menu', () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    vi.spyOn(api, 'mcpStatus').mockResolvedValue({
      enabled: true,
      status: 'available',
      message: '1 MCP server(s) exposed through the tool catalog.',
      servers: [{ name: 'filesystem', status: 'available', tools_count: 2, resources_count: 0, credential_status: 'unknown' }],
      imported_tools: [{ name: 'mcp__filesystem__read_file', server: 'filesystem', local_name: 'read_file', description: 'Read file', enabled: true }],
    })
    vi.spyOn(api, 'mcpResources').mockResolvedValue({
      enabled: true,
      status: 'available',
      message: '1 MCP resource(s) available.',
      resources: [{ server: 'filesystem', uri: 'file:///README.md', name: 'README.md', mime_type: 'text/markdown', description: 'Project readme' }],
    })
    vi.spyOn(api, 'mcpConfig').mockResolvedValue({
      config_path: '/tmp/aether/mcp_servers.json',
      exists: true,
      servers: [{ name: 'filesystem', enabled: true, transport: 'stdio', command: 'node', args: ['server.js'], url: null, env_keys: [], header_keys: [], timeout: null, connect_timeout: null, source: 'file' }],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} onSlashCommand={onSlashCommand} slashCommands={slashCommands} />)

    fireEvent.click(screen.getByLabelText('Open composer menu'))
    expect(screen.getByRole('menu', { name: 'Composer menu' })).toBeTruthy()
    fireEvent.click(screen.getByRole('menuitem', { name: 'MCP' }))

    expect(screen.getByLabelText('MCP inspector')).toBeTruthy()
    expect(onSend).not.toHaveBeenCalled()
    expect(onSlashCommand).not.toHaveBeenCalled()
  })

  it('reads MCP resource content from the inspector', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'mcpStatus').mockResolvedValue({
      enabled: true,
      status: 'available',
      message: '1 MCP server(s) exposed through the tool catalog.',
      servers: [{ name: 'filesystem', status: 'available', tools_count: 2, resources_count: 1, credential_status: 'unknown' }],
      imported_tools: [{ name: 'mcp__filesystem__read_file', server: 'filesystem', local_name: 'read_file', description: 'Read file', enabled: true }],
    })
    vi.spyOn(api, 'mcpResources').mockResolvedValue({
      enabled: true,
      status: 'available',
      message: '1 MCP resource(s) available.',
      resources: [{ server: 'filesystem', uri: 'file:///README.md', name: 'README.md', mime_type: 'text/markdown', description: 'Project readme' }],
    })
    vi.spyOn(api, 'mcpConfig').mockResolvedValue({
      config_path: '/tmp/aether/mcp_servers.json',
      exists: true,
      servers: [{ name: 'filesystem', enabled: true, transport: 'stdio', command: 'node', args: ['server.js'], url: null, env_keys: [], header_keys: [], timeout: null, connect_timeout: null, source: 'file' }],
    })
    vi.spyOn(api, 'mcpResourceRead').mockResolvedValue({
      enabled: true,
      status: 'available',
      message: 'Read resource.',
      server: 'filesystem',
      uri: 'file:///README.md',
      name: 'README.md',
      mime_type: 'text/markdown',
      contents: [{ type: 'text', text: '# README', mime_type: 'text/markdown', uri: 'file:///README.md' }],
    })

    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/mcp', selectionStart: 4 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })
    fireEvent.click(await screen.findByRole('button', { name: 'Read resource' }))

    await waitFor(() => expect(api.mcpResourceRead).toHaveBeenCalledWith('filesystem', 'file:///README.md'))
    expect(await screen.findByRole('region', { name: 'MCP resource content' })).toBeTruthy()
    expect(screen.getByText('# README')).toBeTruthy()
  })

  it('saves and deletes managed MCP servers from the inspector', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'mcpStatus').mockResolvedValue({
      enabled: false,
      status: 'not_configured',
      message: 'No MCP servers are configured for this Aether runtime.',
      servers: [],
      imported_tools: [],
    })
    vi.spyOn(api, 'mcpResources').mockResolvedValue({
      enabled: false,
      status: 'not_configured',
      message: 'No MCP resources are available.',
      resources: [],
    })
    vi.spyOn(api, 'mcpConfig')
      .mockResolvedValueOnce({
        config_path: '/tmp/aether/mcp_servers.json',
        exists: false,
        servers: [],
      })
      .mockResolvedValue({
        config_path: '/tmp/aether/mcp_servers.json',
        exists: true,
        servers: [{ name: 'filesystem', enabled: true, transport: 'stdio', command: 'node', args: ['server.js'], url: null, env_keys: ['TOKEN'], header_keys: [], timeout: null, connect_timeout: null, source: 'file' }],
      })
    vi.spyOn(api, 'upsertMcpServer').mockResolvedValue({
      ok: true,
      config_path: '/tmp/aether/mcp_servers.json',
      message: "MCP server 'filesystem' saved.",
      server: { name: 'filesystem', enabled: true, transport: 'stdio', command: 'node', args: ['server.js'], url: null, env_keys: ['TOKEN'], header_keys: [], timeout: null, connect_timeout: null, source: 'file' },
    })
    vi.spyOn(api, 'deleteMcpServer').mockResolvedValue({
      ok: true,
      config_path: '/tmp/aether/mcp_servers.json',
      message: "MCP server 'filesystem' deleted.",
      server: null,
    })

    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/mcp', selectionStart: 4 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    await screen.findByRole('region', { name: 'MCP server management' })
    fireEvent.change(screen.getByPlaceholderText('filesystem'), { target: { value: 'filesystem' } })
    fireEvent.change(screen.getByPlaceholderText('node'), { target: { value: 'node' } })
    fireEvent.change(screen.getByPlaceholderText('TOKEN=${MCP_TOKEN}'), { target: { value: 'TOKEN=${MCP_TOKEN}' } })
    fireEvent.change(screen.getByLabelText('Args'), { target: { value: 'server.js' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save MCP server' }))

    await waitFor(() => expect(api.upsertMcpServer).toHaveBeenCalledWith({
      name: 'filesystem',
      enabled: true,
      command: 'node',
      args: ['server.js'],
      env: { TOKEN: '${MCP_TOKEN}' },
      transport: 'stdio',
    }))
    expect(await screen.findByText("MCP server 'filesystem' saved.")).toBeTruthy()

    fireEvent.click(await screen.findByRole('button', { name: 'Delete server' }))
    await waitFor(() => expect(api.deleteMcpServer).toHaveBeenCalledWith('filesystem'))
  })

  it('inserts slash and workspace triggers from the composer control menu', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.click(screen.getByLabelText('Open composer menu'))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Slash command' }))
    expect(textbox.value).toBe('/')

    fireEvent.click(screen.getByLabelText('Open composer menu'))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Workspace reference' }))
    expect(textbox.value).toBe('/ @')
  })

  it('preserves drafts independently while switching sessions', async () => {
    const onSend = vi.fn()
    const { rerender } = render(
      <Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-a" />
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: 'draft for a', selectionStart: 11 } })
    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-b" />)
    await waitFor(() => expect(textbox.value).toBe(''))

    fireEvent.change(textbox, { target: { value: 'draft for b', selectionStart: 11 } })
    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-a" />)
    await waitFor(() => expect(textbox.value).toBe('draft for a'))

    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-b" />)
    await waitFor(() => expect(textbox.value).toBe('draft for b'))
  })
  it("preserves workspace-reference drafts independently while switching sessions", async () => {
    const onSend = vi.fn()
    vi.spyOn(api, "workspaceSearch").mockImplementation(async (query) => ({
      root: "/workspace/Aether",
      query,
      entries: query.toLowerCase().includes("read")
        ? [{ kind: "file", name: "README.md", path: "README.md" }]
        : [{ kind: "file", name: "app.ts", path: "src/app.ts" }],
    }))
    const { rerender } = render(
      <Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-a" />
    )
    const textbox = screen.getByRole("textbox") as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: "@app", selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole("option", { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: "Enter" })
    await waitFor(() => expect(textbox.value).toBe("@src/app.ts "))

    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-b" />)
    await waitFor(() => expect(textbox.value).toBe(""))

    fireEvent.change(textbox, { target: { value: "@read", selectionStart: 5 } })
    await waitFor(() => expect(screen.getByRole("option", { name: /README.md/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: "Enter" })
    await waitFor(() => expect(textbox.value).toBe("@README.md "))

    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-a" />)
    await waitFor(() => expect(textbox.value).toBe("@src/app.ts "))
    expect(screen.getByText("src/app.ts")).toBeTruthy()
    expect(screen.queryByText("README.md")).toBeNull()

    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-b" />)
    await waitFor(() => expect(textbox.value).toBe("@README.md "))
    expect(screen.getAllByText("README.md").length).toBeGreaterThan(0)
  })

  it('sends absolute paths as normal prompts', () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        onSlashCommand={onSlashCommand}
        slashCommands={slashCommands}
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/workspace/Aether', selectionStart: 17 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).toHaveBeenCalledWith('/workspace/Aether')
    expect(onSlashCommand).not.toHaveBeenCalled()
  })

  it('previews selected files and sends them with the prompt', async () => {
    const onSend = vi.fn()
    const { container } = render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['notes'], 'notes.md', { type: 'text/markdown' })

    fireEvent.change(input, { target: { files: [file] } })

    await waitFor(() => expect(screen.getByText('notes.md')).toBeTruthy())
    fireEvent.change(textbox, { target: { value: 'summarize', selectionStart: 9 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).toHaveBeenCalledWith('summarize', [
      expect.objectContaining({ type: 'text', name: 'notes.md', mimeType: 'text/markdown' }),
    ])
  })

  it('accepts pasted files as attachments', async () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement
    const file = new File(['plain'], 'clip.txt', { type: 'text/plain' })

    fireEvent.paste(textbox, {
      clipboardData: {
        files: [file],
        items: [],
      },
    })

    await waitFor(() => expect(screen.getByText('clip.txt')).toBeTruthy())
  })

  it('turns selected workspace references into path tokens and attachments', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'workspaceSearch').mockResolvedValue({
      root: '/workspace/Aether',
      query: 'app',
      entries: [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@app', selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(textbox.value).toBe('@src/app.ts ')
    expect(screen.getByText('src/app.ts')).toBeTruthy()
    fireEvent.keyDown(textbox, { key: 'Enter' })
    expect(onSend).toHaveBeenCalledWith('@src/app.ts', [
      expect.objectContaining({ type: 'text', name: 'app.ts', path: 'src/app.ts' }),
    ])
  })

  it('drills into workspace directories before attaching files', async () => {
    const onSend = vi.fn()
    const workspaceTree = vi.spyOn(api, 'workspaceTree').mockImplementation(async (path = '') => {
      if (path === '') {
        return {
          root: '/workspace/Aether',
          path: '',
          parent_path: null,
          entries: [{ kind: 'directory', name: 'src', path: 'src' }],
        }
      }
      if (path === 'src') {
        return {
          root: '/workspace/Aether',
          path: 'src',
          parent_path: '',
          entries: [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
        }
      }
      throw new Error('unexpected tree path: ' + path)
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@', selectionStart: 1 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /src\// })).toBeTruthy())
    fireEvent.click(screen.getByRole('option', { name: /src\// }))

    await waitFor(() => expect(workspaceTree).toHaveBeenCalledWith('src'))
    expect(textbox.value).toBe('@src/')
    expect(screen.queryByLabelText('Workspace context')).toBeNull()

    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())
    fireEvent.click(screen.getByRole('option', { name: /app.ts/ }))

    expect(textbox.value).toBe('@src/app.ts ')
    expect(screen.getByText('src/app.ts')).toBeTruthy()
    fireEvent.keyDown(textbox, { key: 'Enter' })
    expect(onSend).toHaveBeenCalledWith('@src/app.ts', [
      expect.objectContaining({ type: 'text', name: 'app.ts', path: 'src/app.ts' }),
    ])
  })

  it('shows clickable workspace browser breadcrumbs while drilling into directories', async () => {
    const onSend = vi.fn()
    const workspaceTree = vi.spyOn(api, 'workspaceTree').mockImplementation(async (path = '') => {
      if (path === '') {
        return {
          root: '/workspace/Aether',
          path: '',
          parent_path: null,
          entries: [{ kind: 'directory', name: 'src', path: 'src' }],
        }
      }
      if (path === 'src') {
        return {
          root: '/workspace/Aether',
          path: 'src',
          parent_path: '',
          entries: [
            { kind: 'directory', name: 'components', path: 'src/components' },
            { kind: 'file', name: 'app.ts', path: 'src/app.ts' },
          ],
        }
      }
      if (path === 'src/components') {
        return {
          root: '/workspace/Aether',
          path: 'src/components',
          parent_path: 'src',
          entries: [{ kind: 'file', name: 'Button.tsx', path: 'src/components/Button.tsx' }],
        }
      }
      throw new Error('unexpected tree path: ' + path)
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@', selectionStart: 1 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /src\// })).toBeTruthy())
    expect(screen.getByText('1 item / 1 dir')).toBeTruthy()
    fireEvent.click(screen.getByRole('option', { name: /src\// }))

    await waitFor(() => expect(textbox.value).toBe('@src/'))
    const srcBreadcrumb = screen.getByRole('navigation', { name: 'Workspace reference path' })
    expect(within(srcBreadcrumb).getByRole('button', { name: 'root' })).toBeTruthy()
    expect(within(srcBreadcrumb).getByRole('button', { name: 'src' })).toBeTruthy()
    expect(screen.getByText('2 items / 1 dir')).toBeTruthy()
    fireEvent.click(screen.getByRole('option', { name: /components\// }))

    await waitFor(() => expect(textbox.value).toBe('@src/components/'))
    const nestedBreadcrumb = screen.getByRole('navigation', { name: 'Workspace reference path' })
    fireEvent.click(within(nestedBreadcrumb).getByRole('button', { name: 'src' }))
    await waitFor(() => expect(textbox.value).toBe('@src/'))
    fireEvent.click(within(screen.getByRole('navigation', { name: 'Workspace reference path' })).getByRole('button', { name: 'root' }))
    await waitFor(() => expect(textbox.value).toBe('@'))

    expect(workspaceTree).toHaveBeenCalledWith('src/components')
    expect(workspaceTree).toHaveBeenCalledWith('src')
    expect(workspaceTree).toHaveBeenCalledWith('')
    expect(screen.queryByLabelText('Workspace context')).toBeNull()
    expect(onSend).not.toHaveBeenCalled()
  })

  it('navigates search-result directories with keyboard before applying a file', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'workspaceSearch').mockResolvedValue({
      root: '/workspace/Aether',
      query: 'src',
      entries: [{ kind: 'directory', name: 'src', path: 'src' }],
    })
    vi.spyOn(api, 'workspaceTree').mockImplementation(async (path = '') => {
      if (path === '') {
        return { root: '/workspace/Aether', path: '', parent_path: null, entries: [] }
      }
      if (path === 'src') {
        return {
          root: '/workspace/Aether',
          path: 'src',
          parent_path: '',
          entries: [{ kind: 'file', name: 'index.ts', path: 'src/index.ts' }],
        }
      }
      throw new Error('unexpected tree path: ' + path)
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@src', selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /src\// })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    await waitFor(() => expect(textbox.value).toBe('@src/'))
    expect(screen.queryByLabelText('Workspace context')).toBeNull()

    await waitFor(() => expect(screen.getByRole('option', { name: /index.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(textbox.value).toBe('@src/index.ts ')
    expect(screen.getByText('src/index.ts')).toBeTruthy()
    expect(onSend).not.toHaveBeenCalled()
  })

  it('supports file-browser arrow key navigation for workspace folders', async () => {
    const onSend = vi.fn()
    const workspaceTree = vi.spyOn(api, 'workspaceTree').mockImplementation(async (path = '') => {
      if (path === '') {
        return {
          root: '/workspace/Aether',
          path: '',
          parent_path: null,
          entries: [
            { kind: 'directory', name: 'src', path: 'src' },
            { kind: 'file', name: 'README.md', path: 'README.md' },
          ],
        }
      }
      if (path === 'src') {
        return {
          root: '/workspace/Aether',
          path: 'src',
          parent_path: '',
          entries: [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
        }
      }
      throw new Error('unexpected tree path: ' + path)
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@', selectionStart: 1 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /src\// })).toBeTruthy())

    fireEvent.keyDown(textbox, { key: 'ArrowRight' })
    await waitFor(() => expect(textbox.value).toBe('@src/'))
    expect(workspaceTree).toHaveBeenCalledWith('src')
    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())

    fireEvent.keyDown(textbox, { key: 'ArrowLeft' })
    await waitFor(() => expect(textbox.value).toBe('@'))
    await waitFor(() => expect(screen.getByRole('option', { name: /README.md/ })).toBeTruthy())

    fireEvent.keyDown(textbox, { key: 'End' })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(textbox.value).toBe('@README.md ')
    expect(screen.getByLabelText('Workspace context')).toBeTruthy()
    expect(screen.getAllByText('README.md').length).toBeGreaterThan(0)
    expect(onSend).not.toHaveBeenCalled()
  })

  it('shows workspace references as managed context chips and removes matching text tokens', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'workspaceSearch').mockResolvedValue({
      root: '/workspace/Aether',
      query: 'app',
      entries: [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@app', selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    const context = screen.getByLabelText('Workspace context')
    expect(context).toBeTruthy()
    expect(within(context).getByText('Workspace context')).toBeTruthy()
    expect(within(context).getByText('1 ref')).toBeTruthy()
    expect(screen.getByText('app.ts')).toBeTruthy()
    expect(screen.getByText('src/app.ts')).toBeTruthy()

    fireEvent.click(screen.getByLabelText('Remove workspace reference app.ts'))
    await waitFor(() => expect(screen.queryByLabelText('Workspace context')).toBeNull())
    expect(textbox.value).toBe('')
  })

  it("syncs selected workspace references when their visible @path token is edited away", async () => {
    const onSend = vi.fn()
    vi.spyOn(api, "workspaceSearch").mockResolvedValue({
      root: "/workspace/Aether",
      query: "app",
      entries: [{ kind: "file", name: "app.ts", path: "src/app.ts" }],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole("textbox") as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: "@app", selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole("option", { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: "Enter" })

    expect(screen.getByLabelText("Workspace context")).toBeTruthy()
    fireEvent.change(textbox, { target: { value: "summarize", selectionStart: 9 } })

    await waitFor(() => expect(screen.queryByLabelText("Workspace context")).toBeNull())
    fireEvent.keyDown(textbox, { key: "Enter" })
    expect(onSend).toHaveBeenCalledWith("summarize")
  })

  it('previews and copies selected workspace references from the context strip', async () => {
    const onSend = vi.fn()
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    })
    vi.spyOn(api, 'workspaceSearch').mockResolvedValue({
      root: '/workspace/Aether',
      query: 'app',
      entries: [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
    })
    const workspaceFile = vi.spyOn(api, 'workspaceFile').mockResolvedValue({
      root: '/workspace/Aether',
      path: 'src/app.ts',
      name: 'app.ts',
      content: 'export const app = 1',
      size_bytes: 20,
      updated_at: 0,
      language: 'typescript',
      truncated: false,
      binary: false,
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@app', selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    const context = screen.getByLabelText('Workspace context')
    fireEvent.click(within(context).getByRole('button', { name: 'Copy workspace reference app.ts' }))
    fireEvent.click(within(context).getByRole('button', { name: 'Preview workspace reference app.ts' }))

    await waitFor(() => expect(writeText).toHaveBeenCalledWith('src/app.ts'))
    await waitFor(() => expect(workspaceFile).toHaveBeenCalledWith('src/app.ts'))
    const preview = screen.getByLabelText('Workspace reference preview')
    expect(within(preview).getByText('src/app.ts')).toBeTruthy()
    expect(within(preview).getByText('export const app = 1')).toBeTruthy()
  })

  it('reorders selected workspace references from the context strip', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'workspaceSearch').mockImplementation(async (query) => ({
      root: '/workspace/Aether',
      query,
      entries: query.toLowerCase().includes('read')
        ? [{ kind: 'file', name: 'README.md', path: 'README.md' }]
        : [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
    }))
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@app', selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    const nextValue = textbox.value + '@read'
    fireEvent.change(textbox, { target: { value: nextValue, selectionStart: nextValue.length } })
    await waitFor(() => expect(screen.getByRole('option', { name: /README.md/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    const context = screen.getByLabelText('Workspace context')
    const groupLabels = () => within(context)
      .getAllByRole('group')
      .map((item) => item.getAttribute('aria-label'))
    expect(groupLabels()).toEqual(['Workspace reference app.ts', 'Workspace reference README.md'])

    fireEvent.click(within(context).getByRole('button', { name: 'Move workspace reference README.md earlier' }))
    expect(groupLabels()).toEqual(['Workspace reference README.md', 'Workspace reference app.ts'])

    fireEvent.click(within(context).getByRole('button', { name: 'Move workspace reference README.md later' }))
    expect(groupLabels()).toEqual(['Workspace reference app.ts', 'Workspace reference README.md'])
    expect(onSend).not.toHaveBeenCalled()
  })

  it('supports keyboard preview and removal for selected workspace references', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'workspaceSearch').mockResolvedValue({
      root: '/workspace/Aether',
      query: 'app',
      entries: [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
    })
    const workspaceFile = vi.spyOn(api, 'workspaceFile').mockResolvedValue({
      root: '/workspace/Aether',
      path: 'src/app.ts',
      name: 'app.ts',
      content: 'export const app = 1',
      size_bytes: 20,
      updated_at: 0,
      language: 'typescript',
      truncated: false,
      binary: false,
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@app', selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    const context = screen.getByLabelText('Workspace context')
    const chip = within(context).getByRole('group', { name: 'Workspace reference app.ts' })
    fireEvent.keyDown(chip, { key: 'Enter' })
    await waitFor(() => expect(workspaceFile).toHaveBeenCalledWith('src/app.ts'))
    expect(screen.getByLabelText('Workspace reference preview')).toBeTruthy()

    fireEvent.keyDown(chip, { key: 'Backspace' })
    await waitFor(() => expect(screen.queryByLabelText('Workspace context')).toBeNull())
    expect(textbox.value).toBe('')
    expect(onSend).not.toHaveBeenCalled()
  })


  it('debounces draft context estimates into the footer ring', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'estimateContext').mockResolvedValue({
      session_id: 'session-estimate',
      context_engine: 'default',
      compression_count: 0,
      message_count: 1,
      token_estimate: 64000,
      prompt_tokens: 64000,
      context_window: 128000,
      pressure_level: 'medium',
      next_action: 'none',
      breakdown: [{ label: 'Transcript', tokens: 64000, detail: 'draft included' }],
    })
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        slashCommands={slashCommands}
        sessionId="session-estimate"
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: 'Summarize the selected file', selectionStart: 27 } })

    await waitFor(() => expect(api.estimateContext).toHaveBeenCalledWith('session-estimate', expect.objectContaining({ draft: 'Summarize the selected file' })))
    expect(screen.getByLabelText(/Next run estimate: 64,000 tokens/)).toBeTruthy()
  })

  it('refreshes draft context estimates when the selected model changes', async () => {
    const onSend = vi.fn()
    const estimateContext = vi.spyOn(api, 'estimateContext').mockResolvedValue({
      session_id: 'session-model-estimate',
      context_engine: 'default',
      compression_count: 0,
      message_count: 1,
      token_estimate: 1200,
      prompt_tokens: 1200,
      context_window: 128000,
      pressure_level: 'low',
      next_action: 'none',
      breakdown: [],
    })
    const props = {
      disabled: false,
      running: false,
      onCancel: () => undefined,
      onSend,
      slashCommands,
      sessionId: 'session-model-estimate',
    }
    const { rerender } = render(<Composer {...props} provider="openai" model="gpt-4o" />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: 'Estimate this prompt', selectionStart: 20 } })

    await waitFor(() => expect(estimateContext).toHaveBeenCalledTimes(1))
    rerender(<Composer {...props} provider="codex" model="gpt-5.4" />)

    await waitFor(() => expect(estimateContext).toHaveBeenCalledTimes(2))
  })

  it('renders command-surface metadata in the footer', () => {
    const onSend = vi.fn()
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        slashCommands={slashCommands}
        provider="codex"
        model="gpt-5.4"
        mode="plan"
        inputTokens={1200}
        outputTokens={800}
      />,
    )

    expect(screen.getByText('gpt-5.4')).toBeTruthy()
    expect(screen.getByText('Workspace')).toBeTruthy()
    expect(screen.getByText('plan')).toBeTruthy()
    expect(screen.getByLabelText(/2,000 tokens/)).toBeTruthy()
  })
})
