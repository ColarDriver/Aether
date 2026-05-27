// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, getSessionToken, request, setBaseUrl, setSessionToken } from './client'

describe('api client', () => {
  beforeEach(() => {
    setBaseUrl('http://aether.test')
    setSessionToken('token-1')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('sends the Aether session token header', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const result = await request<{ ok: boolean }>('GET', '/api/status')

    expect(result.ok).toBe(true)
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect((init.headers as Record<string, string>)['X-Aether-Session-Token']).toBe('token-1')
  })

  it('encodes documentation path segments without flattening slashes', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ path: 'sprint 20/00_overview.md' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.doc('sprint 20/00_overview.md')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/docs/sprint%2020/00_overview.md',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('posts plan clear requests for web clear commands', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ session_id: 's1', mode: 'agent', has_plan: false }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.clearPlan('s1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/plan/s1/clear',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('posts session fork requests', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ source_session_id: 's1', forked_from_index: 2, messages_copied: 3, info: { session_id: 's2', created_at: 1, updated_at: 1, provider: 'openai', model: 'gpt-5.4', message_count: 3 }, messages: [] }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.forkSession('s1', { message_index: 2 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/sessions/s1/fork',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ message_index: 2 }) }),
    )
  })

  it('posts session rewind requests', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ session_id: 's1', rewound_to_index: 0, messages_kept: 1, messages_removed: 2, info: { session_id: 's1', created_at: 1, updated_at: 1, provider: 'openai', model: 'gpt-5.4', message_count: 1 }, messages: [] }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.rewindSession('s1', { message_index: 0 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/sessions/s1/rewind',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ message_index: 0 }) }),
    )
  })

  it('gets per-file turn checkpoint diffs', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ session_id: 's1', state: 'ok', path: 'app.py', diff: '@@', target: { target_user_message_id: 'turn-1', user_message_index: 0, user_message_count: 1, message_index: 0 } }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.sessionTurnCheckpointDiff('s1', { path: 'app.py', target_user_message_id: 'turn-1', user_message_index: 0 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/sessions/s1/turn-checkpoints/diff?path=app.py&target_user_message_id=turn-1&user_message_index=0',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('renames, exports, and imports session records', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      if (String(_url).endsWith('/rename')) {
        return new Response(JSON.stringify({ session_id: 's2', created_at: 1, updated_at: 1, provider: 'openai', model: 'gpt-5.4', message_count: 0 }), { status: 200 })
      }
      if (String(_url).endsWith('/export')) {
        return new Response(JSON.stringify({ session_id: 's2', data: { session_id: 's2', provider: 'openai', model: 'gpt-5.4' } }), { status: 200 })
      }
      return new Response(JSON.stringify({ source_session_id: 's2', overwritten: false, info: { session_id: 's3', created_at: 1, updated_at: 1, provider: 'openai', model: 'gpt-5.4', message_count: 0 }, messages: [] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    await api.renameSession('s1', 's2')
    await api.exportSession('s2')
    await api.importSession({ data: { session_id: 's2' }, new_session_id: 's3', overwrite: true })

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://aether.test/api/sessions/s1/rename',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ new_session_id: 's2' }) }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://aether.test/api/sessions/s2/export',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      'http://aether.test/api/sessions/import',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ data: { session_id: 's2' }, new_session_id: 's3', overwrite: true }) }),
    )
  })

  it('calls context status, estimate, and compression endpoints', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ session_id: 's1', context_engine: 'default', compression_count: 0, message_count: 2, token_estimate: 120 }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.contextStatus('s1')
    await api.estimateContext('s1', { draft: 'hello', attachments: [{ content: 'file' }] })
    await api.compressContext('s1', { focus: 'auth', force: true })

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://aether.test/api/context/s1/status',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://aether.test/api/context/s1/estimate',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ draft: 'hello', attachments: [{ content: 'file' }] }) }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      'http://aether.test/api/context/s1/compress',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ focus: 'auth', force: true }) }),
    )
  })

  it('fetches provider preflight diagnostics with runtime selectors', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ family: 'openai-compatible', provider_name: 'openai', model: 'gpt-5.4', status: 'ready', ready: true, issues: [], suggestions: [] }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.providerPreflight({
      provider: 'openai',
      model: 'gpt-5.4',
      baseUrl: 'https://provider.test/v1',
    })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/providers/preflight?provider=openai&model=gpt-5.4&base_url=https%3A%2F%2Fprovider.test%2Fv1',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('fetches task observer messages with a limit', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ task_id: 'task-1', messages: [], pending_messages: [], delivered_messages: [], total_count: 0, truncated: false }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.taskMessages('task-1', { limit: 25 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/tasks/task-1/messages?limit=25',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('fetches child task message streams with limits', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ task_id: 'task-1', streams: [], total_count: 0, truncated: false }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.taskChildMessages('task-1', { limit: 10, perTaskLimit: 5 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/tasks/task-1/children/messages?limit=10&per_task_limit=5',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('fetches MCP resources with optional server filter', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ enabled: true, status: 'not_available', message: 'No resources', resources: [] }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.mcpResources('filesystem')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/mcp/resources?server=filesystem',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('reads MCP resources with server and uri query parameters', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ enabled: true, status: 'available', message: 'Read resource', server: 'filesystem', uri: 'file:///README.md', contents: [] }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.mcpResourceRead('filesystem', 'file:///README.md')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/mcp/resources/read?server=filesystem&uri=file%3A%2F%2F%2FREADME.md',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('manages MCP server configuration through REST routes', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      if (init?.method === 'GET') {
        return new Response(JSON.stringify({ config_path: '/tmp/mcp_servers.json', exists: false, servers: [] }), { status: 200 })
      }
      if (init?.method === 'PUT') {
        return new Response(JSON.stringify({ ok: true, config_path: '/tmp/mcp_servers.json', message: 'saved', server: { name: 'local_fs', enabled: true, transport: 'stdio', command: 'node', args: ['server.js'], env_keys: [], header_keys: [], source: 'file' } }), { status: 200 })
      }
      if (init?.method === 'POST') {
        return new Response(JSON.stringify({ enabled: false, status: 'not_configured', message: 'No MCP servers are configured.', servers: [], imported_tools: [] }), { status: 200 })
      }
      return new Response(JSON.stringify({ ok: true, config_path: '/tmp/mcp_servers.json', message: 'deleted' }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    await api.mcpConfig()
    await api.upsertMcpServer({ name: 'local fs', command: 'node', args: ['server.js'] })
    await api.refreshMcp()
    await api.deleteMcpServer('local_fs')

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://aether.test/api/mcp/config',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://aether.test/api/mcp/servers',
      expect.objectContaining({ method: 'PUT', body: JSON.stringify({ name: 'local fs', command: 'node', args: ['server.js'] }) }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      'http://aether.test/api/mcp/refresh',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      'http://aether.test/api/mcp/servers/local_fs',
      expect.objectContaining({ method: 'DELETE' }),
    )
  })

  it('calls workspace git and checkpoint endpoints', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      if (String(_url).includes('/git/status')) {
        return new Response(JSON.stringify({ root: '/repo', git_root: '/repo', available: true, clean: false, ahead: 0, behind: 0, files: [] }), { status: 200 })
      }
      if (String(_url).includes('/git/diff')) {
        return new Response(JSON.stringify({ root: '/repo', path: 'app.py', diff: '@@', staged: false, truncated: false }), { status: 200 })
      }
      if (String(_url).includes('/git/restore')) {
        return new Response(JSON.stringify({ root: '/repo', git_root: '/repo', available: true, clean: true, ahead: 0, behind: 0, files: [] }), { status: 200 })
      }
      if (init?.method === 'GET') {
        return new Response(JSON.stringify({ root: '/repo', checkpoints: [] }), { status: 200 })
      }
      return new Response(JSON.stringify({ checkpoint_id: 'cp-1', label: 'before', created_at: 1, root: '/repo', files: [] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    await api.workspaceGitStatus()
    await api.workspaceGitDiff('src/app.py')
    await api.workspaceGitRestore('src/app.py')
    await api.workspaceCheckpoints()
    await api.createWorkspaceCheckpoint({ label: 'before' })
    await api.restoreWorkspaceCheckpoint('cp-1')

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://aether.test/api/workspace/git/status',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://aether.test/api/workspace/git/diff?path=src%2Fapp.py',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      'http://aether.test/api/workspace/git/restore',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ path: 'src/app.py' }) }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      'http://aether.test/api/workspace/checkpoints',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      'http://aether.test/api/workspace/checkpoints',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ label: 'before' }) }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      'http://aether.test/api/workspace/checkpoints/cp-1/restore',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('calls workspace change management endpoints', async () => {
    const fetchMock = vi.fn(async (_url: string) => {
      if (String(_url).endsWith('/api/workspace/changes')) {
        return new Response(JSON.stringify({ root: '/repo', git_root: '/repo', available: true, changes: [] }), { status: 200 })
      }
      return new Response(JSON.stringify({
        root: '/repo',
        action: 'accepted',
        paths: ['app.py'],
        status: { root: '/repo', git_root: '/repo', available: true, clean: false, ahead: 0, behind: 0, files: [] },
      }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    await api.workspaceChanges()
    await api.acceptWorkspaceChanges(['app.py'])
    await api.rejectWorkspaceChanges({ paths: ['app.py'], checkpoint_id: 'cp-1', expected_hashes: { 'app.py': 'abc' } })

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://aether.test/api/workspace/changes',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://aether.test/api/workspace/changes/accept',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ paths: ['app.py'] }) }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      'http://aether.test/api/workspace/changes/reject',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ paths: ['app.py'], checkpoint_id: 'cp-1', expected_hashes: { 'app.py': 'abc' } }) }),
    )
  })

  it('calls workspace root endpoints', async () => {
    const fetchMock = vi.fn(async (_url: string) => new Response(JSON.stringify({
      root: '/workspace/Aether',
      name: 'Aether',
      exists: true,
      readable: true,
      git_root: '/workspace/Aether',
      is_git: true,
      recent_roots: ['/workspace/Aether'],
    }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.workspaceRoot()
    await api.switchWorkspaceRoot({ path: '/workspace/Other', session_id: 'ses', remember: true })

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://aether.test/api/workspace/root',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://aether.test/api/workspace/root',
      expect.objectContaining({ method: 'PUT', body: JSON.stringify({ path: '/workspace/Other', session_id: 'ses', remember: true }) }),
    )
  })

  it('posts task stop requests', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ task_id: 'task-1', delivered: true, status: 'running', message: 'Stop signal sent to running task.' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.stopTask('task-1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/tasks/task-1/stop',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('posts task follow-up messages', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ task_id: 'task-1', queued: true, status: 'running', message: 'Queued message', queued_chars: 9 }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.sendTaskMessage('task-1', { message: 'follow up' })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/tasks/task-1/messages',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ message: 'follow up' }) }),
    )
  })

  it('fetches task result artifacts', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ task_id: 'task-1', result_path: '/tmp/result.json', result: { summary: 'ok' } }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.taskResult('task-1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://aether.test/api/tasks/task-1/result',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('refreshes the bootstrap token and retries once after a 401', async () => {
    setSessionToken('stale-token')
    let privateCalls = 0
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === 'http://aether.test/api/bootstrap') {
        return new Response(JSON.stringify({ session_token: 'fresh-token', auth_enabled: true, web: { enabled: true } }), { status: 200 })
      }
      if (url === 'http://aether.test/api/private') {
        privateCalls += 1
        if (privateCalls === 1) {
          expect((init?.headers as Record<string, string>)['X-Aether-Session-Token']).toBe('stale-token')
          return new Response(JSON.stringify({ error: { message: 'Unauthorized' } }), { status: 401 })
        }
        expect((init?.headers as Record<string, string>)['X-Aether-Session-Token']).toBe('fresh-token')
        return new Response(JSON.stringify({ ok: true }), { status: 200 })
      }
      return new Response('', { status: 404 })
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await request<{ ok: boolean }>('GET', '/api/private')

    expect(result.ok).toBe(true)
    expect(getSessionToken()).toBe('fresh-token')
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('maps structured errors to ApiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ error: { message: 'bad request' } }), {
          status: 400,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    await expect(request('GET', '/api/private')).rejects.toMatchObject({
      name: 'ApiError',
      status: 400,
      message: 'bad request',
    } satisfies Partial<ApiError>)
  })
})
