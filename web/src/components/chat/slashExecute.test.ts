// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest'
import { executeWebSlashCommand, isSlashCommandInput, parseSlashCommand, type WebSlashResult } from './slashExecute'

const session = {
  session_id: 'session-123456',
  created_at: 1,
  updated_at: 2,
  provider: 'openai',
  model: 'gpt-5.4',
  message_count: 3,
  mode: 'agent',
}

const commands = [
  { name: '/clear', description: 'Clear conversation history', category: 'local' },
  { name: '/exit', description: 'Exit the TUI', category: 'local' },
  { name: '/help', description: 'Show help', category: 'local' },
  { name: '/interrupt', description: 'Interrupt the active turn', category: 'control' },
  { name: '/model', description: 'Show model', category: 'session' },
  { name: '/new', description: 'New session', category: 'session' },
  { name: '/plan', description: 'Plan mode', category: 'session' },
  { name: '/refresh', description: 'Refresh visible state', category: 'local' },
  { name: '/resume', description: 'Resume a session', category: 'session' },
  { name: '/session', description: 'Show session', category: 'session' },
  { name: '/sessions', description: 'List sessions', category: 'session' },
  { name: '/stats', description: 'Show stats', category: 'local' },
  { name: '/system', description: 'System prompt', category: 'session' },
  { name: '/tools', description: 'List tools', category: 'remote' },
  { name: '/verbose', description: 'Toggle verbose', category: 'local' },
]

function messageOf(result: WebSlashResult): string {
  if ('message' in result) return result.message
  throw new Error('Expected slash result with message, got ' + result.type)
}

describe('slashExecute', () => {
  it('parses slash commands without treating file paths as commands', () => {
    expect(isSlashCommandInput('/help')).toBe(true)
    expect(parseSlashCommand('/plan add auth')).toEqual({ name: 'plan', arg: 'add auth' })
    expect(isSlashCommandInput('/workspace/Aether')).toBe(false)
  })

  it('formats help and session notices without starting a run', async () => {
    const help = await executeWebSlashCommand('/help', { session, commands })
    const current = await executeWebSlashCommand('/session', { session, commands })

    expect(help).toMatchObject({ type: 'notice' })
    expect(messageOf(help)).toContain('`/plan`')
    expect(messageOf(help)).toContain('`/context`')
    expect(current).toMatchObject({ type: 'notice' })
    expect(messageOf(current)).toContain('session-123456')
  })

  it('loads sessions and tools for catalog commands', async () => {
    const sessions = vi.fn().mockResolvedValue({ sessions: [session] })
    const tools = vi.fn().mockResolvedValue({ groups: [{ name: 'filesystem', tools: [{ name: 'read_file' }] }] })
    const openView = vi.fn()

    expect(messageOf(await executeWebSlashCommand('/sessions', { session, commands, loadSessions: sessions, openView }))).toContain('session-')
    expect(messageOf(await executeWebSlashCommand('/tools', { session, commands, loadToolGroups: tools, openView }))).toContain('filesystem')
    expect(openView).toHaveBeenCalledWith('sessions')
    expect(openView).toHaveBeenCalledWith('tools')
  })

  it('clears plan state for web clear commands without starting a run', async () => {
    const clearPlan = vi.fn().mockResolvedValue({
      session_id: session.session_id,
      mode: 'agent',
      has_plan: false,
      plan_path: '/tmp/plan.md',
      plan_content: null,
    })
    const onSessionMode = vi.fn()

    await expect(executeWebSlashCommand('/clear', { session: { ...session, mode: 'plan' }, commands, clearPlan, onSessionMode })).resolves.toEqual({ type: 'clear' })
    expect(clearPlan).toHaveBeenCalledWith(session.session_id)
    expect(onSessionMode).toHaveBeenCalledWith(session.session_id, 'agent')
  })

  it('returns clear errors for unsupported or unknown commands', async () => {
    expect(await executeWebSlashCommand('/missing', { session, commands })).toEqual({
      type: 'error',
      message: 'Unknown slash command /missing. Type /help for available commands.',
    })
  })

  it('executes web-local control commands instead of falling back to not-implemented errors', async () => {
    const closeConsole = vi.fn()
    const refreshSession = vi.fn().mockResolvedValue(undefined)
    const cancelRun = vi.fn().mockResolvedValue(undefined)
    const setVerbose = vi.fn()

    const exit = await executeWebSlashCommand('/exit', { session, commands, closeConsole })
    expect(exit.type).toBe('notice')
    expect(messageOf(exit)).not.toContain('not implemented')
    expect(closeConsole).toHaveBeenCalledOnce()

    const refresh = await executeWebSlashCommand('/refresh', { session, commands, refreshSession })
    expect(messageOf(refresh)).toContain('Refreshed session')
    expect(refreshSession).toHaveBeenCalledWith(session.session_id)

    const interrupt = await executeWebSlashCommand('/interrupt', { session, commands, activeRunId: 'run-123', cancelRun })
    expect(messageOf(interrupt)).toContain('Interrupt requested')
    expect(cancelRun).toHaveBeenCalledWith(session.session_id, 'run-123')

    const verbose = await executeWebSlashCommand('/verbose on', { session, commands, verbose: false, setVerbose })
    expect(messageOf(verbose)).toBe('Verbose mode on.')
    expect(setVerbose).toHaveBeenCalledWith(true)
  })

  it('handles web-local inspector commands without falling through to unsupported catalog errors', async () => {
    const result = await executeWebSlashCommand('/context', { session, commands })

    expect(result.type).toBe('notice')
    expect(messageOf(result)).toContain('composer inspector panel')
    expect(messageOf(result)).not.toContain('not implemented')
  })

  it('reports web turn stats from active run status and token usage', async () => {
    const result = await executeWebSlashCommand('/stats', {
      session,
      commands,
      activeRunId: 'run-123',
      verbose: true,
      runStatus: {
        runId: 'run-123',
        sessionId: session.session_id,
        state: 'responding',
        detail: 'writing final answer',
        elapsedMs: 1234,
        tokens: { input_tokens: 100, output_tokens: 23, total_tokens: 123 },
      },
    })

    expect(result.type).toBe('notice')
    expect(messageOf(result)).toContain('# Turn stats')
    expect(messageOf(result)).toContain('123 tokens')
    expect(messageOf(result)).toContain('responding')
    expect(messageOf(result)).toContain('Verbose | on')
  })

  it('does not report known catalog commands as unimplemented in web', async () => {
    const context = {
      session,
      commands,
      createSession: vi.fn().mockResolvedValue({ ...session, session_id: 'created' }),
      loadSessions: vi.fn().mockResolvedValue({ sessions: [session] }),
      loadToolGroups: vi.fn().mockResolvedValue({ groups: [] }),
      loadPlanCurrent: vi.fn().mockResolvedValue({
        session_id: session.session_id,
        mode: 'plan',
        has_plan: false,
        plan_path: '/tmp/plan.md',
        plan_content: null,
      }),
      clearPlan: vi.fn().mockResolvedValue({
        session_id: session.session_id,
        mode: 'agent',
        has_plan: false,
        plan_path: '/tmp/plan.md',
        plan_content: null,
      }),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      setVerbose: vi.fn(),
    }

    for (const command of commands) {
      const result = await executeWebSlashCommand(command.name, context)
      expect('message' in result ? result.message : result.type).not.toContain('not implemented in the web console yet')
    }
  })

  it('enables plan mode and sends plan descriptions as agent prompts', async () => {
    const setPlanMode = vi.fn().mockResolvedValue({
      session_id: session.session_id,
      mode: 'plan',
      has_plan: false,
      plan_path: '/tmp/plan.md',
      plan_content: null,
    })
    const onSessionMode = vi.fn()

    const result = await executeWebSlashCommand('/plan add auth flow', {
      session,
      commands,
      loadPlanCurrent: vi.fn().mockResolvedValue({
        session_id: session.session_id,
        mode: 'agent',
        has_plan: false,
        plan_path: '/tmp/plan.md',
        plan_content: null,
      }),
      setPlanMode,
      onSessionMode,
    })

    expect(result).toEqual({ type: 'send', message: 'add auth flow' })
    expect(setPlanMode).toHaveBeenCalledWith(session.session_id, 'plan')
    expect(onSessionMode).toHaveBeenCalledWith(session.session_id, 'plan')
  })

  it('shows current plan content when already in plan mode', async () => {
    const result = await executeWebSlashCommand('/plan', {
      session: { ...session, mode: 'plan' },
      commands,
      loadPlanCurrent: vi.fn().mockResolvedValue({
        session_id: session.session_id,
        mode: 'plan',
        has_plan: true,
        plan_path: '/tmp/plan.md',
        plan_content: '# Plan\n\n- Inspect',
      }),
    })

    expect(result.type).toBe('notice')
    expect(messageOf(result)).toContain('# Plan')
    expect(messageOf(result)).toContain('Inspect')
  })

  it('creates and resumes sessions through web callbacks', async () => {
    const createSession = vi.fn().mockResolvedValue({ ...session, session_id: 'new-session' })
    const resumeSession = vi.fn().mockResolvedValue({ ...session, session_id: 'old-session' })
    const openView = vi.fn()

    expect(messageOf(await executeWebSlashCommand('/new', { session, commands, createSession, openView }))).toContain('new-session')
    expect(createSession).toHaveBeenCalledWith({ provider: 'openai', model: 'gpt-5.4' })

    expect(messageOf(await executeWebSlashCommand('/resume old', { session, commands, resumeSession, openView }))).toContain('old-session')
    expect(resumeSession).toHaveBeenCalledWith('old')
    expect(openView).toHaveBeenCalledWith('chat')
  })

  it('opens sessions for bare resume and updates model and system prompt', async () => {
    const openView = vi.fn()
    const updateSession = vi.fn()
      .mockResolvedValueOnce({ ...session, model: 'gpt-5.4-mini' })
      .mockResolvedValueOnce({ ...session, system_prompt: 'Be concise' })

    expect(messageOf(await executeWebSlashCommand('/resume', { session, commands, openView }))).toContain('Opened Sessions')
    expect(openView).toHaveBeenCalledWith('sessions')

    expect(messageOf(await executeWebSlashCommand('/model gpt-5.4-mini', { session, commands, updateSession }))).toContain('gpt-5.4-mini')
    expect(updateSession).toHaveBeenCalledWith(session.session_id, { model: 'gpt-5.4-mini' })

    expect(messageOf(await executeWebSlashCommand('/system Be concise', { session, commands, updateSession }))).toContain('Updated system prompt')
    expect(updateSession).toHaveBeenCalledWith(session.session_id, { system_prompt: 'Be concise' })
  })
})
