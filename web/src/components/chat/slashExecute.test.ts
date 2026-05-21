// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest'
import { executeWebSlashCommand, isSlashCommandInput, parseSlashCommand } from './slashExecute'

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
  { name: '/help', description: 'Show help', category: 'local' },
  { name: '/plan', description: 'Plan mode', category: 'session' },
]

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
    expect(help.message).toContain('`/plan`')
    expect(current).toMatchObject({ type: 'notice' })
    expect(current.message).toContain('session-123456')
  })

  it('loads sessions and tools for catalog commands', async () => {
    const sessions = vi.fn().mockResolvedValue({ sessions: [session] })
    const tools = vi.fn().mockResolvedValue({ groups: [{ name: 'filesystem', tools: [{ name: 'read_file' }] }] })

    expect((await executeWebSlashCommand('/sessions', { session, commands, loadSessions: sessions })).message).toContain('session-')
    expect((await executeWebSlashCommand('/tools', { session, commands, loadToolGroups: tools })).message).toContain('filesystem')
  })

  it('returns clear errors for unsupported or unknown commands', async () => {
    expect(await executeWebSlashCommand('/missing', { session, commands })).toEqual({
      type: 'error',
      message: 'Unknown slash command /missing. Type /help for available commands.',
    })
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
    expect(result.message).toContain('# Plan')
    expect(result.message).toContain('Inspect')
  })
})
