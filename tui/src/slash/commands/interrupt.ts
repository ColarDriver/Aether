import type { SlashCommand } from '../dispatcher.js'

export const interruptCommand: SlashCommand = {
  name: '/interrupt',
  category: 'control',
  async execute(_args, ctx) {
    const sessionId = ctx.getSession().sessionId
    if (!sessionId) {
      return { kind: 'note', level: 'warn', text: 'No active session to interrupt.' }
    }
    await ctx.client.request('agent.cancel', { session_id: sessionId })
    return { kind: 'note', level: 'warn', text: 'interrupt requested' }
  }
}
