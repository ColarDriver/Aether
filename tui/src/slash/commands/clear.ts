import type { SlashCommand } from '../dispatcher.js'
import { sessionActions } from '../../store/sessionStore.js'

export const clearCommand: SlashCommand = {
  name: '/clear',
  category: 'local',
  async execute(_args, ctx) {
    const sessionId = ctx.getSession().sessionId
    if (sessionId) {
      await ctx.client
        .request('plan.clear', { session_id: sessionId })
        .catch(() => undefined)
    }
    sessionActions.setMode('agent')
    return { kind: 'clear' }
  }
}
