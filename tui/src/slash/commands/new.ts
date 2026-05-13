import type { SlashCommand } from '../dispatcher.js'

export const newCommand: SlashCommand = {
  name: '/new',
  category: 'remote',
  async execute(_args, ctx) {
    const info = await ctx.createSession()
    return {
      kind: 'session',
      info,
      note: `started new session ${info.session_id.slice(0, 8)}`
    }
  }
}
