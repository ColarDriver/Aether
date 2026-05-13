import type { SlashCommand } from '../dispatcher.js'

export const systemCommand: SlashCommand = {
  name: '/system',
  category: 'remote',
  async execute(args, ctx) {
    const text = args.join(' ').trim()
    if (!text) {
      const current = ctx.getSession().systemOverride
      return {
        kind: 'note',
        level: 'info',
        text: current ? `system override:\n${current}` : 'No system override set.'
      }
    }
    ctx.setSystemOverride(text)
    return {
      kind: 'note',
      level: 'info',
      text: 'System override will be sent with the next agent.run request.'
    }
  }
}
