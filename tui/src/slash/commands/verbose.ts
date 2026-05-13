import type { SlashCommand } from '../dispatcher.js'

export const verboseCommand: SlashCommand = {
  name: '/verbose',
  category: 'local',
  async execute(_args, ctx) {
    return { kind: 'toggle-verbose', enabled: ctx.toggleVerbose() }
  }
}
