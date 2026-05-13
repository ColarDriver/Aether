import type { SlashCommand } from '../dispatcher.js'

export const clearCommand: SlashCommand = {
  name: '/clear',
  category: 'local',
  async execute() {
    return { kind: 'clear' }
  }
}
