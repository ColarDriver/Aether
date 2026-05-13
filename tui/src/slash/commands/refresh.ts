import type { SlashCommand } from '../dispatcher.js'

export const refreshCommand: SlashCommand = {
  name: '/refresh',
  category: 'local',
  async execute() {
    return { kind: 'refresh' }
  }
}
