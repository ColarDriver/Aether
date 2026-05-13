import type { SlashCommand } from '../dispatcher.js'

export const exitCommand: SlashCommand = {
  name: '/exit',
  category: 'local',
  async execute() {
    return { kind: 'exit' }
  }
}
