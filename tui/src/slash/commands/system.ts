import { chatActions } from '../../store/chatStore.js'
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
    // Mirrors Python `_cmd_system` in `aether/cli/commands.py:138-143`:
    // setting a new system prompt clears the visible transcript so the
    // next turn starts with a clean conversation context (the prior
    // exchanges would be inconsistent with the new instruction).
    ctx.setSystemOverride(text)
    chatActions.reset()
    return {
      kind: 'note',
      level: 'success',
      text: 'System override updated; cleared conversation history.'
    }
  }
}
