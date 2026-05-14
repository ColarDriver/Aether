import { chatActions } from '../../store/chatStore.js'
import type { SlashCommand } from '../dispatcher.js'

export const newCommand: SlashCommand = {
  name: '/new',
  category: 'remote',
  async execute(_args, ctx) {
    // Mirrors Python `_cmd_new` in `aether/cli/commands.py:72-81`. The
    // gateway already persisted the previous session on every agent.run
    // (see `_persist_result`), so there's no separate `persist_session`
    // step here — but we MUST clear the visible transcript so the empty
    // new session doesn't keep showing the prior conversation. Python
    // achieves this via `state.messages = []`; we use chatActions.reset().
    chatActions.reset()
    const info = await ctx.createSession()
    return {
      kind: 'session',
      info,
      note: `started new session ${info.session_id.slice(0, 8)}`
    }
  }
}
