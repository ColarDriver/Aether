import { activityState } from '../../store/activityStore.js'
import type { SlashCommand } from '../dispatcher.js'

export const statsCommand: SlashCommand = {
  name: '/stats',
  category: 'local',
  async execute(_args, ctx) {
    // Mirrors Python `_cmd_stats` in `aether/cli/commands.py:429-436`:
    //   last turn: iter=N tool_calls=N errors=N chars=N elapsed=S.SSs
    const last = activityState.get().lastTurn
    if (!last) {
      const usage = ctx.getSession().usage
      return {
        kind: 'note',
        level: 'info',
        text:
          `no turn completed yet · usage input=${usage.input} output=${usage.output} ` +
          `cache_read=${usage.cacheRead} cache_write=${usage.cacheWrite}`
      }
    }
    const elapsedSec = (last.durationMs / 1000).toFixed(2)
    return {
      kind: 'note',
      level: 'info',
      text:
        `last turn: iter=${last.iterations} tool_calls=${last.tools} ` +
        `errors=${last.errors} chars=${last.chars} elapsed=${elapsedSec}s`
    }
  }
}
