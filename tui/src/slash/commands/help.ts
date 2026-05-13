import { OVERLAY_PRIORITY, overlayActions } from '../../store/overlayStore.js'
import { formatCommandCatalog, type SlashCommand } from '../dispatcher.js'

const NO_OVERLAY = process.env.AETHER_NO_OVERLAY_HELP === '1'

export const helpCommand: SlashCommand = {
  name: '/help',
  category: 'local',
  async execute(args, ctx) {
    if (NO_OVERLAY || args.includes('--no-overlay')) {
      return {
        kind: 'note',
        level: 'info',
        text: formatCommandCatalog(ctx.catalog)
      }
    }
    overlayActions.push({
      kind: 'help',
      id: `help_${Date.now()}`,
      payload: { catalog: ctx.catalog },
      createdAt: Date.now(),
      priority: OVERLAY_PRIORITY.help,
      onDismiss: () => undefined
    })
    return { kind: 'noop' }
  }
}
