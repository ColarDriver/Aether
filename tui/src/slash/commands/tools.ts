import type { SlashCommand, ToolsListResult } from '../dispatcher.js'

export const toolsCommand: SlashCommand = {
  name: '/tools',
  category: 'remote',
  async execute(_args, ctx) {
    const result = await ctx.client.request<ToolsListResult>('tools.list')
    if (result.tools.length === 0) {
      return { kind: 'note', level: 'warn', text: 'No tools registered.' }
    }
    const width = Math.max(...result.tools.map((tool) => tool.name.length), 12)
    const lines = result.tools.map((tool) => {
      const desc = (tool.description ?? '').replace(/\s+/g, ' ').trim()
      return `${tool.name.padEnd(width)}  ${truncate(desc, 80)}`
    })
    return {
      kind: 'note',
      level: 'info',
      text: `${result.tools.length} tools registered\n${lines.join('\n')}`
    }
  }
}

function truncate(value: string, max: number): string {
  if (value.length <= max) {
    return value
  }
  return `${value.slice(0, max - 1)}…`
}
