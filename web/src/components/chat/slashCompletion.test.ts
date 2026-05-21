import { describe, expect, it } from 'vitest'
import { filterSlashCommands, findSlashTrigger, replaceSlashToken } from './slashCompletion'

const commands = [
  { name: '/help', description: 'Show help', category: 'local' },
  { name: '/plan', description: 'Plan mode', category: 'session' },
  { name: '/model', description: 'Switch model', category: 'session' },
]

describe('slashCompletion', () => {
  it('finds beginning and whitespace-delimited slash tokens', () => {
    expect(findSlashTrigger('/', 1)).toEqual({ slashPos: 0, filter: '' })
    expect(findSlashTrigger('ask /pl', 7)).toEqual({ slashPos: 4, filter: 'pl' })
  })

  it('rejects paths and completed tokens with spaces', () => {
    expect(findSlashTrigger('/workspace/Aether', '/workspace/Aether'.length)).toBeNull()
    expect(findSlashTrigger('/plan add auth', '/plan add auth'.length)).toBeNull()
  })

  it('filters and replaces commands', () => {
    expect(filterSlashCommands(commands, 'pl').map((item) => item.name)).toEqual(['/plan'])

    expect(replaceSlashToken('ask /pl later', 7, '/plan')).toEqual({
      value: 'ask /plan  later',
      cursorPosition: 10,
    })
  })
})
