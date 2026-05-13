import { beforeEach, describe, expect, it } from 'vitest'

import type { PermissionToolRequest } from '../gatewayTypes.js'
import {
  matchesRule,
  permissionRulesActions,
  permissionRulesState,
  type PermissionRule
} from '../store/permissionRulesStore.js'

function makeRequest(overrides: Partial<PermissionToolRequest> = {}): PermissionToolRequest {
  return {
    tool_call_id: 'tc',
    tool_name: 'shell',
    arguments: {},
    category: 'shell',
    risk: 'medium',
    allow_session: true,
    ...overrides
  }
}

describe('permissionRulesStore', () => {
  beforeEach(() => {
    permissionRulesActions.resetForTests()
  })

  it('starts empty', () => {
    expect(permissionRulesState.get().rules).toEqual([])
  })

  it('adds rules and replaces duplicates by signature', () => {
    permissionRulesActions.add({ toolName: 'shell', commandPrefix: 'ls' })
    permissionRulesActions.add({ toolName: 'shell', commandPrefix: 'ls', reason: 'updated' })

    const rules = permissionRulesState.get().rules
    expect(rules).toHaveLength(1)
    expect(rules[0]?.reason).toBe('updated')
  })

  it('keeps separate entries for distinct signatures', () => {
    permissionRulesActions.add({ toolName: 'shell', commandPrefix: 'ls' })
    permissionRulesActions.add({ toolName: 'shell', commandPrefix: 'cat' })
    permissionRulesActions.add({ toolName: 'file_edit', pathPrefix: 'src/' })

    expect(permissionRulesState.get().rules).toHaveLength(3)
  })

  it('matches a path-prefix rule against a request that targets a nested file', () => {
    const rule: PermissionRule = { toolName: 'file_edit', pathPrefix: 'src/' }
    permissionRulesActions.add(rule)

    const hit = matchesRule(makeRequest({ tool_name: 'file_edit', arguments: { path: 'src/foo/bar.ts' } }))
    expect(hit).toEqual(rule)
  })

  it('does not match a path-prefix rule when the request targets a different directory', () => {
    permissionRulesActions.add({ toolName: 'file_edit', pathPrefix: 'src/' })

    const hit = matchesRule(makeRequest({ tool_name: 'file_edit', arguments: { path: 'tests/foo.ts' } }))
    expect(hit).toBeNull()
  })

  it('matches a command-prefix rule', () => {
    permissionRulesActions.add({ toolName: 'shell', commandPrefix: 'ls' })
    const hit = matchesRule(makeRequest({ arguments: { command: 'ls -la /tmp' } }))
    expect(hit?.commandPrefix).toBe('ls')
  })

  it('matches a tool-name-only rule against any request for that tool', () => {
    permissionRulesActions.add({ toolName: 'file_read' })
    const hit = matchesRule(makeRequest({ tool_name: 'file_read', arguments: {} }))
    expect(hit?.toolName).toBe('file_read')
  })

  it('returns null when the tool name does not match', () => {
    permissionRulesActions.add({ toolName: 'file_edit', pathPrefix: 'src/' })
    const hit = matchesRule(makeRequest({ tool_name: 'shell', arguments: { command: 'ls' } }))
    expect(hit).toBeNull()
  })
})
