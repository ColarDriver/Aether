import { atom } from 'nanostores'

import type { PermissionToolRequest } from '../gatewayTypes.js'

export interface PermissionRule {
  toolName: string
  pathPrefix?: string
  commandPrefix?: string
  reason?: string
}

export interface PermissionRulesState {
  rules: PermissionRule[]
}

const initialState: PermissionRulesState = { rules: [] }

export const permissionRulesState = atom<PermissionRulesState>(initialState)

export const permissionRulesActions = {
  resetForTests(): void {
    permissionRulesState.set(initialState)
  },

  add(rule: PermissionRule): void {
    const current = permissionRulesState.get()
    // Replace any rule with the same signature so the cache cannot grow
    // unbounded if the user reissues `S` for the same tool/prefix.
    const filtered = current.rules.filter((existing) => !sameRule(existing, rule))
    permissionRulesState.set({ rules: [...filtered, rule] })
  },

  clear(): void {
    permissionRulesState.set({ rules: [] })
  }
}

/**
 * Returns the first cached rule whose tool name matches and whose prefix is
 * a prefix of the corresponding request field.
 *
 * NOTE: this only powers the *default highlighted option* in the modal UI.
 * The server is still authoritative — every permission.request still appears,
 * we just preselect Allow when the cache says we previously approved a
 * matching prefix.
 */
export function matchesRule(
  request: PermissionToolRequest,
  rules: PermissionRule[] = permissionRulesState.get().rules
): PermissionRule | null {
  for (const rule of rules) {
    if (rule.toolName !== request.tool_name) {
      continue
    }
    if (rule.pathPrefix) {
      if (pathFromArguments(request).startsWith(rule.pathPrefix)) {
        return rule
      }
      continue
    }
    if (rule.commandPrefix) {
      if (commandFromArguments(request).startsWith(rule.commandPrefix)) {
        return rule
      }
      continue
    }
    // Tool-name-only rule (no prefix gate) matches every request for that tool.
    return rule
  }
  return null
}

/**
 * Heuristic argument extraction. Most file-targeting tools we ship use one of
 * these argument keys; if none match we return an empty string so prefix
 * comparison falls through to a non-match instead of throwing.
 */
export function pathFromArguments(request: PermissionToolRequest): string {
  const args = request.arguments ?? {}
  for (const key of ['path', 'file', 'filename', 'target', 'destination']) {
    const value = args[key]
    if (typeof value === 'string') {
      return value
    }
  }
  return ''
}

export function commandFromArguments(request: PermissionToolRequest): string {
  const args = request.arguments ?? {}
  const command = args['command']
  if (typeof command === 'string') {
    return command
  }
  return ''
}

function sameRule(a: PermissionRule, b: PermissionRule): boolean {
  return (
    a.toolName === b.toolName &&
    (a.pathPrefix ?? '') === (b.pathPrefix ?? '') &&
    (a.commandPrefix ?? '') === (b.commandPrefix ?? '')
  )
}
