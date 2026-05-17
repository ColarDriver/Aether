import { spawn } from 'node:child_process'

import { normaliseMode, sessionActions } from '../../store/sessionStore.js'
import type {
  PlanCurrentResult,
  PlanModeResult,
  SlashCommand,
  SlashCtx
} from '../dispatcher.js'

const NO_PLAN = 'No plan written yet.'
const ALREADY_NO_PLAN = 'Already in plan mode. No plan written yet.'

export const planCommand: SlashCommand = {
  name: '/plan',
  category: 'remote',
  async execute(args, ctx) {
    const sessionId = await ensureSessionId(ctx)
    const description = args.join(' ').trim()

    const currentMode = ctx.getSession().mode
    const wasPlanMode = currentMode === 'plan'
    let enabled = false
    if (!wasPlanMode) {
      const result = await ctx.client.request<PlanModeResult>('plan.mode_set', {
        session_id: sessionId,
        mode: 'plan'
      })
      sessionActions.setMode(normaliseMode(result.mode) ?? 'plan')
      enabled = true
    }

    if (args.length === 1 && args[0] === 'open') {
      if (enabled) {
        return { kind: 'note', level: 'success', text: 'Enabled plan mode' }
      }
      return openCurrentPlan(ctx, sessionId)
    }

    if (description) {
      return {
        kind: 'query',
        query: description,
        level: 'success',
        ...(enabled ? { note: 'Enabled plan mode' } : {})
      }
    }

    if (enabled) {
      return { kind: 'note', level: 'success', text: 'Enabled plan mode' }
    }

    const current = await ctx.client.request<PlanCurrentResult>('plan.current', {
      session_id: sessionId
    })
    sessionActions.setMode(normaliseMode(current.mode) ?? 'plan')
    if (current.has_plan && current.plan_content !== null) {
      return { kind: 'note', level: 'info', text: formatCurrentPlan(current) }
    }
    return { kind: 'note', level: 'info', text: ALREADY_NO_PLAN }
  }
}

async function ensureSessionId(ctx: SlashCtx): Promise<string> {
  const current = ctx.getSession().sessionId
  if (current) {
    return current
  }
  const info = await ctx.createSession()
  sessionActions.setSession(info)
  return info.session_id
}

async function openCurrentPlan(ctx: SlashCtx, sessionId: string) {
  const current = await ctx.client.request<PlanCurrentResult>('plan.current', {
    session_id: sessionId
  })
  sessionActions.setMode(normaliseMode(current.mode) ?? ctx.getSession().mode)
  if (!current.has_plan) {
    return { kind: 'note' as const, level: 'warn' as const, text: NO_PLAN }
  }
  if (!current.plan_path) {
    return {
      kind: 'note' as const,
      level: 'error' as const,
      text: 'Could not open plan: plan path is unavailable'
    }
  }
  try {
    if (ctx.openFile) {
      await ctx.openFile(current.plan_path)
    } else {
      openPathInEditor(current.plan_path)
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return {
      kind: 'note' as const,
      level: 'error' as const,
      text: `Could not open plan: ${message}`
    }
  }
  return {
    kind: 'note' as const,
    level: 'success' as const,
    text: `Opened plan: ${current.plan_path}`
  }
}

export function openPathInEditor(path: string): void {
  const editor = process.env.AETHER_EDITOR || process.env.VISUAL || process.env.EDITOR
  if (!editor?.trim()) {
    throw new Error('EDITOR is not configured')
  }
  const child = spawn(editor, [path], {
    detached: true,
    shell: true,
    stdio: 'ignore'
  })
  child.unref()
}

function formatCurrentPlan(current: PlanCurrentResult): string {
  const lines = ['Current Plan']
  if (current.plan_path) {
    lines.push(current.plan_path)
  }
  lines.push('', current.plan_content ?? '')
  if (current.plan_path && hasEditor()) {
    lines.push('', '"/plan open" to edit this plan')
  }
  return lines.join('\n')
}

function hasEditor(): boolean {
  return Boolean((process.env.AETHER_EDITOR || process.env.VISUAL || process.env.EDITOR)?.trim())
}
