import type { SessionInfo, TranscriptMessage } from '../../gatewayTypes.js'
import { chatActions } from '../../store/chatStore.js'
import { OVERLAY_PRIORITY, overlayActions } from '../../store/overlayStore.js'
import { sessionActions } from '../../store/sessionStore.js'
import type { SessionListResult, SessionResumeResult, SlashCommand } from '../dispatcher.js'

export const resumeCommand: SlashCommand = {
  name: '/resume',
  category: 'remote',
  async execute(args, ctx) {
    const prefix = args[0]?.trim()

    // /resume with no argument → interactive picker. The picker resolves the
    // chosen session via session.resume and writes the transcript into the
    // chat store directly so we do not need to thread a callback through
    // applySlashResult.
    if (!prefix) {
      const sessions = await ctx.client.request<SessionListResult>('session.list', {
        limit: 20
      })
      overlayActions.push({
        kind: 'session-picker',
        id: `picker_${Date.now()}`,
        payload: {
          sessions: sessions.sessions,
          async resolveResume(sessionId: string) {
            const result = await ctx.client.request<SessionResumeResult>('session.resume', {
              session_id: sessionId
            })
            return { info: result.info, messages: result.messages }
          },
          onResume(info: SessionInfo, messages: TranscriptMessage[]) {
            sessionActions.setSession(info)
            chatActions.replaceTranscript(messages)
            chatActions.pushNote(
              `resumed session ${info.session_id.slice(0, 8)}`,
              'success'
            )
          }
        },
        createdAt: Date.now(),
        priority: OVERLAY_PRIORITY['session-picker'],
        onDismiss: () => undefined
      })
      return { kind: 'noop' }
    }

    // Mirrors Python `_match_session` in `aether/cli/commands.py:394-404`:
    // an exact session_id match wins outright (no ambiguity checks), then
    // fall back to prefix matching where a unique prefix wins. Without
    // the exact-match branch a user pasting their full session id from
    // `/sessions` would hit the ambiguous-prefix error path when the id
    // matches itself but also prefixes others.
    const sessions = await ctx.client.request<SessionListResult>('session.list', { limit: 100 })
    const exact = sessions.sessions.find((session) => session.session_id === prefix)
    const matches = exact
      ? [exact]
      : sessions.sessions.filter((session) => session.session_id.startsWith(prefix))
    if (matches.length === 0) {
      return { kind: 'note', level: 'warn', text: `session not found: ${prefix}` }
    }
    if (matches.length > 1) {
      return {
        kind: 'note',
        level: 'warn',
        text: `ambiguous session prefix: ${matches.map((match) => match.session_id.slice(0, 8)).join(', ')}`
      }
    }

    const target = matches[0]
    if (!target) {
      return { kind: 'note', level: 'warn', text: `session not found: ${prefix}` }
    }
    const resumed = await ctx.client.request<SessionResumeResult>('session.resume', {
      session_id: target.session_id
    })
    return { kind: 'replace-history', messages: resumed.messages, info: resumed.info }
  }
}
