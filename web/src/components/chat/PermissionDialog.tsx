import type { PermissionPrompt } from '../../stores/chatStore'
import { DiffViewer } from './DiffViewer'

type Props = {
  prompt: PermissionPrompt
  onAllow: () => void
  onAllowSession: () => void
  onDeny: () => void
}

export function PermissionDialog({ prompt, onAllow, onAllowSession, onDeny }: Props) {
  const preview = prompt.request.preview
  return (
    <div className="modal-backdrop" role="presentation">
      <section className="prompt-modal" role="dialog" aria-modal="true" aria-label="Tool permission request">
        <header>
          <strong>{preview?.title || prompt.request.tool_name || 'Tool permission'}</strong>
          <span>{prompt.request.risk || prompt.request.category}</span>
        </header>
        <div className="prompt-body">
          {preview?.subtitle ? <p className="muted">{preview.subtitle}</p> : null}
          {preview?.command ? <pre className="command-preview">{preview.command}</pre> : null}
          {preview?.body ? <p>{preview.body}</p> : null}
          {preview?.diff ? <DiffViewer diff={preview.diff} /> : null}
          <pre>{JSON.stringify(prompt.request.arguments ?? {}, null, 2)}</pre>
        </div>
        <footer>
          <button type="button" onClick={onDeny}>Deny</button>
          {prompt.request.allow_session ? <button type="button" onClick={onAllowSession}>Allow session</button> : null}
          <button type="button" className="primary-action" onClick={onAllow}>Allow once</button>
        </footer>
      </section>
    </div>
  )
}
