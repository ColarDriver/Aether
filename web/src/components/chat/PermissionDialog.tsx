import { ShieldQuestion } from 'lucide-react'
import type { PermissionPrompt } from '../../stores/chatStore'
import { PermissionPreviewContent } from './PromptContent'

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
          <span className="prompt-modal-icon" aria-hidden="true"><ShieldQuestion size={17} /></span>
          <div className="prompt-modal-title">
            <strong>{preview?.title || prompt.request.tool_name || 'Tool permission'}</strong>
            <span>{prompt.request.risk || prompt.request.category || 'permission required'}</span>
          </div>
        </header>
        <div className="prompt-body">
          <PermissionPreviewContent
            args={prompt.request.arguments}
            preview={preview}
            reason={prompt.request.reason}
          />
        </div>
        <footer>
          <button type="button" className="prompt-action prompt-action-danger" onClick={onDeny}>Deny</button>
          {prompt.request.allow_session ? (
            <button type="button" className="prompt-action" onClick={onAllowSession}>Allow session</button>
          ) : null}
          <button type="button" className="prompt-action prompt-action-primary" onClick={onAllow}>Allow once</button>
        </footer>
      </section>
    </div>
  )
}
