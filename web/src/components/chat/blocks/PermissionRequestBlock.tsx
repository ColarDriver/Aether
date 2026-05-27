import { CheckCircle2, CircleAlert, ShieldQuestion, XCircle } from 'lucide-react'
import type { PermissionRequestBlock as PermissionRequest } from '../../../chat-rendering'
import { PermissionPreviewContent } from '../PromptContent'

type Props = {
  block: PermissionRequest
  onRespond?: (decision: Record<string, unknown>) => void
}

export function PermissionRequestBlock({ block, onRespond }: Props) {
  const Icon = promptStateIcon(block.state)
  return (
    <article className={'chat-block prompt-inline-block prompt-inline-' + block.state}>
      <header>
        <span className="prompt-inline-icon"><Icon size={16} /></span>
        <div>
          <strong>{block.preview?.title || block.toolName}</strong>
          {block.risk || block.category ? <small>{block.risk || block.category}</small> : null}
        </div>
        <span>{block.state}</span>
      </header>
      <PermissionPreviewContent
        args={block.arguments}
        preview={block.preview}
        reason={block.reason}
      />
      {block.statusMessage ? <p className="prompt-inline-status">{block.statusMessage}</p> : null}
      {block.state === 'pending' && onRespond ? (
        <footer>
          <button type="button" onClick={() => onRespond({ type: 'deny' })}>Deny</button>
          <button type="button" onClick={() => onRespond({ type: 'allow_once' })}>Allow once</button>
          {block.allowSession ? (
            <button type="button" onClick={() => onRespond({ type: 'allow_session' })}>Allow session</button>
          ) : null}
        </footer>
      ) : null}
    </article>
  )
}

function promptStateIcon(state: string) {
  if (state === 'allowed') return CheckCircle2
  if (state === 'denied' || state === 'aborted') return XCircle
  if (state === 'expired' || state === 'stale' || state === 'missing' || state === 'disconnected') return CircleAlert
  return ShieldQuestion
}
