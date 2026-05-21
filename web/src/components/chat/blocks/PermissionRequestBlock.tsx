import type { PermissionRequestBlock as PermissionRequest } from '../../../chat-rendering'
import { PermissionPreviewContent } from '../PromptContent'

type Props = {
  block: PermissionRequest
  onRespond?: (decision: Record<string, unknown>) => void
}

export function PermissionRequestBlock({ block, onRespond }: Props) {
  return (
    <article className="chat-block prompt-inline-block">
      <header>
        <strong>{block.preview?.title || block.toolName}</strong>
        <span>{block.state}</span>
      </header>
      <PermissionPreviewContent
        args={block.arguments}
        preview={block.preview}
        reason={block.reason}
      />
      {block.state === 'pending' && onRespond ? (
        <footer>
          <button type="button" onClick={() => onRespond({ type: 'deny' })}>Deny</button>
          <button type="button" onClick={() => onRespond({ type: 'allow_once' })}>Allow once</button>
          <button type="button" onClick={() => onRespond({ type: 'allow_session' })}>Allow session</button>
        </footer>
      ) : null}
    </article>
  )
}
