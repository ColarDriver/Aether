import type { PermissionRequestBlock as PermissionRequest } from '../../../chat-rendering'
import { DiffViewer } from '../DiffViewer'

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
      {block.reason ? <p>{block.reason}</p> : null}
      {block.preview?.command ? <pre className="command-preview">{block.preview.command}</pre> : null}
      {block.preview?.body ? <p>{block.preview.body}</p> : null}
      {block.preview?.diff ? <DiffViewer diff={block.preview.diff} /> : null}
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
