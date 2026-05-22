import { Edit3, Quote, RotateCcw } from 'lucide-react'
import type { ComponentType } from 'react'
import { CopyButton } from '../shared/CopyButton'

export type MessageActionKind = 'quote' | 'edit' | 'retry'

export type MessageAction = {
  kind: MessageActionKind
  label: string
  onClick: () => void
  disabled?: boolean
}

type Props = {
  copyText?: string
  copyLabel: string
  align?: 'start' | 'end'
  actions?: MessageAction[]
}

const ACTION_ICONS: Record<MessageActionKind, ComponentType<{ size?: number; 'aria-hidden'?: boolean }>> = {
  quote: Quote,
  edit: Edit3,
  retry: RotateCcw,
}

export function MessageActionBar({ copyText, copyLabel, align = 'start', actions = [] }: Props) {
  const canCopy = Boolean(copyText?.trim())
  const visibleActions = actions.filter((action) => Boolean(action.onClick))
  if (!canCopy && visibleActions.length === 0) return null

  return (
    <div className={'message-action-bar message-action-bar-' + align} data-message-actions="">
      {canCopy ? (
        <CopyButton
          text={copyText || ''}
          label={copyLabel}
          displayLabel="Copy"
          displayCopiedLabel="Copied"
          className="message-action-button"
        />
      ) : null}
      {visibleActions.map((action) => {
        const Icon = ACTION_ICONS[action.kind]
        return (
          <button
            type="button"
            className="message-action-button"
            key={action.kind + '-' + action.label}
            aria-label={action.label}
            title={action.label}
            disabled={action.disabled}
            onClick={action.onClick}
          >
            <Icon size={13} aria-hidden={true} />
            <span>{actionLabel(action.kind)}</span>
          </button>
        )
      })}
    </div>
  )
}

function actionLabel(kind: MessageActionKind): string {
  if (kind === 'quote') return 'Quote'
  if (kind === 'edit') return 'Edit'
  return 'Retry'
}
