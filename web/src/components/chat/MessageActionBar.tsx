import { Edit3, GitBranch, Quote, RotateCcw, RotateCw, Undo2 } from 'lucide-react'
import type { ComponentType } from 'react'
import { CopyButton } from '../shared/CopyButton'

export type MessageActionKind = 'quote' | 'edit' | 'retry' | 'fork' | 'rewind' | 'restore' | 'undo'

export type MessageAction = {
  kind: MessageActionKind
  label: string
  onClick: () => void
  disabled?: boolean
  reason?: string | null
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
  fork: GitBranch,
  rewind: Undo2,
  restore: RotateCw,
  undo: RotateCcw,
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
        const title = action.reason ? action.label + ': ' + action.reason : action.label
        return (
          <button
            type="button"
            className="message-action-button"
            key={action.kind + '-' + action.label}
            aria-label={action.label}
            title={title}
            disabled={action.disabled}
            onClick={action.onClick}
          >
            <Icon size={13} aria-hidden={true} />
            <span>{action.label}</span>
          </button>
        )
      })}
    </div>
  )
}
