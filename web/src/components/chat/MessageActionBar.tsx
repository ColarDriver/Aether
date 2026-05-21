import { CopyButton } from '../shared/CopyButton'

type Props = {
  copyText?: string
  copyLabel: string
  align?: 'start' | 'end'
}

export function MessageActionBar({ copyText, copyLabel, align = 'start' }: Props) {
  if (!copyText?.trim()) return null

  return (
    <div className={'message-action-bar message-action-bar-' + align} data-message-actions="">
      <CopyButton
        text={copyText}
        label={copyLabel}
        displayLabel="Copy"
        displayCopiedLabel="Copied"
        className="message-action-button"
      />
    </div>
  )
}
