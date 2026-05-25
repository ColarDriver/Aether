import { TriangleAlert } from 'lucide-react'
import { useEffect, useId, useRef } from 'react'

type Props = {
  title: string
  description: string
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: () => void
  onCancel: () => void
}

export function ConfirmDialog({
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
}: Props) {
  const titleId = useId()
  const descriptionId = useId()
  const cancelRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    cancelRef.current?.focus()
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onCancel()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onCancel])

  return (
    <div className="modal-backdrop" role="presentation">
      <section
        className="prompt-modal confirm-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <header>
          <span className="prompt-modal-icon confirm-modal-icon-danger" aria-hidden="true">
            <TriangleAlert size={17} />
          </span>
          <div className="prompt-modal-title">
            <strong id={titleId}>{title}</strong>
            <span>destructive action</span>
          </div>
        </header>
        <div className="prompt-body confirm-modal-body">
          <p id={descriptionId} className="confirm-modal-description">{description}</p>
        </div>
        <footer>
          <button ref={cancelRef} type="button" className="prompt-action" onClick={onCancel}>{cancelLabel}</button>
          <button type="button" className="prompt-action prompt-action-danger-primary" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </footer>
      </section>
    </div>
  )
}

