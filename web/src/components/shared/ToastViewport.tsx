import { X } from 'lucide-react'
import { useToastStore } from '../../stores/toastStore'

export function ToastViewport() {
  const toasts = useToastStore((state) => state.toasts)
  const dismiss = useToastStore((state) => state.dismiss)
  if (toasts.length === 0) return null
  return (
    <div className="toast-viewport" aria-live="polite" aria-label="Notifications">
      {toasts.map((toast) => (
        <div className={'toast toast-' + toast.tone} role="status" key={toast.id}>
          <span>{toast.message}</span>
          <button type="button" aria-label="Dismiss notification" onClick={() => dismiss(toast.id)}>
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  )
}
