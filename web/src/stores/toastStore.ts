import { create } from 'zustand'

export type ToastTone = 'success' | 'error' | 'info'

export type ToastItem = {
  id: number
  message: string
  tone: ToastTone
}

type ToastState = {
  toasts: ToastItem[]
  notify: (message: string, tone?: ToastTone) => number
  dismiss: (id: number) => void
  clear: () => void
}

let nextToastId = 1

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  notify: (message, tone = 'info') => {
    const id = nextToastId++
    set((state) => ({ toasts: [...state.toasts, { id, message, tone }] }))
    window.setTimeout(() => useToastStore.getState().dismiss(id), 4000)
    return id
  },
  dismiss: (id) => set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
  clear: () => set({ toasts: [] }),
}))
