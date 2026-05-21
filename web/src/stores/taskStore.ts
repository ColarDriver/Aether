import { create } from 'zustand'
import { api } from '../api/client'
import type { TaskSummary } from '../api/types'

type TaskState = {
  tasksBySession: Record<string, TaskSummary[]>
  isLoadingBySession: Record<string, boolean>
  errorBySession: Record<string, string | null>
  loadSessionTasks: (sessionId: string) => Promise<void>
}

export const useTaskStore = create<TaskState>((set) => ({
  tasksBySession: {},
  isLoadingBySession: {},
  errorBySession: {},
  loadSessionTasks: async (sessionId) => {
    if (!sessionId) return
    set((state) => ({
      isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: true },
      errorBySession: { ...state.errorBySession, [sessionId]: null },
    }))
    try {
      const result = await api.sessionTasks(sessionId, { limit: 100 })
      set((state) => ({
        tasksBySession: { ...state.tasksBySession, [sessionId]: result.tasks },
        isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
      }))
    } catch (error) {
      set((state) => ({
        isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
        errorBySession: {
          ...state.errorBySession,
          [sessionId]: error instanceof Error ? error.message : String(error),
        },
      }))
    }
  },
}))
