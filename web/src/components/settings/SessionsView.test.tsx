// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "../../api/client"
import type { SessionInfo } from "../../api/types"
import { useAppStore } from "../../stores/appStore"
import { useSessionStore } from "../../stores/sessionStore"
import { useToastStore } from "../../stores/toastStore"
import { SessionsView } from "./SessionsView"

const sessionOne: SessionInfo = {
  session_id: "session-one",
  created_at: 1,
  updated_at: 2,
  provider: "codex",
  model: "gpt-5.4",
  message_count: 1,
  summary: "First session",
}

const sessionTwo: SessionInfo = {
  session_id: "session-two",
  created_at: 3,
  updated_at: 4,
  provider: "claude",
  model: "claude-sonnet-4-6",
  message_count: 1,
  summary: "Second session",
}

beforeEach(() => {
  useSessionStore.setState({
    sessions: [sessionOne],
    activeSessionId: "session-one",
    isLoading: false,
    error: null,
  })
  useAppStore.setState({ activeView: "sessions" })
})

afterEach(() => {
  cleanup()
  useToastStore.getState().clear()
  vi.restoreAllMocks()
})

describe("SessionsView", () => {
  it("searches, inspects, resumes, and deletes sessions", async () => {
    vi.spyOn(api, "sessions").mockResolvedValue({ sessions: [sessionTwo, sessionOne] })
    vi.spyOn(api, "sessionDetail").mockImplementation(async (sessionId) => ({
      session_id: sessionId,
      info: sessionId === "session-two" ? sessionTwo : sessionOne,
      messages: [{ role: "user", text: sessionId === "session-two" ? "Second hello" : "First hello" }],
    }))
    const searchSessions = vi.spyOn(api, "searchSessions").mockResolvedValue({ sessions: [sessionTwo] })
    const resumeSession = vi.spyOn(api, "resumeSession").mockResolvedValue({
      session_id: "session-two",
      info: sessionTwo,
      messages: [],
    })
    const deleteSession = vi.spyOn(api, "deleteSession").mockResolvedValue(undefined)

    render(<SessionsView />)

    expect(await screen.findByText("First hello")).toBeTruthy()

    fireEvent.change(screen.getByLabelText("Search session records"), { target: { value: "second" } })
    await waitFor(() => expect(searchSessions).toHaveBeenCalledWith("second"))
    fireEvent.click(await screen.findByText("Second session"))

    expect(await screen.findByText("Second hello")).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Resume session" }))
    await waitFor(() => expect(resumeSession).toHaveBeenCalledWith("session-two"))
    expect(useSessionStore.getState().activeSessionId).toBe("session-two")
    expect(useAppStore.getState().activeView).toBe("chat")

    fireEvent.click(screen.getByRole("button", { name: "Delete session" }))
    expect(screen.getByRole("dialog", { name: "Delete session" })).toBeTruthy()
    expect(screen.getByText('Delete session "Second session"? This removes its conversation context.')).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Delete" }))
    await waitFor(() => expect(deleteSession).toHaveBeenCalledWith("session-two"))
  })

  it("renders persisted tool and diff blocks through the chat timeline", async () => {
    vi.spyOn(api, "sessions").mockResolvedValue({ sessions: [] })
    vi.spyOn(api, "sessionDetail").mockResolvedValue({
      session_id: "session-one",
      info: sessionOne,
      messages: [
        { role: "user", text: "edit file" },
        {
          role: "assistant",
          text: "I will edit it.",
          tool_calls: [
            { id: "call-1", name: "file_edit", arguments: { path: "app.py" } },
          ],
        },
        {
          role: "tool",
          name: "file_edit",
          tool_call_id: "call-1",
          text: "updated",
          metadata: { diff: "@@ -1 +1 @@\n-old\n+new", path: "app.py" },
        },
      ],
    })

    render(<SessionsView />)

    expect(await screen.findByText("file_edit")).toBeTruthy()
    expect(screen.getByRole("table", { name: "Code diff" })).toBeTruthy()
    expect(screen.getByText("new")).toBeTruthy()
  })
})
