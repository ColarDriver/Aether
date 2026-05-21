// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useToastStore } from "../../stores/toastStore"
import { ToastViewport } from "./ToastViewport"

afterEach(() => {
  cleanup()
  useToastStore.getState().clear()
  vi.useRealTimers()
})

describe("ToastViewport", () => {
  it("renders notifications and dismisses them from the viewport", () => {
    render(<ToastViewport />)

    act(() => {
      useToastStore.getState().notify("Saved environment", "success")
    })

    expect(screen.getByRole("status").textContent).toContain("Saved environment")

    fireEvent.click(screen.getByRole("button", { name: "Dismiss notification" }))

    expect(screen.queryByText("Saved environment")).toBeNull()
  })

  it("auto-dismisses notifications after the timeout", () => {
    vi.useFakeTimers()
    render(<ToastViewport />)

    act(() => {
      useToastStore.getState().notify("Request failed", "error")
    })

    expect(screen.getByText("Request failed")).toBeTruthy()

    act(() => {
      vi.advanceTimersByTime(4000)
    })

    expect(screen.queryByText("Request failed")).toBeNull()
  })
})
