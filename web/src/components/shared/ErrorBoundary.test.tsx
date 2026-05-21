// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ErrorBoundary } from "./ErrorBoundary"

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function BrokenChild() {
  throw new Error("render exploded")
  return <div />
}

describe("ErrorBoundary", () => {
  it("renders children when no rendering error occurs", () => {
    render(
      <ErrorBoundary>
        <div>Console ready</div>
      </ErrorBoundary>,
    )

    expect(screen.getByText("Console ready")).toBeTruthy()
  })

  it("renders fallback details when a child crashes", () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined)

    render(
      <ErrorBoundary>
        <BrokenChild />
      </ErrorBoundary>,
    )

    expect(screen.getByRole("alert")).toBeTruthy()
    expect(screen.getByText("Something went wrong.")).toBeTruthy()
    expect(screen.getByText("render exploded")).toBeTruthy()
  })
})
