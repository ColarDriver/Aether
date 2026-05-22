// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("mermaid", () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn(async () => ({
      svg: "<svg viewBox=\"0 0 120 60\" role=\"img\"><text>diagram</text></svg>",
    })),
  },
}))

vi.mock("dompurify", () => ({
  default: {
    sanitize: vi.fn((value: string) => value),
  },
}))

import mermaid from "mermaid"
import { MermaidRenderer } from "./MermaidRenderer"

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe("MermaidRenderer", () => {
  it("renders a Mermaid diagram and opens the zoomable preview", async () => {
    render(<MermaidRenderer code={"graph TD\nA-->B"} />)

    expect(screen.getByLabelText("Rendering Mermaid diagram")).toBeTruthy()
    await screen.findByLabelText("Mermaid diagram")
    expect(screen.getByText("Mermaid")).toBeTruthy()
    expect(document.querySelector(".mermaid-stage svg")?.textContent).toBe("diagram")

    fireEvent.click(screen.getByRole("button", { name: "Open Mermaid preview" }))
    expect(screen.getByRole("dialog", { name: "Mermaid diagram preview" })).toBeTruthy()
    expect(screen.getByTestId("mermaid-preview-viewport")).toBeTruthy()
  })

  it("falls back to source when Mermaid rendering fails", async () => {
    vi.mocked(mermaid.render).mockRejectedValueOnce(new Error("bad syntax"))

    render(<MermaidRenderer code={"graph TD\nA-->"} />)

    await waitFor(() => expect(screen.getByRole("region", { name: "Mermaid render error" })).toBeTruthy())
    expect(screen.getByText("bad syntax")).toBeTruthy()
    expect(screen.getByText("Mermaid source")).toBeTruthy()
  })
})
