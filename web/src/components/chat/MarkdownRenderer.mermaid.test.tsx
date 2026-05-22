// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("./MermaidRenderer", () => ({
  MermaidRenderer: ({ code }: { code: string }) => <div data-testid="mermaid-renderer">{code}</div>,
}))

import { MarkdownRenderer } from "./MarkdownRenderer"

afterEach(cleanup)

describe("MarkdownRenderer Mermaid routing", () => {
  it("routes Mermaid fenced blocks to the diagram renderer", () => {
    const fence = String.fromCharCode(96, 96, 96)
    render(<MarkdownRenderer text={fence + "mermaid\ngraph TD\nA-->B\n" + fence} />)

    expect(screen.getByTestId("mermaid-renderer").textContent).toContain("graph TD")
    expect(screen.queryByText("mermaid")).toBeNull()
  })

  it("detects unlabeled Mermaid diagrams in plain code fences", () => {
    const fence = String.fromCharCode(96, 96, 96)
    render(<MarkdownRenderer text={fence + "\nsequenceDiagram\nA->>B: hello\n" + fence} />)

    expect(screen.getByTestId("mermaid-renderer").textContent).toContain("sequenceDiagram")
  })
})
