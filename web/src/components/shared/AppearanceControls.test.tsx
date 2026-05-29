// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "../../api/client"
import { useAppearanceStore } from "../../stores/appearanceStore"
import { useToastStore } from "../../stores/toastStore"
import { AppearanceControls } from "./AppearanceControls"

afterEach(() => {
  cleanup()
  useAppearanceStore.setState({ theme: "light", locale: "en", isLoaded: false, error: null })
  useToastStore.getState().clear()
  vi.restoreAllMocks()
})

describe("AppearanceControls", () => {
  it("renders theme and language selectors and persists changes", async () => {
    const setPref = vi.spyOn(api, "setPref").mockResolvedValue({ ok: true, key: "web.theme" })

    render(<AppearanceControls />)

    expect(screen.getByRole("option", { name: "Aether Studio" })).toBeTruthy()

    fireEvent.change(screen.getByLabelText("Theme"), { target: { value: "dark" } })
    fireEvent.change(screen.getByLabelText("Language"), { target: { value: "zh" } })

    await waitFor(() => expect(setPref).toHaveBeenCalledWith({ key: "web.theme", value: "dark" }))
    expect(setPref).toHaveBeenCalledWith({ key: "web.locale", value: "zh" })
    expect(document.documentElement.dataset.theme).toBe("dark")
    expect(document.documentElement.lang).toBe("zh")
  })
})
