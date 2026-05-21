// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "../../api/client"
import { useToastStore } from "../../stores/toastStore"
import { SettingsView } from "./SettingsView"

const config = {
  values: {
    provider: "openai",
    model: "gpt-5.4",
  },
}

const paths = {
  aether_home: "/tmp/aether",
  sessions_dir: "/tmp/aether/sessions",
  prefs_file: "/tmp/aether/prefs.json",
}

afterEach(() => {
  cleanup()
  useToastStore.getState().clear()
  vi.restoreAllMocks()
})

describe("SettingsView", () => {
  it("loads settings and mutates preferences", async () => {
    let prefs: Record<string, unknown> = {
      version: 1,
      "ui.theme": "light",
      last_model_by_provider: { openai: "gpt-5.4" },
    }
    vi.spyOn(api, "config").mockResolvedValue(config)
    vi.spyOn(api, "configPaths").mockResolvedValue(paths)
    vi.spyOn(api, "prefs").mockImplementation(async () => prefs)
    const setPref = vi.spyOn(api, "setPref").mockImplementation(async ({ key, value }) => {
      prefs = { ...prefs, [key]: value }
      return { ok: true, key, value }
    })
    const deletePref = vi.spyOn(api, "deletePref").mockImplementation(async (key) => {
      delete prefs[key]
      return { ok: true, key, deleted: true }
    })

    render(<SettingsView />)

    expect(await screen.findByText("ui.theme")).toBeTruthy()
    expect(screen.getByText("/tmp/aether/prefs.json")).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Edit ui.theme" }))
    fireEvent.change(screen.getByLabelText("Preference value for ui.theme"), { target: { value: "dark" } })
    fireEvent.click(screen.getByTitle("Save preference"))

    await waitFor(() => expect(setPref).toHaveBeenCalledWith({ key: "ui.theme", value: "dark" }))
    expect(await screen.findByText("dark")).toBeTruthy()

    fireEvent.change(screen.getByLabelText("New preference key"), { target: { value: "ui.flag" } })
    fireEvent.change(screen.getByLabelText("New preference value"), { target: { value: "true" } })
    fireEvent.click(screen.getByRole("button", { name: /Add/ }))

    await waitFor(() => expect(setPref).toHaveBeenCalledWith({ key: "ui.flag", value: true }))

    fireEvent.click(screen.getByRole("button", { name: "Delete ui.theme" }))
    await waitFor(() => expect(deletePref).toHaveBeenCalledWith("ui.theme"))
  })
})
