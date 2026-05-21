// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "../api/client"
import { applyAppearance, useAppearanceStore } from "./appearanceStore"

function resetAppearanceStore() {
  localStorage.clear()
  document.documentElement.removeAttribute("data-theme")
  document.documentElement.removeAttribute("lang")
  useAppearanceStore.setState({ theme: "light", locale: "en", isLoaded: false, error: null })
}

afterEach(() => {
  resetAppearanceStore()
  vi.restoreAllMocks()
})

describe("appearanceStore", () => {
  it("bootstraps theme and locale from service preferences", async () => {
    vi.spyOn(api, "prefs").mockResolvedValue({ "web.theme": "dark", "web.locale": "zh" })

    await useAppearanceStore.getState().bootstrap()

    expect(useAppearanceStore.getState().theme).toBe("dark")
    expect(useAppearanceStore.getState().locale).toBe("zh")
    expect(document.documentElement.dataset.theme).toBe("dark")
    expect(document.documentElement.lang).toBe("zh")
    expect(localStorage.getItem("aether-web-theme")).toBe("dark")
  })

  it("saves theme and locale changes through prefs", async () => {
    const setPref = vi.spyOn(api, "setPref").mockResolvedValue({ ok: true, key: "web.theme" })

    await useAppearanceStore.getState().setTheme("terminal")
    await useAppearanceStore.getState().setLocale("ja")

    expect(setPref).toHaveBeenCalledWith({ key: "web.theme", value: "terminal" })
    expect(setPref).toHaveBeenCalledWith({ key: "web.locale", value: "ja" })
    expect(document.documentElement.dataset.theme).toBe("terminal")
    expect(document.documentElement.lang).toBe("ja")
  })

  it("applies appearance directly to the document root", () => {
    applyAppearance("dark", "fr")

    expect(document.documentElement.dataset.theme).toBe("dark")
    expect(document.documentElement.lang).toBe("fr")
  })
})
