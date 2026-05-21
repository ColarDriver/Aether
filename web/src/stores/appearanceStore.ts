import { create } from "zustand"
import { api } from "../api/client"

export type AppearanceThemeId = "light" | "dark" | "terminal"
export type AppearanceLocale = "en" | "zh" | "ja" | "de" | "es" | "fr"

export type AppearanceTheme = {
  id: AppearanceThemeId
  label: string
  description: string
  swatch: [string, string, string]
}

export type AppearanceLanguage = {
  id: AppearanceLocale
  label: string
  shortLabel: string
}

export const APPEARANCE_THEMES: AppearanceTheme[] = [
  {
    id: "light",
    label: "Aether Light",
    description: "Quiet work surface with high contrast panels.",
    swatch: ["#f6f7f9", "#ffffff", "#2563eb"],
  },
  {
    id: "dark",
    label: "Aether Dark",
    description: "Dim neutral console for long sessions.",
    swatch: ["#101114", "#1b1f27", "#7aa2ff"],
  },
  {
    id: "terminal",
    label: "Terminal",
    description: "Dense graphite and green operator palette.",
    swatch: ["#090d0b", "#101914", "#41d67a"],
  },
]

export const APPEARANCE_LANGUAGES: AppearanceLanguage[] = [
  { id: "en", label: "English", shortLabel: "EN" },
  { id: "zh", label: "简体中文", shortLabel: "简" },
  { id: "ja", label: "日本語", shortLabel: "日" },
  { id: "de", label: "Deutsch", shortLabel: "DE" },
  { id: "es", label: "Español", shortLabel: "ES" },
  { id: "fr", label: "Français", shortLabel: "FR" },
]

const THEME_PREF_KEY = "web.theme"
const LOCALE_PREF_KEY = "web.locale"
const THEME_STORAGE_KEY = "aether-web-theme"
const LOCALE_STORAGE_KEY = "aether-web-locale"

export type AppearanceState = {
  theme: AppearanceThemeId
  locale: AppearanceLocale
  isLoaded: boolean
  error: string | null
  bootstrap: () => Promise<void>
  setTheme: (theme: AppearanceThemeId) => Promise<void>
  setLocale: (locale: AppearanceLocale) => Promise<void>
}

export const useAppearanceStore = create<AppearanceState>((set, get) => ({
  theme: readStoredTheme(),
  locale: readStoredLocale(),
  isLoaded: false,
  error: null,
  bootstrap: async () => {
    applyAppearance(get().theme, get().locale)
    try {
      const prefs = await api.prefs()
      const theme = asTheme(prefs[THEME_PREF_KEY]) ?? get().theme
      const locale = asLocale(prefs[LOCALE_PREF_KEY]) ?? get().locale
      writeStoredAppearance(theme, locale)
      applyAppearance(theme, locale)
      set({ theme, locale, isLoaded: true, error: null })
    } catch (error) {
      applyAppearance(get().theme, get().locale)
      set({ isLoaded: true, error: error instanceof Error ? error.message : String(error) })
    }
  },
  setTheme: async (theme) => {
    const locale = get().locale
    writeStoredAppearance(theme, locale)
    applyAppearance(theme, locale)
    set({ theme, error: null })
    try {
      await api.setPref({ key: THEME_PREF_KEY, value: theme })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) })
      throw error
    }
  },
  setLocale: async (locale) => {
    const theme = get().theme
    writeStoredAppearance(theme, locale)
    applyAppearance(theme, locale)
    set({ locale, error: null })
    try {
      await api.setPref({ key: LOCALE_PREF_KEY, value: locale })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) })
      throw error
    }
  },
}))

export function applyAppearance(theme: AppearanceThemeId, locale: AppearanceLocale) {
  if (typeof document === "undefined") return
  document.documentElement.dataset.theme = theme
  document.documentElement.lang = locale
}

function readStoredTheme(): AppearanceThemeId {
  if (typeof localStorage === "undefined") return "light"
  return asTheme(localStorage.getItem(THEME_STORAGE_KEY)) ?? "light"
}

function readStoredLocale(): AppearanceLocale {
  if (typeof localStorage === "undefined") return "en"
  return asLocale(localStorage.getItem(LOCALE_STORAGE_KEY)) ?? "en"
}

function writeStoredAppearance(theme: AppearanceThemeId, locale: AppearanceLocale) {
  if (typeof localStorage === "undefined") return
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme)
    localStorage.setItem(LOCALE_STORAGE_KEY, locale)
  } catch {
    // Best effort only; service-backed prefs remain authoritative.
  }
}

function asTheme(value: unknown): AppearanceThemeId | null {
  return typeof value === "string" && APPEARANCE_THEMES.some((theme) => theme.id === value)
    ? (value as AppearanceThemeId)
    : null
}

function asLocale(value: unknown): AppearanceLocale | null {
  return typeof value === "string" && APPEARANCE_LANGUAGES.some((language) => language.id === value)
    ? (value as AppearanceLocale)
    : null
}
