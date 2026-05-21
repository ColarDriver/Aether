import { Languages, Palette } from "lucide-react"
import {
  APPEARANCE_LANGUAGES,
  APPEARANCE_THEMES,
  type AppearanceLocale,
  type AppearanceThemeId,
  useAppearanceStore,
} from "../../stores/appearanceStore"
import { useToastStore } from "../../stores/toastStore"

type Props = {
  compact?: boolean
}

export function AppearanceControls({ compact = false }: Props) {
  const theme = useAppearanceStore((state) => state.theme)
  const locale = useAppearanceStore((state) => state.locale)
  const setTheme = useAppearanceStore((state) => state.setTheme)
  const setLocale = useAppearanceStore((state) => state.setLocale)
  const notify = useToastStore((state) => state.notify)

  const updateTheme = (nextTheme: AppearanceThemeId) => {
    void setTheme(nextTheme).catch((error: unknown) => notify(error instanceof Error ? error.message : String(error), "error"))
  }

  const updateLocale = (nextLocale: AppearanceLocale) => {
    void setLocale(nextLocale).catch((error: unknown) => notify(error instanceof Error ? error.message : String(error), "error"))
  }

  return (
    <div className={compact ? "appearance-controls appearance-controls-compact" : "appearance-controls"}>
      <label className="appearance-select">
        <Palette size={14} />
        <span className="sr-only">Theme</span>
        <select aria-label="Theme" value={theme} onChange={(event) => updateTheme(event.target.value as AppearanceThemeId)}>
          {APPEARANCE_THEMES.map((item) => (
            <option key={item.id} value={item.id}>{item.label}</option>
          ))}
        </select>
      </label>
      <label className="appearance-select">
        <Languages size={14} />
        <span className="sr-only">Language</span>
        <select aria-label="Language" value={locale} onChange={(event) => updateLocale(event.target.value as AppearanceLocale)}>
          {APPEARANCE_LANGUAGES.map((item) => (
            <option key={item.id} value={item.id}>{compact ? item.shortLabel : item.label}</option>
          ))}
        </select>
      </label>
    </div>
  )
}
