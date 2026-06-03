type Props = {
  // While a run is active the static "A" stays put and only its edge breathes.
  active?: boolean
  className?: string
}

// The Aether mark: a static geometric "A" (from the wordmark). It never spins —
// when `active`, a soft light breathes around the tile's edge instead.
export function AetherAvatar({ active = false, className }: Props) {
  return (
    <span
      className={'aether-avatar' + (active ? ' aether-avatar-active' : '') + (className ? ' ' + className : '')}
      aria-hidden="true"
    >
      <svg className="aether-avatar-mark" viewBox="0 0 48 48" fill="none">
        <path d="M14 36 L24 13 L34 36" />
        <path d="M18.7 28.6 H29.3" />
      </svg>
      {active ? (
        <svg className="aether-avatar-edge" viewBox="0 0 48 48" fill="none">
          <rect x="2" y="2" width="44" height="44" rx="12" />
        </svg>
      ) : null}
    </span>
  )
}
