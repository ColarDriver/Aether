type Props = {
  label: string
}

export function Spinner({ label }: Props) {
  return (
    <div className="spinner-row" role="status">
      <span className="spinner" />
      <span>{label}</span>
    </div>
  )
}
