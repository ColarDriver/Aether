import { Check, Copy } from 'lucide-react'
import { useEffect, useState } from 'react'
import { copyTextToClipboard } from './clipboard'

type Props = {
  text: string
  label?: string
  copiedLabel?: string
  displayLabel?: string
  displayCopiedLabel?: string
  className?: string
  disabled?: boolean
}

export function CopyButton({
  text,
  label = 'Copy',
  copiedLabel = 'Copied',
  displayLabel = 'Copy',
  displayCopiedLabel = 'Copied',
  className = '',
  disabled = false,
}: Props) {
  const [copied, setCopied] = useState(false)
  const isDisabled = disabled || !text

  useEffect(() => {
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(false), 1500)
    return () => window.clearTimeout(timer)
  }, [copied])

  const handleCopy = async () => {
    if (isDisabled) return
    const ok = await copyTextToClipboard(text)
    setCopied(ok)
  }

  const currentLabel = copied ? copiedLabel : label
  const buttonText = copied ? displayCopiedLabel : displayLabel
  const Icon = copied ? Check : Copy

  return (
    <button
      type="button"
      className={['copy-button', copied ? 'copy-button-copied' : '', className].filter(Boolean).join(' ')}
      aria-label={currentLabel}
      title={currentLabel}
      disabled={isDisabled}
      onClick={handleCopy}
    >
      <Icon aria-hidden="true" size={13} />
      <span>{buttonText}</span>
    </button>
  )
}
