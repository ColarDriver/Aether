import { Send, Square } from 'lucide-react'
import { useState } from 'react'
import { Button } from '../shared/Button'

type Props = {
  disabled: boolean
  running: boolean
  onSend: (message: string) => void
  onCancel: () => void
}

export function Composer({ disabled, running, onSend, onCancel }: Props) {
  const [value, setValue] = useState('')

  const submit = () => {
    const text = value.trim()
    if (!text || disabled || running) return
    setValue('')
    onSend(text)
  }

  return (
    <div className="composer">
      <textarea
        value={value}
        disabled={disabled}
        placeholder={disabled ? 'Select a session' : 'Ask Aether'}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            submit()
          }
        }}
      />
      {running ? (
        <Button aria-label="Stop run" onClick={onCancel}>
          <Square size={16} />
        </Button>
      ) : (
        <Button aria-label="Send message" onClick={submit} disabled={disabled || !value.trim()}>
          <Send size={16} />
        </Button>
      )}
    </div>
  )
}
