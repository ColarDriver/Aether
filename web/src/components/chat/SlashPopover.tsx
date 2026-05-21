import { ChevronRight } from 'lucide-react'
import { forwardRef, useEffect, useImperativeHandle, useMemo, useState } from 'react'
import type { KeyboardEvent } from 'react'
import type { SlashCommandInfo } from '../../api/types'
import { filterSlashCommands, findSlashTrigger, replaceSlashToken } from './slashCompletion'

export type SlashPopoverHandle = {
  handleKey: (event: KeyboardEvent<HTMLTextAreaElement>) => boolean
}

type Props = {
  commands: SlashCommandInfo[]
  value: string
  cursorPosition: number
  disabled?: boolean
  onApply: (value: string, cursorPosition: number) => void
}

export const SlashPopover = forwardRef<SlashPopoverHandle, Props>(function SlashPopover(
  { commands, value, cursorPosition, disabled = false, onApply },
  ref,
) {
  const trigger = useMemo(() => findSlashTrigger(value, cursorPosition), [cursorPosition, value])
  const token = trigger ? value.slice(trigger.slashPos, cursorPosition) : ''
  const [dismissedToken, setDismissedToken] = useState<string | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const items = useMemo(
    () => trigger ? filterSlashCommands(commands, trigger.filter) : [],
    [commands, trigger],
  )
  const visible = !disabled && trigger !== null && items.length > 0 && dismissedToken !== token

  useEffect(() => {
    setSelectedIndex(0)
    if (dismissedToken && dismissedToken !== token) setDismissedToken(null)
  }, [dismissedToken, token])

  const apply = (command: SlashCommandInfo | undefined) => {
    if (!command || !trigger) return
    const next = replaceSlashToken(value, cursorPosition, command.name)
    onApply(next.value, next.cursorPosition)
    setDismissedToken(null)
  }

  useImperativeHandle(ref, () => ({
    handleKey: (event) => {
      if (!visible) return false
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setSelectedIndex((current) => (current + 1) % items.length)
        return true
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        setSelectedIndex((current) => (current - 1 + items.length) % items.length)
        return true
      }
      if (event.key === 'Tab' || event.key === 'Enter') {
        event.preventDefault()
        apply(items[selectedIndex])
        return true
      }
      if (event.key === 'Escape') {
        event.preventDefault()
        setDismissedToken(token)
        return true
      }
      return false
    },
  }), [cursorPosition, items, onApply, selectedIndex, token, trigger, value, visible])

  if (!visible) return null

  return (
    <div className="slash-popover" role="listbox" aria-label="Slash commands">
      {items.map((item, index) => {
        const active = index === selectedIndex
        return (
          <button
            type="button"
            role="option"
            aria-selected={active}
            className={'slash-option' + (active ? ' slash-option-active' : '')}
            key={item.name}
            onClick={() => apply(item)}
            onMouseEnter={() => setSelectedIndex(index)}
          >
            <ChevronRight aria-hidden="true" size={13} />
            <strong>{item.name}</strong>
            <span>{item.description}</span>
            {item.category ? <em>{item.category}</em> : null}
          </button>
        )
      })}
    </div>
  )
})
