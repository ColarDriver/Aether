import { ChevronLeft, ChevronRight, FileText, Folder, ImageIcon, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { ChatAttachment } from '../../chat-rendering'

type Props = {
  attachments?: ChatAttachment[]
  align?: 'start' | 'end'
  onRemove?: (index: number) => void
}

type GalleryImage = {
  src: string
  name: string
}

export function AttachmentGallery({ attachments = [], align = 'start', onRemove }: Props) {
  const [activeImageIndex, setActiveImageIndex] = useState<number | null>(null)
  const images = useMemo(
    () => attachments.flatMap((attachment) => {
      const src = imageSource(attachment)
      return src ? [{ src, name: attachmentLabel(attachment, 'image') }] : []
    }),
    [attachments],
  )

  if (attachments.length === 0) return null

  return (
    <>
      <div className={'attachment-gallery attachment-gallery-' + align}>
        {attachments.map((attachment, index) => {
          const src = imageSource(attachment)
          const key = (attachment.path || attachment.url || attachment.name || attachment.type) + '-' + index
          if (attachment.type === 'image' && src) {
            const imageIndex = images.findIndex((image) => image.src === src)
            return (
              <div className="attachment-item" key={key}>
                <button
                  type="button"
                  className="attachment-image"
                  onClick={() => setActiveImageIndex(imageIndex >= 0 ? imageIndex : null)}
                >
                  <img src={src} alt={attachmentLabel(attachment, 'image')} loading="lazy" />
                  <span>{attachmentLabel(attachment, 'image')}</span>
                </button>
                {onRemove ? <AttachmentRemoveButton index={index} name={attachmentLabel(attachment, 'image')} onRemove={onRemove} /> : null}
              </div>
            )
          }
          return (
            <div className="attachment-item" key={key}>
              <AttachmentChip attachment={attachment} />
              {onRemove ? <AttachmentRemoveButton index={index} name={attachmentLabel(attachment, attachment.type)} onRemove={onRemove} /> : null}
            </div>
          )
        })}
      </div>
      {activeImageIndex !== null ? (
        <ImageGalleryModal
          activeIndex={activeImageIndex}
          images={images}
          onClose={() => setActiveImageIndex(null)}
          onSelect={setActiveImageIndex}
        />
      ) : null}
    </>
  )
}

function AttachmentRemoveButton({
  index,
  name,
  onRemove,
}: {
  index: number
  name: string
  onRemove: (index: number) => void
}) {
  return (
    <button
      type="button"
      className="attachment-remove"
      aria-label={'Remove ' + name}
      onClick={() => onRemove(index)}
    >
      <X aria-hidden="true" size={12} />
    </button>
  )
}

function AttachmentChip({ attachment }: { attachment: ChatAttachment }) {
  const Icon = attachment.isDirectory ? Folder : attachment.type === 'image' ? ImageIcon : FileText
  return (
    <div className="attachment-chip">
      <Icon aria-hidden="true" size={15} />
      <span>{attachmentLabel(attachment, attachment.type)}</span>
      {attachment.path ? <small>{attachment.path}</small> : null}
      {lineRangeLabel(attachment) ? <em>{lineRangeLabel(attachment)}</em> : null}
    </div>
  )
}

function ImageGalleryModal({
  images,
  activeIndex,
  onClose,
  onSelect,
}: {
  images: GalleryImage[]
  activeIndex: number
  onClose: () => void
  onSelect: (index: number) => void
}) {
  const activeImage = images[activeIndex]

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (images.length <= 1) return
      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        onSelect((activeIndex - 1 + images.length) % images.length)
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        onSelect((activeIndex + 1) % images.length)
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [activeIndex, images.length, onClose, onSelect])

  if (!activeImage) return null

  return (
    <div className="image-gallery-backdrop" role="dialog" aria-modal="true" aria-label={activeImage.name}>
      <div className="image-gallery-modal">
        <header>
          <div>
            <strong>{activeImage.name}</strong>
            <span>{activeIndex + 1} / {images.length}</span>
          </div>
          <button type="button" aria-label="Close image preview" onClick={onClose}>
            <X aria-hidden="true" size={16} />
          </button>
        </header>
        <div className="image-gallery-stage">
          {images.length > 1 ? (
            <button type="button" aria-label="Previous image" onClick={() => onSelect((activeIndex - 1 + images.length) % images.length)}>
              <ChevronLeft aria-hidden="true" size={18} />
            </button>
          ) : null}
          <img src={activeImage.src} alt={activeImage.name} />
          {images.length > 1 ? (
            <button type="button" aria-label="Next image" onClick={() => onSelect((activeIndex + 1) % images.length)}>
              <ChevronRight aria-hidden="true" size={18} />
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}

function imageSource(attachment: ChatAttachment): string | null {
  if (attachment.type !== 'image') return null
  if (attachment.data) {
    if (attachment.data.startsWith('data:')) return attachment.data
    return attachment.mimeType ? 'data:' + attachment.mimeType + ';base64,' + attachment.data : attachment.data
  }
  return attachment.url ?? null
}

function attachmentLabel(attachment: ChatAttachment, fallback: string): string {
  if (attachment.name?.trim()) return attachment.name
  if (attachment.path) return attachment.path.split('/').filter(Boolean).pop() || attachment.path
  return fallback
}

function lineRangeLabel(attachment: ChatAttachment): string {
  if (!attachment.lineStart) return ''
  if (attachment.lineEnd && attachment.lineEnd !== attachment.lineStart) {
    return 'L' + attachment.lineStart + '-L' + attachment.lineEnd
  }
  return 'L' + attachment.lineStart
}
