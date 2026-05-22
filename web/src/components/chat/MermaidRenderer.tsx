import DOMPurify from 'dompurify'
import { AlertTriangle, GitBranch, LoaderCircle, Maximize2, Minus, Plus, RotateCcw, X } from 'lucide-react'
import mermaid from 'mermaid'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CopyButton } from '../shared/CopyButton'
import { CodeBlock } from './blocks/CodeBlock'

type Props = {
  code: string
}

type SvgMetrics = {
  width: number
  height: number
}

type DragState = {
  pointerId: number
  startX: number
  startY: number
  scrollLeft: number
  scrollTop: number
}

let initialized = false
let idCounter = 0

const MIN_ZOOM = 0.5
const MAX_ZOOM = 3
const ZOOM_STEP = 0.25

function initializeMermaid() {
  if (initialized) return
  mermaid.initialize({
    startOnLoad: false,
    theme: 'base',
    securityLevel: 'strict',
    suppressErrorRendering: true,
    fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
    themeVariables: {
      background: '#ffffff',
      primaryColor: '#eaf2ff',
      primaryBorderColor: '#7aa7ff',
      primaryTextColor: '#172033',
      lineColor: '#65758b',
      secondaryColor: '#f1f5f9',
      tertiaryColor: '#f8fafc',
      noteBkgColor: '#fff7d6',
      noteTextColor: '#172033',
    },
  })
  initialized = true
}

function sanitizeSvg(svg: string): string {
  return DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true } })
}

function parseSvgMetrics(svg: string): SvgMetrics | null {
  const viewBoxMatch = svg.match(/viewBox="([^"]+)"/i)
  if (viewBoxMatch?.[1]) {
    const values = viewBoxMatch[1].split(/[\s,]+/).map((part) => Number.parseFloat(part))
    if (values.length === 4 && values.every((value) => Number.isFinite(value))) {
      const width = values[2]
      const height = values[3]
      if (width && height) return { width, height }
    }
  }

  const widthMatch = svg.match(/\bwidth="([0-9.]+)(?:px)?"/i)
  const heightMatch = svg.match(/\bheight="([0-9.]+)(?:px)?"/i)
  if (!widthMatch?.[1] || !heightMatch?.[1]) return null
  const width = Number.parseFloat(widthMatch[1])
  const height = Number.parseFloat(heightMatch[1])
  return Number.isFinite(width) && Number.isFinite(height) ? { width, height } : null
}

function clampZoom(value: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value))
}

function pointerPosition(event: Pick<React.PointerEvent<HTMLDivElement>, 'clientX' | 'clientY' | 'pageX' | 'pageY'>) {
  const x = Number.isFinite(event.clientX) ? event.clientX : event.pageX
  const y = Number.isFinite(event.clientY) ? event.clientY : event.pageY
  return {
    x: Number.isFinite(x) ? x : 0,
    y: Number.isFinite(y) ? y : 0,
  }
}

export function MermaidRenderer({ code }: Props) {
  const previewViewportRef = useRef<HTMLDivElement>(null)
  const dragStateRef = useRef<DragState | null>(null)
  const [svg, setSvg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [previewZoom, setPreviewZoom] = useState(1)
  const [isDragging, setIsDragging] = useState(false)

  useEffect(() => {
    let cancelled = false
    initializeMermaid()
    setSvg(null)
    setError(null)

    const id = `aether-mermaid-${++idCounter}`
    mermaid.render(id, code).then(
      ({ svg: renderedSvg }) => {
        if (cancelled) return
        setSvg(sanitizeSvg(renderedSvg))
        setError(null)
      },
      (reason) => {
        if (cancelled) return
        setSvg(null)
        setError(String(reason?.message || reason || 'Unable to render Mermaid diagram.'))
      },
    )

    return () => {
      cancelled = true
    }
  }, [code])

  useEffect(() => {
    if (!previewOpen) {
      setPreviewZoom(1)
      setIsDragging(false)
      dragStateRef.current = null
    }
  }, [previewOpen])

  const metrics = useMemo(() => (svg ? parseSvgMetrics(svg) : null), [svg])
  const previewCanvasStyle = metrics
    ? { width: `${metrics.width * previewZoom}px`, height: `${metrics.height * previewZoom}px` }
    : undefined

  const stopDragging = useCallback(() => {
    const viewport = previewViewportRef.current
    const dragState = dragStateRef.current
    if (viewport && dragState) {
      try {
        viewport.releasePointerCapture(dragState.pointerId)
      } catch {
        // Synthetic events can release capture out of order in tests.
      }
    }
    dragStateRef.current = null
    setIsDragging(false)
  }, [])

  useEffect(() => stopDragging, [stopDragging])

  const handlePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return
    const viewport = previewViewportRef.current
    if (!viewport) return
    const { x, y } = pointerPosition(event)
    dragStateRef.current = {
      pointerId: event.pointerId,
      startX: x,
      startY: y,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop,
    }
    setIsDragging(true)
    viewport.setPointerCapture(event.pointerId)
  }, [])

  const handlePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const viewport = previewViewportRef.current
    const dragState = dragStateRef.current
    if (!viewport || !dragState || dragState.pointerId !== event.pointerId) return
    event.preventDefault()
    const { x, y } = pointerPosition(event)
    viewport.scrollLeft = dragState.scrollLeft - (x - dragState.startX)
    viewport.scrollTop = dragState.scrollTop - (y - dragState.startY)
  }, [])

  const handlePointerUp = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (dragStateRef.current?.pointerId !== event.pointerId) return
    stopDragging()
  }, [stopDragging])

  const handleWheel = useCallback((event: React.WheelEvent<HTMLDivElement>) => {
    if (!event.ctrlKey && !event.metaKey) return
    event.preventDefault()
    setPreviewZoom((value) => clampZoom(value + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP)))
  }, [])

  if (error) {
    return (
      <section className="mermaid-card mermaid-card-error" aria-label="Mermaid render error">
        <header className="mermaid-card-header">
          <span><AlertTriangle aria-hidden="true" size={14} /> Mermaid error</span>
          <CopyButton text={code} label="Copy Mermaid source" className="mermaid-copy" />
        </header>
        <p>{error}</p>
        <CodeBlock code={code} language="mermaid" title="Mermaid source" wrap />
      </section>
    )
  }

  if (!svg) {
    return (
      <section className="mermaid-card mermaid-card-loading" aria-label="Rendering Mermaid diagram">
        <LoaderCircle aria-hidden="true" size={16} />
        <span>Rendering diagram...</span>
      </section>
    )
  }

  return (
    <>
      <section className="mermaid-card" aria-label="Mermaid diagram">
        <header className="mermaid-card-header">
          <span><GitBranch aria-hidden="true" size={14} /> Mermaid</span>
          <div>
            <button type="button" className="mermaid-action" onClick={() => setPreviewOpen(true)}>
              <Maximize2 aria-hidden="true" size={13} />
              <span>Preview</span>
            </button>
            <CopyButton text={code} label="Copy Mermaid source" className="mermaid-copy" />
          </div>
        </header>
        <button type="button" className="mermaid-stage" aria-label="Open Mermaid preview" onClick={() => setPreviewOpen(true)} dangerouslySetInnerHTML={{ __html: svg }} />
      </section>
      {previewOpen ? (
        <div className="mermaid-preview-backdrop" role="dialog" aria-modal="true" aria-label="Mermaid diagram preview">
          <div className="mermaid-preview-modal">
            <header>
              <strong><GitBranch aria-hidden="true" size={16} /> Mermaid Diagram</strong>
              <div>
                <button type="button" aria-label="Zoom out" onClick={() => setPreviewZoom((value) => clampZoom(value - ZOOM_STEP))}>
                  <Minus aria-hidden="true" size={14} />
                </button>
                <button type="button" className="mermaid-zoom-value" onClick={() => setPreviewZoom(1)}>
                  <RotateCcw aria-hidden="true" size={13} />
                  <span>{Math.round(previewZoom * 100)}%</span>
                </button>
                <button type="button" aria-label="Zoom in" onClick={() => setPreviewZoom((value) => clampZoom(value + ZOOM_STEP))}>
                  <Plus aria-hidden="true" size={14} />
                </button>
                <CopyButton text={code} label="Copy Mermaid source" className="mermaid-copy" />
                <button type="button" aria-label="Close Mermaid preview" onClick={() => setPreviewOpen(false)}>
                  <X aria-hidden="true" size={15} />
                </button>
              </div>
            </header>
            <div
              ref={previewViewportRef}
              data-testid="mermaid-preview-viewport"
              className="mermaid-preview-viewport"
              data-dragging={isDragging ? 'true' : 'false'}
              onWheel={handleWheel}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              onPointerLeave={handlePointerUp}
            >
              <div className="mermaid-preview-canvas" style={previewCanvasStyle} dangerouslySetInnerHTML={{ __html: svg }} />
            </div>
            <p>Drag to pan. Use the controls to zoom, or hold Ctrl/Command while scrolling.</p>
          </div>
        </div>
      ) : null}
    </>
  )
}
