import DOMPurify from 'dompurify'
import { useEffect, useState } from 'react'

type Props = {
  source: string
  display?: boolean
}

type RenderState = {
  html: string
  failed: boolean
  loading: boolean
}

type KatexApi = typeof import('katex')

let katexLoadPromise: Promise<KatexApi> | null = null

async function loadKatex(): Promise<KatexApi> {
  katexLoadPromise ??= Promise.all([
    import('katex'),
    import('katex/dist/katex.min.css'),
  ]).then(([module]) => module)
  return katexLoadPromise
}

function renderMath(katex: KatexApi, source: string, display: boolean): { html: string; failed: boolean } {
  try {
    const html = katex.default.renderToString(source, {
      displayMode: display,
      throwOnError: false,
      trust: false,
      strict: 'ignore',
      output: 'html',
    })
    return { html: DOMPurify.sanitize(html), failed: false }
  } catch {
    return { html: '', failed: true }
  }
}

export function MathRenderer({ source, display = false }: Props) {
  const [rendered, setRendered] = useState<RenderState>({ html: '', failed: false, loading: true })

  useEffect(() => {
    let cancelled = false
    setRendered({ html: '', failed: false, loading: true })

    loadKatex().then(
      (katex) => {
        if (cancelled) return
        setRendered({ ...renderMath(katex, source, display), loading: false })
      },
      () => {
        if (cancelled) return
        setRendered({ html: '', failed: true, loading: false })
      },
    )

    return () => {
      cancelled = true
    }
  }, [display, source])

  if (rendered.loading) {
    const Tag = display ? 'div' : 'span'
    return <Tag className={display ? 'math-renderer math-renderer-display math-renderer-loading' : 'math-renderer math-renderer-inline math-renderer-loading'}>{source}</Tag>
  }

  if (rendered.failed || !rendered.html) {
    const Tag = display ? 'div' : 'code'
    return <Tag className={display ? 'math-renderer math-renderer-display math-renderer-fallback' : 'math-renderer math-renderer-inline math-renderer-fallback'}>{source}</Tag>
  }

  if (display) {
    return <div className="math-renderer math-renderer-display" dangerouslySetInnerHTML={{ __html: rendered.html }} />
  }

  return <span className="math-renderer math-renderer-inline" dangerouslySetInnerHTML={{ __html: rendered.html }} />
}
