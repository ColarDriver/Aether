import { Component, type ErrorInfo, type ReactNode } from 'react'

export class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; message: string }> {
  state = { hasError: false, message: '' }

  static getDerivedStateFromError(error: unknown) {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : 'Unknown error',
    }
  }

  componentDidCatch(error: unknown, errorInfo: ErrorInfo) {
    console.error('Aether web console crashed', error, errorInfo)
  }

  render() {
    if (!this.state.hasError) return this.props.children
    return (
      <main className="error-boundary" role="alert">
        <section>
          <h1>Something went wrong.</h1>
          <p>The web console hit a rendering error. Reload the page after checking Diagnostics or Logs.</p>
          {this.state.message ? <pre>{this.state.message}</pre> : null}
          <button type="button" onClick={() => window.location.reload()}>Reload</button>
        </section>
      </main>
    )
  }
}
