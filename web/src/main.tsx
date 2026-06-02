import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { ErrorBoundary } from './components/shared/ErrorBoundary'
import { startMainThreadStallProbe } from './debug/freezeProbe'
import './styles.css'

startMainThreadStallProbe()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
