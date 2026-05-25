import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

const backend = process.env.AETHER_WEB_BACKEND ?? 'http://127.0.0.1:9120'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: backend,
        ws: true,
      },
    },
  },
  test: {
    exclude: ['node_modules/**', 'dist/**', 'e2e/**', 'playwright-report/**', 'test-results/**'],
  },
})
