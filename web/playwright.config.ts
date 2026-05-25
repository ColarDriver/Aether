import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, devices } from '@playwright/test'

const localChromium = resolve('../tmp/playwright-local/chrome-linux64/chrome')
const cachedChromium = '/root/.cache/ms-playwright/chromium-1200/chrome-linux64/chrome'
const chromiumExecutable = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
  || (existsSync(localChromium) ? localChromium : '')
  || (existsSync(cachedChromium) ? cachedChromium : '')
const launchOptions = chromiumExecutable ? { executablePath: chromiumExecutable } : undefined
const webServerPort = Number(process.env.PLAYWRIGHT_PORT || 4187)
const webServerUrl = 'http://127.0.0.1:' + webServerPort

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: webServerUrl,
    trace: 'retain-on-failure',
    ...(launchOptions ? { launchOptions } : {}),
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port ' + webServerPort + ' --strictPort',
    url: webServerUrl,
    reuseExistingServer: process.env.PLAYWRIGHT_REUSE_SERVER === '1',
    stdout: 'pipe',
    stderr: 'pipe',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
