import { expect, test, type Locator, type Page, type Route } from '@playwright/test'

const session = {
  session_id: 'session-1',
  created_at: 1_800_000_000,
  updated_at: 1_800_000_100,
  provider: 'openai',
  model: 'gpt-5.4',
  message_count: 6,
  summary: 'Browser acceptance session',
  mode: 'agent',
}

const rootTask = {
  task_id: 'task-1',
  parent_session_id: 'session-1',
  subagent_type: 'reviewer',
  prompt: 'Review auth flow',
  status: 'completed',
  started_at: 1_800_000_010,
  finished_at: 1_800_000_020,
  last_heartbeat: 1_800_000_020,
  model: 'gpt-5.4',
  isolation: 'shared',
  worktree_path: null,
  parent_task_id: null,
  child_depth: 0,
  background: true,
  tool_use_count: 2,
  input_tokens: 100,
  output_tokens: 40,
  iterations: 1,
  summary: 'Task inspected auth flow.',
  error: null,
  result_path: '/tmp/aether/tasks/task-1/result.json',
  output_tail: 'done',
  metadata: { owner: 'browser-smoke' },
}

const childTask = {
  ...rootTask,
  task_id: 'task-2',
  subagent_type: 'implementer',
  prompt: 'Review child implementation',
  parent_task_id: 'task-1',
  child_depth: 1,
  tool_use_count: 1,
  input_tokens: 55,
  output_tokens: 20,
  summary: 'Child reviewed implementation details.',
  result_path: null,
}

test.beforeEach(async ({ page }) => {
  await installMockRunSocket(page)
  await mockAetherApi(page)
})

test('renders persisted chat blocks inside the real app shell', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('main')).toContainText('Browser acceptance session')
  await expect(page.getByText('Inspect authentication flow')).toBeVisible()
  await expect(page.getByText('Auth flow summary')).toBeVisible()
  await expect(page.getByRole('button', { name: /bash.*npm test -- auth/i })).toBeVisible()
  await expect(page.getByText('2 passed')).toBeVisible()
  await expect(page.getByLabel('Changed files')).toContainText('src/auth.ts')
  await expect(page.getByText('Use token storage?')).toBeVisible()
  await expect(page.getByText('Encrypted session storage').first()).toBeVisible()
  await expect(page.getByLabel('Workspace files')).toContainText('README.md')
})

test('resizes sidebar, workspace tree, and dedicated file preview panels from the real app shell', async ({ page }) => {
  await page.setViewportSize({ width: 2200, height: 900 })
  await page.goto('/')

  await expect(page.locator('.workspace-rail-empty-preview')).toHaveCount(0)
  await expect(page.getByText('Select a file')).toHaveCount(0)

  const sidebar = page.locator('.sidebar')
  const workspaceRail = page.getByLabel('Workspace files')
  const sidebarBefore = await boxWidth(sidebar)
  const workspaceRailBefore = await boxWidth(workspaceRail)

  await dragHorizontalSeparator(page, page.getByRole('separator', { name: 'Resize sessions sidebar' }), 80)
  expect(await boxWidth(sidebar)).toBeGreaterThan(sidebarBefore + 48)

  await dragHorizontalSeparator(page, page.getByRole('separator', { name: 'Resize workspace panel' }), -80)
  expect(await boxWidth(workspaceRail)).toBeGreaterThan(workspaceRailBefore + 48)

  await workspaceRail.getByTitle('README.md').click()
  const filePreview = page.getByRole('complementary', { name: 'Workspace file preview' })
  await expect(filePreview).toContainText('Aether workspace readme')
  await expect(filePreview).toContainText('Browser acceptance fixture')
  await filePreview.getByRole('button', { name: 'Edit workspace file' }).click()
  const fileEditor = filePreview.getByLabel('Workspace file editor')
  expect(await boxHeight(fileEditor)).toBeGreaterThan(500)
  await fileEditor.fill('# Aether workspace readme\n\nEdited from the file panel.')
  await filePreview.getByRole('button', { name: 'Save workspace file' }).click()
  await expect(filePreview).toContainText('Edited from the file panel.')

  const previewPosition = await page.evaluate(() => {
    const preview = document.querySelector('.workspace-file-panel')?.getBoundingClientRect()
    const chat = document.querySelector('.chat-surface')?.getBoundingClientRect()
    return { previewLeft: preview?.left ?? 0, previewRight: preview?.right ?? 0, chatLeft: chat?.left ?? 0 }
  })
  expect(previewPosition.previewLeft).toBeLessThan(previewPosition.chatLeft)
  expect(previewPosition.previewRight).toBeLessThanOrEqual(previewPosition.chatLeft + 8)

  const filePreviewBefore = await boxWidth(filePreview)
  await dragHorizontalSeparator(page, page.getByRole('separator', { name: 'Resize workspace file preview' }), 420)
  const filePreviewAfter = await boxWidth(filePreview)
  const chatAfter = await boxWidth(page.locator('.chat-surface'))
  expect(filePreviewAfter).toBeGreaterThan(filePreviewBefore + 240)
  expect(filePreviewAfter).toBeGreaterThan(chatAfter)
  await expectNoDocumentHorizontalOverflow(page)
})

test('swaps sessions and workspace panels from the toolbar and persists after reload', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 900 })
  await page.goto('/')

  const appRail = page.getByLabel('Console sections')
  const workspaceRail = page.getByLabel('Workspace files')
  const rightSessionPanel = page.locator('.chat-workbench > .sidebar')

  await expect(workspaceRail).toBeVisible()
  await expect(rightSessionPanel).toHaveCount(0)

  await page.getByRole('button', { name: 'Swap sessions and workspace panels' }).click()

  await expect(rightSessionPanel).toBeVisible()
  const swapped = await page.evaluate(() => {
    const rail = document.querySelector('.app-rail')?.getBoundingClientRect()
    const workspace = document.querySelector('.workspace-rail')?.getBoundingClientRect()
    const main = document.querySelector('main.workspace')?.getBoundingClientRect()
    const rightSidebar = document.querySelector('.chat-workbench > .sidebar')?.getBoundingClientRect()
    const chatSurface = document.querySelector('.chat-surface')?.getBoundingClientRect()
    return {
      railRight: rail?.right ?? 0,
      workspaceLeft: workspace?.left ?? 0,
      workspaceRight: workspace?.right ?? 0,
      mainLeft: main?.left ?? 0,
      rightSidebarLeft: rightSidebar?.left ?? 0,
      chatSurfaceRight: chatSurface?.right ?? 0,
    }
  })
  expect(swapped.workspaceLeft).toBeGreaterThanOrEqual(swapped.railRight - 1)
  expect(swapped.workspaceRight).toBeLessThanOrEqual(swapped.mainLeft + 1)
  expect(swapped.rightSidebarLeft).toBeGreaterThan(swapped.chatSurfaceRight - 8)

  await page.reload()
  await expect(appRail).toBeVisible()
  await expect(workspaceRail).toBeVisible()
  await expect(rightSessionPanel).toBeVisible()
})

test('deletes a sidebar session after confirmation and keeps it gone after reload', async ({ page }) => {
  await page.goto('/')

  await page.getByRole('button', { name: /^Browser acceptance session now/ }).hover()
  await page.getByRole('button', { name: 'Delete session Browser acceptance session' }).click()
  const dialog = page.getByRole('dialog', { name: 'Delete session' })
  await expect(dialog).toContainText('Delete session "Browser acceptance session"?')
  await dialog.getByRole('button', { name: 'Delete' }).click()

  await expect(page.getByText('No sessions yet')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'New chat' })).toBeVisible()
  await expect(page.getByText('Select a session or start a new run')).toBeVisible()

  await page.reload()
  await expect(page.getByText('No sessions yet')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'New chat' })).toBeVisible()
})

test('renders notebook execution metadata and outputs from persisted transcripts', async ({ page }) => {
  await page.goto('/?scenario=notebook')

  const notebook = page.getByRole('region', { name: 'Notebook edit' })
  await expect(notebook).toContainText('analysis.ipynb')
  await expect(notebook).toContainText('ok')
  await expect(notebook).toContainText('#4')
  await expect(notebook).toContainText('1.3s')
  await expect(notebook).toContainText('python3')
  await expect(notebook).toContainText('truncated')
  await expect(notebook.getByLabel('Notebook lifecycle')).toContainText('queued')
  await expect(notebook.getByLabel('Notebook lifecycle')).toContainText('finished')
  await expect(notebook).toContainText('stdout')
  await expect(notebook).toContainText('hello world')
  await expect(notebook).toContainText('ValueError')
  await expect(notebook.getByRole('img', { name: 'image/png' })).toBeVisible()
  await expectNoDocumentHorizontalOverflow(page)
})

test('renders provider web-search metadata and result cards from persisted transcripts', async ({ page }) => {
  await page.goto('/?scenario=web-search')

  const webResults = page.getByRole('region', { name: 'Web results' })
  await expect(webResults).toContainText('brave')
  await expect(webResults).toContainText('Aether provider')
  await expect(webResults).toContainText('Aether docs')
  await expect(webResults).toContainText('Documentation snippet')
  await expect(webResults).toContainText('Provider guide snippet Extra provider context')
  await expect(webResults).toContainText('https://example.com/aether')
  await expectNoDocumentHorizontalOverflow(page)
})

test('renders rich assistant markdown inside the real app shell', async ({ page }) => {
  await page.goto('/?scenario=rich-markdown')

  const main = page.getByRole('main')
  await expect(main.getByRole('heading', { level: 2, name: 'Render acceptance' })).toBeVisible()
  const table = main.getByRole('table')
  await expect(table.getByRole('columnheader', { name: 'Surface' })).toBeVisible()
  await expect(table.getByRole('cell', { name: 'rendered' }).first()).toBeVisible()
  await expect(main.locator('.math-renderer-inline')).toBeVisible()
  await expect(main.locator('.math-renderer-display')).toBeVisible()
  await expect(main.getByText('typescript')).toBeVisible()
  await expect(main.locator('.syntax-keyword').filter({ hasText: 'const' }).first()).toBeVisible()
  await expect(main.getByRole('region', { name: 'Mermaid diagram' })).toBeVisible({ timeout: 10_000 })
  await expect(main.getByRole('button', { name: 'Open Mermaid preview' })).toBeVisible()
  await expect(main.getByRole('link', { name: 'docs' })).toHaveAttribute('href', 'https://example.com/docs')
  await expectNoDocumentHorizontalOverflow(page)
})

test('keeps rich assistant markdown within a mobile-width viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 760 })
  await page.goto('/?scenario=rich-markdown')

  const main = page.getByRole('main')
  await expect(main.getByRole('heading', { level: 2, name: 'Render acceptance' })).toBeVisible()
  await expect(main.locator('.math-renderer-display')).toBeVisible()
  await expect(main.getByRole('region', { name: 'Mermaid diagram' })).toBeVisible({ timeout: 10_000 })
  await expect(page.getByPlaceholder('Ask Aether')).toBeVisible()

  await expectWithinViewport(page, ['.app-shell', '.chat-surface', '.composer', '.markdown-table-wrap', '.markdown-renderer .code-block', '.mermaid-card'])
  await expectNoDocumentHorizontalOverflow(page)
})

test('renders copy-only local binary artifacts and explicit artifact URLs', async ({ page }) => {
  await page.goto('/?scenario=artifacts')

  const artifacts = page.getByRole('region', { name: 'Tool artifacts' })
  await expect(artifacts).toContainText('weights.bin')
  await expect(artifacts).toContainText('Binary preview unavailable')
  await expect(artifacts).toContainText('report.json')
  await expect(artifacts.getByRole('button', { name: 'Copy weights.bin path' })).toBeVisible()
  await expect(artifacts.getByRole('button', { name: 'Copy report.json contents' })).toBeVisible()
  await expect(artifacts.getByRole('link', { name: 'Open' })).toHaveAttribute('href', 'https://example.com/bundle.zip')
  await expect(artifacts.getByText('bundle.zip', { exact: true })).toBeVisible()
  await expectNoDocumentHorizontalOverflow(page)
})

test('keeps long chat history scrollable while composer and navigation stay visible during live updates', async ({ page }) => {
  await page.setViewportSize({ width: 980, height: 720 })
  await page.goto('/?scenario=long')

  const chatScroll = page.locator('.chat-scroll')
  const composer = page.locator('.composer')
  await expect(page.getByLabel('Console sections')).toBeVisible()
  await expect(composer).toBeVisible()
  await expect(page.getByPlaceholder('Ask Aether')).toBeVisible()
  await expect(page.getByText('Long transcript message 35')).toBeVisible()

  const metrics = await chatScroll.evaluate((element) => ({
    scrollHeight: element.scrollHeight,
    clientHeight: element.clientHeight,
  }))
  expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight + 400)

  await chatScroll.evaluate((element) => {
    element.scrollTop = 0
    element.dispatchEvent(new Event('scroll', { bubbles: true }))
  })
  await expect(page.getByText('Long transcript message 1:', { exact: false })).toBeVisible()
  const beforeRunScrollTop = await chatScroll.evaluate((element) => element.scrollTop)

  await page.getByPlaceholder('Ask Aether').fill('Run live smoke while I read old context')
  await page.getByRole('button', { name: 'Send message' }).click()
  await expect(page.getByText('Live answer from websocket')).toBeAttached()
  await page.waitForTimeout(80)

  const afterRunScrollTop = await chatScroll.evaluate((element) => element.scrollTop)
  expect(afterRunScrollTop).toBeLessThan(120)
  expect(Math.abs(afterRunScrollTop - beforeRunScrollTop)).toBeLessThan(120)
  await expect(composer).toBeVisible()
  await expect(page.getByPlaceholder('Ask Aether')).toBeVisible()
})

test('opens task drill-down with messages, child streams, and result artifacts', async ({ page }) => {
  await page.goto('/')

  await page.getByRole('button', { name: /Subagent tasks/ }).click()
  await page.getByRole('button', { name: 'Open task task-1' }).click()

  const dialog = page.getByRole('dialog', { name: 'Task details' })
  await expect(dialog).toContainText('Review auth flow')
  await expect(dialog.getByLabel('Related tasks')).toContainText('Review child implementation')
  await expect(dialog.getByLabel('Task message stream').first()).toContainText('Queued parent messages')
  await expect(dialog.getByLabel('Task message stream').first()).toContainText('Please inspect auth edge cases')
  await expect(dialog.getByLabel('Task message stream').first()).toContainText('Delivered parent messages')
  await expect(dialog.getByLabel('Task message stream').first()).toContainText('Use stricter validation')
  await expect(dialog.getByLabel('Task message stream').first()).toContainText('assistant observed root task')
  await expect(dialog.getByLabel('Task message stream').first()).toContainText('shell output from task')
  await expect(dialog.getByLabel('Child task message streams')).toContainText('Review child implementation')
  await expect(dialog.getByLabel('Child task message streams')).toContainText('child assistant response')
  await expect(dialog).toContainText('Result artifact')
  await expect(dialog).toContainText('Task completed from artifact')
})

test('keeps primary shell views navigable without blanking the console', async ({ page }) => {
  await page.goto('/')

  await page.getByLabel('Models').click()
  await expect(page.getByRole('main')).toContainText('Provider and model')
  await expect(page.getByRole('main')).toContainText('gpt-5.4')

  await page.getByLabel('Tools').click()
  await expect(page.getByLabel('Tool catalog')).toContainText('read_file')
  await expect(page.getByLabel('Tool details')).toContainText('Read a file')

  await page.getByLabel('Workspace').click()
  await expect(page.getByLabel('Workspace browser')).toContainText('README.md')
  await expect(page.getByLabel('Workspace file preview')).toContainText('Aether workspace readme')

  await page.getByLabel('Settings').click()
  await expect(page.getByRole('main')).toContainText('Settings')
  await expect(page.getByRole('main')).toContainText('AETHER_HOME')
  await expect(page.getByLabel('Available themes')).toBeVisible()

  await page.getByLabel('Chat').click()
  await expect(page.getByPlaceholder('Ask Aether')).toBeVisible()
  await expect(page.getByLabel('Console sections')).toBeVisible()
})

test('contains long terminal output without hiding composer or overflowing narrow layouts', async ({ page }) => {
  await page.setViewportSize({ width: 760, height: 720 })
  await page.goto('/?scenario=long-tool')

  const terminal = page.getByLabel('Terminal output').first()
  await expect(terminal).toBeVisible()
  await expect(terminal).toContainText('terminal line 080')
  await expect(page.getByRole('button', { name: /Show .* more lines/ })).toBeVisible()

  const outputMetrics = await terminal.locator('.terminal-chrome-output').evaluate((element) => ({
    scrollHeight: element.scrollHeight,
    clientHeight: element.clientHeight,
  }))
  expect(outputMetrics.scrollHeight).toBeGreaterThan(outputMetrics.clientHeight + 40)

  await page.getByRole('button', { name: /Show .* more lines/ }).click()
  await expect(terminal).toContainText('terminal line 180')
  await expect(page.getByPlaceholder('Ask Aether')).toBeVisible()
  await expect(page.getByLabel('Console sections')).toBeVisible()

  const layout = await page.evaluate(() => ({
    docWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
    composer: document.querySelector('.composer')?.getBoundingClientRect().toJSON(),
    terminal: document.querySelector('.terminal-chrome')?.getBoundingClientRect().toJSON(),
  }))
  expect(layout.docWidth).toBeLessThanOrEqual(layout.viewportWidth + 1)
  const composerBox = layout.composer as DOMRect | null
  const terminalBox = layout.terminal as DOMRect | null
  expect(composerBox?.left ?? -1).toBeGreaterThanOrEqual(0)
  expect((composerBox?.right ?? 9999)).toBeLessThanOrEqual(layout.viewportWidth + 1)
  expect(terminalBox?.left ?? -1).toBeGreaterThanOrEqual(0)
  expect((terminalBox?.right ?? 9999)).toBeLessThanOrEqual(layout.viewportWidth + 1)
})

test('keeps composer workspace references usable at narrow widths', async ({ page }) => {
  await page.setViewportSize({ width: 680, height: 720 })
  await page.goto('/')

  const composer = page.locator('.composer')
  await expect(composer).toBeVisible()
  const textarea = page.getByPlaceholder('Ask Aether')
  await textarea.fill('Inspect @auth')

  const popover = page.getByRole('listbox', { name: 'Workspace references' })
  await expect(popover).toContainText('auth.ts')
  await popover.getByRole('option', { name: /auth\.ts/ }).click()

  await expect(page.getByLabel('Workspace context')).toContainText('src/auth.ts')
  await expect(textarea).toHaveValue(/@src\/auth\.ts\s*$/)

  await page.getByRole('button', { name: 'Preview workspace reference auth.ts' }).click()
  await expect(page.getByLabel('Workspace reference preview')).toContainText('src/auth.ts')
  await expect(page.getByLabel('Workspace reference preview')).toContainText('Browser acceptance fixture')
  await expectWithinViewport(page, ['.composer', '.composer-workspace-context', '.composer-workspace-preview'])
  await expectNoDocumentHorizontalOverflow(page)

  const withSecondReference = (await textarea.inputValue()) + '@auth.test'
  await textarea.fill(withSecondReference)
  await page.getByRole('listbox', { name: 'Workspace references' }).getByRole('option', { name: /auth[.]test[.]ts/ }).click()
  const context = page.getByLabel('Workspace context')
  await expect(context).toContainText('2 refs')
  await context.getByRole('button', { name: 'Move workspace reference auth.test.ts earlier' }).click()
  await expect.poll(async () => context.getByRole('group').evaluateAll((items) => items.map((item) => item.getAttribute('aria-label')))).toEqual([
    'Workspace reference auth.test.ts',
    'Workspace reference auth.ts',
  ])
  await expectWithinViewport(page, ['.composer', '.composer-workspace-context', '.composer-workspace-preview'])
  await expectNoDocumentHorizontalOverflow(page)

  await page.getByRole('button', { name: 'Remove workspace reference auth.ts' }).click()
  await page.getByRole('button', { name: 'Remove workspace reference auth.test.ts' }).click()
  await expect(page.getByLabel('Workspace context')).toBeHidden()
  const finalComposerValue = await textarea.inputValue()
  expect(finalComposerValue).not.toContain('@src/auth.ts')
  expect(finalComposerValue).not.toContain('@src/auth.test.ts')

})

test("syncs edited workspace reference tokens before sending a run", async ({ page }) => {
  await page.goto("/")

  const textarea = page.getByPlaceholder("Ask Aether")
  await textarea.fill("Inspect @auth")
  const popover = page.getByRole("listbox", { name: "Workspace references" })
  await expect(popover).toContainText("auth.ts")
  await popover.getByRole("option", { name: /auth[.]ts/ }).click()
  await expect(page.getByLabel("Workspace context")).toContainText("src/auth.ts")

  await textarea.fill("Inspect auth without context")
  await expect(page.getByLabel("Workspace context")).toBeHidden()
  await page.getByRole("button", { name: "Send message" }).click()

  await expect.poll(async () => page.evaluate(() => {
    const frames = (window as unknown as { __aetherRunFrames?: Array<{ type: string; payload?: Record<string, unknown> }> }).__aetherRunFrames ?? []
    return frames.find((frame) => frame.type === "run.start")?.payload ?? null
  })).toEqual(expect.objectContaining({ user_message: "Inspect auth without context" }))

  const runStartPayload = await page.evaluate(() => {
    const frames = (window as unknown as { __aetherRunFrames?: Array<{ type: string; payload?: Record<string, unknown> }> }).__aetherRunFrames ?? []
    return frames.find((frame) => frame.type === "run.start")?.payload ?? null
  })
  expect(runStartPayload).not.toHaveProperty("attachments")
})

test('keeps mobile-width chat, composer, task bar, and prompts within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 720 })
  await page.goto('/?scenario=responsive')

  await expect(page.getByText('Responsive stress response')).toBeVisible()
  await expect(page.getByLabel('Changed files')).toContainText('src/components/really-long-responsive-file-name-that-should-clip.ts')
  await expect(page.getByPlaceholder('Ask Aether')).toBeVisible()
  await expect(page.getByRole('button', { name: /Subagent tasks/ })).toBeVisible()

  await expectWithinViewport(page, ['.app-shell', '.chat-surface', '.composer', '.session-task-bar', '.terminal-chrome'])
  await expectNoDocumentHorizontalOverflow(page)

  await page.getByPlaceholder('Ask Aether').fill('Run live smoke')
  await page.getByRole('button', { name: 'Send message' }).click()
  const permissionDialog = page.getByRole('dialog', { name: 'Tool permission request' })
  await expect(permissionDialog).toContainText('Streaming smoke permission')
  await expectWithinViewport(page, ['.prompt-modal'])
  await expectNoDocumentHorizontalOverflow(page)
})

test('renders non-streamed run.result final text as an assistant reply', async ({ page }) => {
  await page.goto('/')

  await page.getByPlaceholder('Ask Aether').fill('Run non streaming smoke')
  await page.getByRole('button', { name: 'Send message' }).click()

  await expect(page.getByText('Final non-streamed answer')).toBeVisible()
})

test('queues concurrent permission and approval prompts without overwriting the active modal', async ({ page }) => {
  await page.goto('/')

  await page.getByPlaceholder('Ask Aether').fill('Run prompt queue smoke')
  await page.getByRole('button', { name: 'Send message' }).click()

  const permissionDialog = page.getByRole('dialog', { name: 'Tool permission request' })
  await expect(permissionDialog).toContainText('Queue permission one')
  await expect(permissionDialog).not.toContainText('Queue permission two')

  await permissionDialog.getByRole('button', { name: 'Deny' }).click()
  await expect(permissionDialog).toContainText('Queue permission two')
  await expect(permissionDialog).not.toContainText('Queue permission one')

  await permissionDialog.getByRole('button', { name: 'Allow once' }).click()
  const approvalDialog = page.getByRole('dialog', { name: 'Approval request' })
  await expect(approvalDialog).toContainText('Queued live plan')
  await approvalDialog.getByRole('button', { name: 'Approve' }).click()
  await expect(approvalDialog).toBeHidden()
  await expect(page.getByText('Queued prompt answer')).toBeVisible()
})

test('renders live streaming, permission, approval, and activity states from websocket frames', async ({ page }) => {
  await page.goto('/')

  await page.getByPlaceholder('Ask Aether').fill('Run live smoke')
  await page.getByRole('button', { name: 'Send message' }).click()

  await expect(page.getByText('Live answer from websocket')).toBeVisible()
  await expect(page.getByText('Thinking through the request')).toBeVisible()
  const permissionDialog = page.getByRole('dialog', { name: 'Tool permission request' })
  await expect(permissionDialog).toContainText('Streaming smoke permission')
  await expect(permissionDialog).toContainText('Permission diff preview')

  await permissionDialog.getByRole('button', { name: 'Deny' }).click()
  await expect(permissionDialog).toBeHidden()
  const approvalDialog = page.getByRole('dialog', { name: 'Approval request' })
  await expect(approvalDialog).toContainText('Live plan')

  await approvalDialog.getByRole('button', { name: 'Approve' }).click()
  await expect(approvalDialog).toBeHidden()
  await expect(page.getByText('Live answer from websocket')).toBeVisible()
})

test('matches desktop chat visual baseline', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 820 })
  await page.goto('/')
  await expect(page.getByText('Auth flow summary')).toBeVisible()
  await stabilizeVisuals(page)

  await expect(page.locator('.app-shell')).toHaveScreenshot('chat-shell-desktop.png', {
    animations: 'disabled',
    caret: 'hide',
    maxDiffPixelRatio: 0.015,
  })
})

test('matches narrow composer workspace-reference visual baseline', async ({ page }) => {
  await page.setViewportSize({ width: 680, height: 720 })
  await page.goto('/')
  const textarea = page.getByPlaceholder('Ask Aether')
  await textarea.fill('Inspect @auth')
  await page.getByRole('listbox', { name: 'Workspace references' }).getByRole('option', { name: /auth\.ts/ }).click()
  await expect(page.getByLabel('Workspace context')).toContainText('src/auth.ts')
  await stabilizeVisuals(page)

  await expect(page.locator('.chat-surface')).toHaveScreenshot('chat-surface-narrow-workspace.png', {
    animations: 'disabled',
    caret: 'hide',
    maxDiffPixelRatio: 0.025,
  })
})

test('matches long terminal visual baseline at narrow width', async ({ page }) => {
  await page.setViewportSize({ width: 760, height: 720 })
  await page.goto('/?scenario=long-tool')
  await expect(page.getByLabel('Terminal output').first()).toContainText('terminal line 080')
  await page.locator('.chat-scroll').evaluate((element) => {
    element.scrollTop = 0
    element.dispatchEvent(new Event('scroll', { bubbles: true }))
  })
  await stabilizeVisuals(page)

  await expect(page.locator('.chat-surface')).toHaveScreenshot('chat-surface-long-terminal.png', {
    animations: 'disabled',
    caret: 'hide',
  })
})

test('matches permission modal visual baseline with diff preview', async ({ page }) => {
  await page.setViewportSize({ width: 980, height: 720 })
  await page.goto('/')
  await page.getByPlaceholder('Ask Aether').fill('Run live smoke')
  await page.getByRole('button', { name: 'Send message' }).click()
  await expect(page.getByRole('dialog', { name: 'Tool permission request' })).toContainText('Streaming smoke permission')
  await stabilizeVisuals(page)

  await expect(page.locator('.modal-backdrop')).toHaveScreenshot('permission-modal-diff.png', {
    animations: 'disabled',
    caret: 'hide',
  })
})

async function boxWidth(locator: Locator): Promise<number> {
  const box = await locator.boundingBox()
  expect(box).not.toBeNull()
  return box?.width ?? 0
}

async function boxHeight(locator: Locator): Promise<number> {
  const box = await locator.boundingBox()
  expect(box).not.toBeNull()
  return box?.height ?? 0
}

async function dragHorizontalSeparator(page: Page, locator: Locator, deltaX: number): Promise<void> {
  const box = await locator.boundingBox()
  expect(box).not.toBeNull()
  if (!box) return
  const x = box.x + box.width / 2
  const y = box.y + box.height / 2
  await page.mouse.move(x, y)
  await page.mouse.down()
  await page.mouse.move(x + deltaX, y, { steps: 4 })
  await page.mouse.up()
}

async function expectNoDocumentHorizontalOverflow(page: Page) {
  const metrics = await page.evaluate(() => ({
    docWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
    bodyWidth: document.body.scrollWidth,
  }))
  expect(metrics.docWidth).toBeLessThanOrEqual(metrics.viewportWidth + 1)
  expect(metrics.bodyWidth).toBeLessThanOrEqual(metrics.viewportWidth + 1)
}

async function expectWithinViewport(page: Page, selectors: string[]) {
  const boxes = await page.evaluate((items) => {
    const viewportWidth = document.documentElement.clientWidth
    return items.map((selector) => {
      const element = document.querySelector(selector)
      if (!element) return { selector, missing: true, left: 0, right: 0, viewportWidth }
      const rect = element.getBoundingClientRect()
      return { selector, missing: false, left: rect.left, right: rect.right, viewportWidth }
    })
  }, selectors)
  for (const box of boxes) {
    expect(box.missing, box.selector).toBe(false)
    expect(box.left, box.selector + ' left').toBeGreaterThanOrEqual(-1)
    expect(box.right, box.selector + ' right').toBeLessThanOrEqual(box.viewportWidth + 1)
  }
}
async function stabilizeVisuals(page: Page) {
  await page.addStyleTag({
    content: `
      *, *::before, *::after {
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition-duration: 0s !important;
        transition-delay: 0s !important;
        caret-color: transparent !important;
      }
      .aether-shimmer-text, .spinner-shimmer, .web-activity-bar {
        animation: none !important;
      }
    `,
  })
  await page.evaluate(async () => {
    const active = document.activeElement
    if (active instanceof HTMLElement) active.blur()
    const fonts = 'fonts' in document ? document.fonts : null
    if (fonts) await fonts.ready
  })
  await page.waitForTimeout(50)
}

async function installMockRunSocket(page: Page) {
  await page.addInitScript(() => {
    type Listener = ((event: Event) => void) | null
    type MessageListener = ((event: MessageEvent) => void) | null
    type SocketFrame = { type: string; id?: string | number | null; payload?: Record<string, unknown> }
    const capturedWindow = window as unknown as { __aetherRunFrames?: SocketFrame[] }
    capturedWindow.__aetherRunFrames = []

    class MockRunWebSocket {
      static CONNECTING = 0
      static OPEN = 1
      static CLOSING = 2
      static CLOSED = 3

      readonly url: string
      readyState = MockRunWebSocket.CONNECTING
      onopen: Listener = null
      onclose: Listener = null
      onerror: Listener = null
      onmessage: MessageListener = null

      constructor(url: string) {
        this.url = url
        window.setTimeout(() => {
          this.readyState = MockRunWebSocket.OPEN
          this.onopen?.(new Event('open'))
          this.emit({ type: 'ready', payload: { ok: true } })
        }, 0)
      }

      send(raw: string) {
        const frame = JSON.parse(raw) as SocketFrame
        capturedWindow.__aetherRunFrames = [...(capturedWindow.__aetherRunFrames ?? []), frame]
        if (frame.type === 'run.start') this.emitRunFrames(frame)
        if (frame.type === 'permission.respond') {
          const promptId = frame.payload?.prompt_id
          this.emit({ type: 'prompt.resolved', payload: { prompt_id: promptId, result: frame.payload } })
          if (promptId === 'perm-queue-1') return
          if (promptId === 'perm-queue-2') {
            this.emitApprovalRequest('Queued live plan')
            return
          }
          this.emitApprovalRequest()
        }
        if (frame.type === 'approval.respond') {
          this.emit({ type: 'prompt.resolved', payload: { prompt_id: frame.payload?.prompt_id } })
          this.emit({
            type: 'run.finished',
            payload: { session_id: 'session-1', run_id: 'browser-run' },
          })
        }
      }

      close() {
        this.readyState = MockRunWebSocket.CLOSED
        this.onclose?.(new Event('close'))
      }

      addEventListener(type: string, listener: EventListener) {
        if (type === 'message') this.onmessage = listener as MessageListener
        if (type === 'open') this.onopen = listener as Listener
        if (type === 'close') this.onclose = listener as Listener
        if (type === 'error') this.onerror = listener as Listener
      }

      removeEventListener() {
        return undefined
      }

      private emit(frame: SocketFrame) {
        this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(frame) }))
      }

      private emitApprovalRequest(title = 'Live plan') {
        const base = { session_id: 'session-1', run_id: 'browser-run' }
        this.emit({
          type: 'approval.requested',
          payload: {
            ...base,
            prompt_id: title === 'Queued live plan' ? 'approval-queue-1' : 'approval-1',
            kind: 'plan',
            plan_text: '# ' + title + '\n\n- Verify prompt rendering\n- Continue implementation',
            plan_path: '/tmp/aether/plans/session-1.md',
            questions: [],
          },
        })
      }

      private emitPermissionRequest(base: Record<string, unknown>, promptId: string, title: string, path: string) {
        this.emit({
          type: 'permission.requested',
          payload: {
            ...base,
            prompt_id: promptId,
            request: {
              tool_name: 'file_edit',
              tool_call_id: promptId + '-tool',
              arguments: { path },
              category: 'write',
              risk: 'medium',
              reason: title + ' requires review',
              allow_session: false,
              preview: {
                title,
                subtitle: path,
                diff: '--- a/' + path + '\n+++ b/' + path + '\n@@ -1 +1 @@\n-old\n+new\n',
              },
            },
          },
        })
      }

      private emitRunFrames(frame: SocketFrame) {
        const runId = typeof frame.id === 'string' ? frame.id : 'browser-run'
        const sessionId = typeof frame.payload?.session_id === 'string' ? frame.payload.session_id : 'session-1'
        const base = { session_id: sessionId, run_id: runId }
        const message = typeof frame.payload?.user_message === 'string' ? frame.payload.user_message : ''
        window.setTimeout(() => {
          this.emit({ type: 'run.accepted', payload: { ...base } })
          if (message.includes('prompt queue')) {
            this.emit({ type: 'assistant.delta', payload: { ...base, text: 'Queued prompt answer', sequence: 1 } })
            this.emitPermissionRequest(base, 'perm-queue-1', 'Queue permission one', 'src/one.ts')
            this.emitPermissionRequest(base, 'perm-queue-2', 'Queue permission two', 'src/two.ts')
            return
          }
          if (message.includes('non streaming')) {
            this.emit({ type: 'run.status', payload: { ...base, kind: 'responding', detail: 'finalizing' } })
            this.emit({
              type: 'run.result',
              payload: {
                ...base,
                final_text: 'Final non-streamed answer',
                usage: { input_tokens: 12, output_tokens: 9, total_tokens: 21 },
                metadata: { hosted_web_search: { provider: 'codex', source_count: 1 } },
              },
            })
            return
          }
          this.emit({ type: 'reasoning.delta', payload: { ...base, text: 'Thinking through the request', sequence: 1 } })
          this.emit({ type: 'assistant.delta', payload: { ...base, text: 'Live answer from websocket', sequence: 2 } })
          this.emit({ type: 'run.status', payload: { ...base, kind: 'tool_use', detail: 'checking permission' } })
          this.emit({ type: 'token.usage', payload: { ...base, input_tokens: 10, output_tokens: 4, total_tokens: 14 } })
          this.emit({
            type: 'permission.requested',
            payload: {
              ...base,
              prompt_id: 'perm-1',
              request: {
                tool_name: 'file_edit',
                tool_call_id: 'tool-live',
                arguments: { path: 'src/live.ts' },
                category: 'write',
                risk: 'medium',
                reason: 'Permission diff preview',
                allow_session: true,
                preview: {
                  title: 'Streaming smoke permission',
                  subtitle: 'src/live.ts',
                  diff: '--- a/src/live.ts\n+++ b/src/live.ts\n@@ -1 +1 @@\n-old\n+new\n',
                },
              },
            },
          })
        }, 10)
      }
    }

    window.WebSocket = MockRunWebSocket as unknown as typeof WebSocket
  })
}

async function mockAetherApi(page: Page) {
  let sessionDeleted = false
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    const method = route.request().method()
    const path = url.pathname
    if (!path.startsWith('/api/')) return route.continue()

    if (method === 'GET' && path === '/api/status') return json(route, { ok: true, name: 'Aether', version: 'e2e', web: { enabled: true } })
    if (method === 'GET' && path === '/api/health') {
      return json(route, {
        status: 'ok',
        runtime: { python_version: '3.12', platform: 'linux', implementation: 'cpython' },
        services: [{ name: 'agent', available: true, status: 'ok' }],
        diagnostics: { enabled: true, pending_count: 0 },
      })
    }
    if (method === 'GET' && path === '/api/prefs') return json(route, {})
    if (method === 'GET' && path === '/api/commands') {
      return json(route, {
        commands: [
          { name: '/help', description: 'Show help', category: 'local' },
          { name: '/plan', description: 'Enable plan mode or view the current session plan', category: 'session' },
          { name: '/clear', description: 'Clear the current session view', category: 'session' },
        ],
      })
    }
    if (method === 'GET' && path === '/api/providers') {
      return json(route, {
        providers: [{ name: 'openai', display_name: 'OpenAI Compatible', requires_api_key: false, default_base_url: null }],
      })
    }
    if (method === 'GET' && path === '/api/providers/current') {
      return json(route, {
        family: 'openai',
        provider_name: 'openai',
        model: 'gpt-5.4',
        base_url: null,
        api_key_env_names: [],
        model_env_names: [],
        base_url_env_names: [],
        source: 'e2e',
        credential: null,
      })
    }
    if (method === 'GET' && path === '/api/providers/openai/models') {
      return json(route, { models: [{ id: 'gpt-5.4', display_name: 'gpt-5.4', context_window: 1_000_000 }], discovery: { kind: 'static', count: 1 } })
    }
    if (method === 'GET' && path === '/api/tools/groups') {
      return json(route, {
        groups: [{
          name: 'files',
          tools: [{
            name: 'read_file',
            description: 'Read a file from the workspace',
            parameters: { type: 'object', properties: { path: { type: 'string', description: 'Workspace path' } } },
            required: ['path'],
            enabled: true,
          }],
        }],
      })
    }
    if (method === 'GET' && path === '/api/config') return json(route, { values: { provider: 'openai', model: 'gpt-5.4' } })
    if (method === 'GET' && path === '/api/config/paths') return json(route, { aether_home: '/tmp/aether', sessions_dir: '/tmp/aether/sessions', prefs_file: '/tmp/aether/prefs.json' })
    if (method === 'GET' && path === '/api/sessions') return json(route, { sessions: sessionDeleted ? [] : [session] })
    if (method === 'DELETE' && path === '/api/sessions/session-1') {
      sessionDeleted = true
      return route.fulfill({ status: 204 })
    }
    if (method === 'GET' && path === '/api/sessions/session-1/messages') return json(route, { session_id: 'session-1', messages: currentScenarioMessages(page) })
    if (method === 'GET' && path === '/api/sessions/session-1/tasks') return json(route, { tasks: [rootTask, childTask], active_count: 0, total_count: 2 })
    if (method === 'GET' && path === '/api/tasks/task-1') return json(route, rootTask)
    if (method === 'GET' && path === '/api/tasks/task-2') return json(route, childTask)
    if (method === 'GET' && path === '/api/tasks/task-1/messages') return json(route, taskMessages('task-1'))
    if (method === 'GET' && path === '/api/tasks/task-2/messages') return json(route, taskMessages('task-2'))
    if (method === 'GET' && path === '/api/tasks/task-1/children/messages') return json(route, childMessages())
    if (method === 'GET' && path === '/api/tasks/task-2/children/messages') return json(route, { task_id: 'task-2', streams: [], total_count: 0, truncated: false })
    if (method === 'GET' && path === '/api/tasks/task-1/result') return json(route, { task_id: 'task-1', result_path: rootTask.result_path, result: { summary: 'Task completed from artifact', files: ['src/auth.ts'] } })
    if (method === 'GET' && path === '/api/workspace/file') {
      const requestedPath = url.searchParams.get('path') || 'README.md'
      return json(route, workspaceFilePayload(requestedPath, '# Aether workspace readme\n\nBrowser acceptance fixture.'))
    }
    if (method === 'PUT' && path === '/api/workspace/file') {
      const body = route.request().postDataJSON() as { path?: string; content?: string }
      const requestedPath = body.path || 'README.md'
      return json(route, workspaceFilePayload(requestedPath, body.content || ''))
    }
    if (method === 'GET' && path === '/api/workspace/raw') {
      return route.fulfill({ status: 200, contentType: 'image/png', body: Buffer.from('iVBORw0KGgo=', 'base64') })
    }
    if (method === 'GET' && path === '/api/workspace/search') {
      return json(route, {
        root: '/workspace/Aether',
        query: url.searchParams.get('q') || '',
        entries: [
          { path: 'src/auth.ts', name: 'auth.ts', kind: 'file', size_bytes: 2400, updated_at: 1_800_000_000 },
          { path: 'src/auth.test.ts', name: 'auth.test.ts', kind: 'file', size_bytes: 1800, updated_at: 1_800_000_000 },
        ],
      })
    }
    if (method === 'GET' && path === '/api/workspace/tree') {
      return json(route, {
        root: '/workspace/Aether',
        path: '',
        parent_path: null,
        entries: [
          { path: 'README.md', name: 'README.md', kind: 'file', size_bytes: 100, updated_at: 1_800_000_000 },
          { path: 'src', name: 'src', kind: 'directory', size_bytes: null, updated_at: 1_800_000_000 },
        ],
      })
    }
    return json(route, {})
  })
}

function workspaceFilePayload(requestedPath: string, content: string) {
  return {
    root: '/workspace/Aether',
    path: requestedPath,
    name: requestedPath.split('/').pop() || requestedPath,
    content,
    size_bytes: content.length,
    updated_at: 1_800_000_000,
    language: requestedPath.endsWith('.md') ? 'markdown' : 'text',
    mime_type: requestedPath.endsWith('.md') ? 'text/markdown' : 'text/plain',
    truncated: false,
    binary: false,
  }
}

function currentScenarioMessages(page: Page) {
  const scenario = new URL(page.url()).searchParams.get('scenario')
  if (scenario === 'long') return longTranscriptMessages()
  if (scenario === 'long-tool') return longToolTranscriptMessages()
  if (scenario === 'responsive') return responsiveStressMessages()
  if (scenario === 'notebook') return notebookTranscriptMessages()
  if (scenario === 'web-search') return webSearchTranscriptMessages()
  if (scenario === 'artifacts') return artifactTranscriptMessages()
  if (scenario === 'rich-markdown') return richMarkdownTranscriptMessages()
  return transcriptMessages()
}

function responsiveStressMessages() {
  const longPath = 'src/components/really-long-responsive-file-name-that-should-clip.ts'
  const longToken = 'supercalifragilisticexpialidocious'.repeat(4)
  return [
    {
      role: 'user',
      text: 'Please inspect ' + longToken,
      attachments: [{ type: 'file', name: 'really-long-responsive-file-name-that-should-clip.ts', path: longPath }],
      metadata: null,
    },
    {
      role: 'assistant',
      text: '## Responsive stress response\n\n| Surface | Expectation |\n| --- | --- |\n| composer | remains visible |\n| diff | clips without page overflow |\n\n' +
        '```ts\nexport const extremelyLongIdentifier = "' + longToken + '"\n```',
      tool_calls: [{ id: 'call-responsive-bash', name: 'bash', arguments: { command: 'node scripts/' + longToken + '.js --flag=' + longToken } }],
      metadata: null,
    },
    {
      role: 'tool',
      name: 'bash',
      tool_call_id: 'call-responsive-bash',
      text: 'responsive output ' + longToken,
      is_error: false,
      metadata: { kind: 'terminal', command: 'node scripts/' + longToken + '.js --flag=' + longToken, exit_code: 0, duration_ms: 1200 },
    },
    {
      role: 'tool',
      name: 'file_edit',
      tool_call_id: 'call-responsive-edit',
      text: 'Updated ' + longPath,
      is_error: false,
      metadata: {
        path: longPath,
        diff: '--- a/' + longPath + '\n+++ b/' + longPath + '\n@@ -1 +1 @@\n-const oldValue = "' + longToken + '"\n+const newValue = "' + longToken + '"\n',
      },
    },
  ]
}
function richMarkdownTranscriptMessages() {
  const fence = String.fromCharCode(96, 96, 96)
  return [
    {
      role: 'user',
      text: 'Render a markdown-heavy answer for browser acceptance',
      metadata: null,
    },
    {
      role: 'assistant',
      text: [
        '## Render acceptance',
        '',
        '| Surface | Status |',
        '| --- | --- |',
        '| table | rendered |',
        '| math | rendered |',
        '',
        String.raw`Inline math \(x^2 + y^2\) stays readable and a safe [docs](https://example.com/docs) link remains clickable.`,
        '',
        String.raw`\[`,
        'E = mc^2',
        String.raw`\]`,
        '',
        fence + 'typescript',
        'const value = {"ok": true, "count": 2}',
        '// browser acceptance',
        fence,
        '',
        fence + 'mermaid',
        'flowchart LR',
        '  A[Plan] --> B[Implement]',
        '  B --> C[Verify]',
        fence,
        '',
        '> Keep the markdown renderer inside the console shell.',
      ].join('\n'),
      metadata: null,
    },
  ]
}

function webSearchTranscriptMessages() {
  return [
    {
      role: 'user',
      text: 'Search the web for Aether provider docs',
      metadata: null,
    },
    {
      role: 'assistant',
      text: 'I searched the configured web provider and found documentation references.',
      tool_calls: [{ id: 'call-web-search', name: 'web_search', arguments: { query: 'Aether provider' } }],
      metadata: null,
    },
    {
      role: 'tool',
      name: 'web_search',
      tool_call_id: 'call-web-search',
      text: JSON.stringify({
        web: {
          results: [
            { title: 'Aether docs', url: 'https://example.com/aether', description: 'Documentation snippet' },
            { title: 'Aether provider guide', url: 'https://example.com/aether/provider', extra_snippets: ['Provider guide snippet', 'Extra provider context'] },
          ],
        },
      }),
      is_error: false,
      metadata: { provider: 'brave', source_count: 2 },
    },
  ]
}

function notebookTranscriptMessages() {
  return [
    {
      role: 'user',
      text: 'Execute the analysis notebook cell and summarize the result',
      attachments: [{ type: 'file', name: 'analysis.ipynb', path: 'notebooks/analysis.ipynb' }],
      metadata: null,
    },
    {
      role: 'assistant',
      text: 'I executed the selected notebook cell and captured the outputs.',
      tool_calls: [{
        id: 'call-notebook',
        name: 'notebook_edit',
        arguments: {
          notebook_path: 'notebooks/analysis.ipynb',
          cell_idx: 2,
          cell_type: 'code',
          edit_mode: 'execute',
        },
      }],
      metadata: null,
    },
    {
      role: 'tool',
      name: 'notebook_edit',
      tool_call_id: 'call-notebook',
      text: JSON.stringify({
        summary: 'executed cell',
        status: 'ok',
        outputs: [
          { output_type: 'stream', name: 'stdout', text: ['hello world'] },
          { output_type: 'error', ename: 'ValueError', evalue: 'bad value', traceback: ['Traceback line', 'ValueError: bad value'] },
          { output_type: 'display_data', data: { 'image/png': 'iVBORw0KGgo=' } },
        ],
      }),
      is_error: false,
      metadata: {
        path: 'analysis.ipynb',
        edit_mode: 'execute',
        execution_status: 'ok',
        execution_count: 4,
        queued_at: '10:00:00',
        started_at: '10:00:01',
        finished_at: '10:00:02',
        duration_ms: 1250,
        kernel_name: 'python3',
        outputs_truncated: true,
      },
    },
  ]
}

function longTranscriptMessages() {
  const messages: Array<Record<string, unknown>> = []
  for (let index = 1; index <= 36; index += 1) {
    messages.push({
      role: 'user',
      text: 'Long transcript message ' + index + ': inspect module boundary and keep this history readable.',
      metadata: null,
    })
    messages.push({
      role: 'assistant',
      text: '### Long transcript response ' + index + '\n\n' +
        'This response intentionally contains enough text to force the chat surface to scroll. ' +
        'The composer, navigation rail, and current scroll position should remain stable while new frames arrive.\n\n' +
        '    export const sample' + index + ' = \"stable-scroll\"',
      metadata: null,
    })
  }
  return messages
}

function artifactTranscriptMessages() {
  return [
    {
      role: 'user',
      text: 'Inspect generated artifacts',
      metadata: null,
    },
    {
      role: 'assistant',
      text: 'I collected the generated artifacts.',
      tool_calls: [{ id: 'call-artifacts', name: 'artifact_tool', arguments: {} }],
      metadata: null,
    },
    {
      role: 'tool',
      name: 'artifact_tool',
      tool_call_id: 'call-artifacts',
      text: JSON.stringify({
        artifacts: [
          { name: 'weights.bin', path: '/tmp/aether/weights.bin', mime_type: 'application/octet-stream', size_bytes: 4096, binary: true },
          { name: 'report.json', path: '/tmp/aether/report.json', mime_type: 'application/json', content: { ok: true, count: 2 } },
          { name: 'bundle.zip', path: '/tmp/aether/bundle.zip', download_url: 'https://example.com/bundle.zip', mime_type: 'application/zip', size_bytes: 1000 },
        ],
      }),
      is_error: false,
      metadata: null,
    },
  ]
}

function longToolTranscriptMessages() {
  return [
    {
      role: 'user',
      text: 'Run a command with a long terminal output',
      metadata: null,
    },
    {
      role: 'assistant',
      text: 'I will run the smoke command and keep the output contained.',
      tool_calls: [{ id: 'call-long-bash', name: 'bash', arguments: { command: 'npm run long-smoke -- --verbose' } }],
      metadata: null,
    },
    {
      role: 'tool',
      name: 'bash',
      tool_call_id: 'call-long-bash',
      text: longTerminalOutput(),
      is_error: false,
      metadata: { kind: 'terminal', command: 'npm run long-smoke -- --verbose', exit_code: 0, duration_ms: 9400 },
    },
  ]
}

function longTerminalOutput() {
  const lines: string[] = []
  for (let index = 1; index <= 180; index += 1) {
    lines.push('terminal line ' + String(index).padStart(3, '0') + '  ' + 'x'.repeat(24))
  }
  return lines.join('\n')
}

function transcriptMessages() {
  return [
    {
      role: 'user',
      text: 'Inspect authentication flow',
      attachments: [{ type: 'file', name: 'auth.ts', path: 'src/auth.ts' }],
      metadata: null,
    },
    {
      role: 'assistant',
      text: '## Auth flow summary\n\n| Step | Status |\n| --- | --- |\n| Login | checked |\n\nThe main path is covered.',
      tool_calls: [{ id: 'call-bash', name: 'bash', arguments: { command: 'npm test -- auth' } }],
      metadata: null,
    },
    {
      role: 'tool',
      name: 'bash',
      tool_call_id: 'call-bash',
      text: '2 passed',
      is_error: false,
      metadata: { kind: 'terminal', command: 'npm test -- auth', exit_code: 0, duration_ms: 1400 },
    },
    {
      role: 'assistant',
      text: '',
      tool_calls: [{ id: 'call-question', name: 'ask_user_question', arguments: { questions: [{ id: 'storage', question: 'Use token storage?', options: [{ label: 'Encrypted session storage' }, { label: 'Memory only' }] }] } }],
      metadata: null,
    },
    {
      role: 'tool',
      name: 'ask_user_question',
      tool_call_id: 'call-question',
      text: 'Encrypted session storage',
      is_error: false,
      metadata: { answers: { storage: 'Encrypted session storage' } },
    },
    {
      role: 'tool',
      name: 'file_edit',
      tool_call_id: 'call-edit',
      text: 'Updated src/auth.ts',
      is_error: false,
      metadata: {
        path: 'src/auth.ts',
        diff: '--- a/src/auth.ts\n+++ b/src/auth.ts\n@@ -1 +1 @@\n-export const mode = "old"\n+export const mode = "new"\n',
      },
    },
  ]
}

function taskMessages(taskId: string) {
  if (taskId === 'task-2') {
    return {
      task_id: 'task-2',
      messages: [{ index: 0, role: 'assistant', content: 'child assistant response', iteration: 1, elapsed_ms: 900, raw: {} }],
      pending_messages: [],
      delivered_messages: [],
      total_count: 1,
      truncated: false,
    }
  }
  return {
    task_id: 'task-1',
    messages: [
      { index: 0, role: 'assistant', content: 'assistant observed root task', iteration: 1, elapsed_ms: 1200, raw: {} },
      { index: 1, role: 'tool', name: 'bash', tool_call_id: 'call-task-shell', content: 'shell output from task', iteration: 1, elapsed_ms: 1600, raw: {} },
    ],
    pending_messages: [{ index: 0, message: 'Please inspect auth edge cases', ts: 1_800_000_011, raw: {} }],
    delivered_messages: [{ index: 0, message: 'Use stricter validation', ts: 1_800_000_012, delivered_at: 1_800_000_013, raw: {} }],
    total_count: 4,
    truncated: false,
  }
}

function childMessages() {
  return {
    task_id: 'task-1',
    streams: [{
      task: childTask,
      messages: [{ index: 0, role: 'assistant', content: 'child assistant response', iteration: 1, elapsed_ms: 900, raw: {} }],
      pending_messages: [],
      delivered_messages: [],
      total_count: 1,
      truncated: false,
    }],
    total_count: 1,
    truncated: false,
  }
}

function json(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}
