// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $toolDisclosureStates } from '@/store/tool-view'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({
      message: { id: 'msg-1', parts: [], status: { type: 'complete' } },
      thread: { isRunning: false, messages: [{ id: 'msg-1' }] }
    })
}))

const { ToolFallback } = await import('./fallback')

const originalDesktop = window.hermesDesktop
const fetchLinkTitle = vi.fn(async () => 'AI Tools Directory - dealsbe.com')

function renderFailedWrite(message: string) {
  const props = {
    args: { path: 'C:\\Users\\me\\projects\\myrepo\\README.md' },
    completedAt: 2,
    isError: true,
    result: message,
    timestamp: 1,
    toolCallId: 'call-write-file',
    toolName: 'write_file'
  } as unknown as ComponentProps<typeof ToolFallback>

  render(<ToolFallback {...props} />)
  fireEvent.click(screen.getByRole('button', { expanded: false }))
}

beforeEach(() => {
  fetchLinkTitle.mockClear()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      fetchLinkTitle,
      openExternal: vi.fn(async () => undefined)
    }
  })
})

afterEach(() => {
  cleanup()
  $toolDisclosureStates.set({})
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: originalDesktop
  })
})

describe('ToolFallback error summary links', () => {
  it('keeps filename-shaped Windows paths as authored text and never fetches their titles', async () => {
    renderFailedWrite('Could not write C:\\Users\\me\\projects\\myrepo\\README.md because the file is unreadable.')

    expect(screen.queryByRole('link', { name: /README\.md/i })).toBeNull()
    expect(screen.getByText(content => content.includes('C:\\Users\\me\\projects\\myrepo\\README.md'))).toBeTruthy()

    await waitFor(() => expect(fetchLinkTitle).not.toHaveBeenCalled())
  })

  it('still links explicit URLs in tool error summaries without prettifying them', () => {
    renderFailedWrite('Could not write README.md. See https://example.com/docs for details.')

    const links = screen.getAllByRole('link')
    expect(links).toHaveLength(1)
    expect(links[0]?.textContent).toBe('https://example.com/docs')
    expect(links[0]?.getAttribute('href')).toBe('https://example.com/docs')
    expect(screen.queryByRole('link', { name: /README\.md/i })).toBeNull()
    expect(fetchLinkTitle).not.toHaveBeenCalled()
  })
})
