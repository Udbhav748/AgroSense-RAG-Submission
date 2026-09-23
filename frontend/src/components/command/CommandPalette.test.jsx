import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import CommandPalette from './CommandPalette'
import * as chatService from '../../services/chatService'

const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
}))

const mockSetTheme = vi.fn()
let currentTheme = 'light'
vi.mock('../../hooks/useTheme', () => ({
  default: () => ({
    theme: currentTheme,
    setTheme: mockSetTheme,
  }),
}))

vi.mock('../../services/chatService', () => ({
  listSessions: vi.fn(),
}))

describe('CommandPalette', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    currentTheme = 'light'
    chatService.listSessions.mockResolvedValue({
      sessions: [
        { session_id: 'sess-1', title: 'Financial Overview Q3' },
        { session_id: 'sess-2', title: 'Project Roadmap' },
      ],
      total: 2,
    })
  })

  it('does not render dialog content when open is false', () => {
    render(<CommandPalette open={false} onClose={vi.fn()} />)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('renders dialog and loads sessions when open is true', async () => {
    render(<CommandPalette open={true} onClose={vi.fn()} />)

    expect(screen.getByRole('dialog', { name: 'Command palette' })).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Jump to a page, search conversations...')).toBeInTheDocument()

    expect(screen.getByText('New Chat')).toBeInTheDocument()
    expect(screen.getByText('Documents')).toBeInTheDocument()
    expect(screen.getByText('Upload Document')).toBeInTheDocument()
    expect(screen.getByText('Settings')).toBeInTheDocument()
    expect(screen.getByText('Switch to dark theme')).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('Financial Overview Q3')).toBeInTheDocument()
      expect(screen.getByText('Project Roadmap')).toBeInTheDocument()
    })
  })

  it('filters items dynamically based on search query', async () => {
    render(<CommandPalette open={true} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText('Financial Overview Q3')).toBeInTheDocument()
    })

    const input = screen.getByPlaceholderText('Jump to a page, search conversations...')
    fireEvent.change(input, { target: { value: 'upload' } })

    expect(screen.getByText('Upload Document')).toBeInTheDocument()
    expect(screen.queryByText('New Chat')).not.toBeInTheDocument()
    expect(screen.queryByText('Financial Overview Q3')).not.toBeInTheDocument()

    // Test non-matching query
    fireEvent.change(input, { target: { value: 'nonexistentquery123' } })
    expect(screen.getByText('No matches.')).toBeInTheDocument()
  })

  it('executes static navigation action and closes on click', () => {
    const onClose = vi.fn()
    render(<CommandPalette open={true} onClose={onClose} />)

    const uploadItem = screen.getByText('Upload Document')
    fireEvent.click(uploadItem)

    expect(mockNavigate).toHaveBeenCalledWith('/upload')
    expect(onClose).toHaveBeenCalled()
  })

  it('executes session item navigation with session ID in route state', async () => {
    const onClose = vi.fn()
    render(<CommandPalette open={true} onClose={onClose} />)

    await waitFor(() => {
      expect(screen.getByText('Project Roadmap')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('Project Roadmap'))

    expect(mockNavigate).toHaveBeenCalledWith('/chat', {
      state: { sessionId: 'sess-2' },
    })
    expect(onClose).toHaveBeenCalled()
  })

  it('toggles theme when theme action item is clicked', () => {
    const onClose = vi.fn()
    currentTheme = 'dark'
    render(<CommandPalette open={true} onClose={onClose} />)

    const themeItem = screen.getByText('Switch to light theme')
    fireEvent.click(themeItem)

    expect(mockSetTheme).toHaveBeenCalledWith('light')
    expect(onClose).toHaveBeenCalled()
  })

  it('closes when Escape key is pressed', () => {
    const onClose = vi.fn()
    render(<CommandPalette open={true} onClose={onClose} />)

    const input = screen.getByPlaceholderText('Jump to a page, search conversations...')
    fireEvent.keyDown(input, { key: 'Escape' })

    expect(onClose).toHaveBeenCalled()
  })

  it('supports keyboard navigation via ArrowDown, ArrowUp and Enter', async () => {
    const onClose = vi.fn()
    render(<CommandPalette open={true} onClose={onClose} />)

    const input = screen.getByPlaceholderText('Jump to a page, search conversations...')

    // Arrow down moves to second item ('New Chat')
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(mockNavigate).toHaveBeenCalledWith('/chat')
    expect(onClose).toHaveBeenCalled()
  })
})
