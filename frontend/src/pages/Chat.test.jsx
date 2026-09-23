import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import Chat from './Chat'
import * as useChatHook from '../hooks/useChat'

const mockNavigate = vi.fn()
let mockLocationState = null
let mockLocationSearch = ''
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({
    state: mockLocationState,
    search: mockLocationSearch,
    pathname: '/chat',
    key: 'chat-key-1',
  }),
}))

const mockShowToast = vi.fn()
vi.mock('../hooks/useToast', () => ({
  default: () => ({ showToast: mockShowToast }),
}))

vi.mock('../hooks/useUpload', () => ({
  default: () => ({
    file: null,
    progress: 0,
    status: 'idle',
    selectFile: vi.fn(),
    startUpload: vi.fn(),
    reset: vi.fn(),
  }),
}))

vi.mock('../services/documentService', () => ({
  getDocumentName: vi.fn((id) => (id === 'doc-rag-1' ? 'InsightAI_Overview.pdf' : null)),
}))

vi.mock('../hooks/useChat', () => ({
  default: vi.fn(),
}))

describe('Chat Page', () => {
  const mockAsk = vi.fn()
  const mockRegenerate = vi.fn()
  const mockClearSession = vi.fn()
  const mockLoadSession = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockLocationState = null
    mockLocationSearch = ''

    // Default useChat implementation
    vi.spyOn(useChatHook, 'default').mockReturnValue({
      messages: [],
      isSending: false,
      isLoadingSession: false,
      ask: mockAsk,
      regenerate: mockRegenerate,
      clearSession: mockClearSession,
      loadSession: mockLoadSession,
    })
  })

  it('renders empty state with suggestion pills when no messages exist and no query params', () => {
    render(<Chat />)

    expect(screen.getByText('Ask anything about your documents')).toBeInTheDocument()
    expect(
      screen.getByText('Upload a PDF, then ask questions here. Answers are grounded in the content you\'ve uploaded.')
    ).toBeInTheDocument()

    const suggestionBtn = screen.getByText('Summarize the key points of my uploaded document.')
    expect(suggestionBtn).toBeInTheDocument()

    fireEvent.click(suggestionBtn)
    expect(mockAsk).toHaveBeenCalledWith('Summarize the key points of my uploaded document.', '')
  })

  it('hydrates diagnostic context from URL query params and displays active banner and pre-filled suggestion pill', () => {
    mockLocationSearch = '?crop=tomato&disease=early%20blight&severity=Severe'

    render(<Chat />)

    // Check Diagnostic Context Active banner
    expect(screen.getByTestId('diagnostic-context-banner')).toBeInTheDocument()
    expect(
      screen.getByText(/Diagnosed Leaf Context Active:\s*Tomato\s*-\s*Early Blight\s*\(Severe\)/i)
    ).toBeInTheDocument()

    // Pre-filled suggestion pill
    const suggestionPills = screen.getAllByText(/What is the organic spray rotation schedule for this Early Blight\?/i)
    expect(suggestionPills.length).toBeGreaterThan(0)

    // Click suggestion pill
    fireEvent.click(suggestionPills[0])
    expect(mockAsk).toHaveBeenCalledWith('What is the organic spray rotation schedule for this Early Blight?', '')
  })

  it('allows user to dismiss the diagnostic context banner', () => {
    mockLocationSearch = '?crop=apple&disease=apple%20scab&severity=Moderate'

    render(<Chat />)

    expect(screen.getByTestId('diagnostic-context-banner')).toBeInTheDocument()

    const dismissBtn = screen.getByLabelText(/Dismiss diagnostic context banner/i)
    fireEvent.click(dismissBtn)

    expect(screen.queryByTestId('diagnostic-context-banner')).not.toBeInTheDocument()
  })

  it('submits a new query from ChatInput with selected persona', () => {
    render(<Chat />)

    const textarea = screen.getByPlaceholderText('Ask a question about your documents...')
    fireEvent.change(textarea, { target: { value: 'How does multi-modal indexing work?' } })

    const personaSelect = screen.getByRole('combobox', { name: 'Answer style' })
    fireEvent.change(personaSelect, { target: { value: 'concise' } })

    const sendBtn = screen.getByRole('button', { name: 'Send message' })
    fireEvent.click(sendBtn)

    expect(mockAsk).toHaveBeenCalledWith('How does multi-modal indexing work?', 'concise', 'en')
  })

  it('renders streaming assistant message with loading indicator', () => {
    vi.spyOn(useChatHook, 'default').mockReturnValue({
      messages: [
        { id: 'user-1', role: 'user', content: 'What is hybrid search?' },
        {
          id: 'assistant-1',
          role: 'assistant',
          content: 'Hybrid search combines BM25 keyword matching with dense vectors',
          isStreaming: true,
          sources: [],
          trace: [{ stage: 'retrieving', detail: { k: 4 } }],
        },
      ],
      isSending: true,
      isLoadingSession: false,
      ask: mockAsk,
      regenerate: mockRegenerate,
      clearSession: mockClearSession,
      loadSession: mockLoadSession,
    })

    render(<Chat />)

    expect(screen.getAllByText('What is hybrid search?').length).toBeGreaterThan(0)
    expect(
      screen.getAllByText(/Hybrid search combines BM25 keyword matching with dense vectors/).length
    ).toBeGreaterThan(0)
  })

  it('renders citation source cards and handles expansion', () => {
    vi.spyOn(useChatHook, 'default').mockReturnValue({
      messages: [
        { id: 'user-1', role: 'user', content: 'Explain chunking strategy' },
        {
          id: 'assistant-1',
          role: 'assistant',
          content: 'We use semantic sentence chunking [1].',
          isStreaming: false,
          sources: [
            {
              chunk_id: 'chunk-101',
              document_id: 'doc-rag-1',
              number: 1,
              page_number: 4,
              excerpt: 'Semantic sentence chunking groups related sentences by cosine similarity.',
            },
          ],
          trace: [],
          followUpQuestions: ['What overlap size is used?'],
        },
      ],
      isSending: false,
      isLoadingSession: false,
      ask: mockAsk,
      regenerate: mockRegenerate,
      clearSession: mockClearSession,
      loadSession: mockLoadSession,
    })

    render(<Chat />)

    expect(screen.getAllByText('Explain chunking strategy').length).toBeGreaterThan(0)
    expect(screen.getByText('InsightAI_Overview.pdf · page 4')).toBeInTheDocument()

    // Toggle citation source card expansion
    const sourceToggle = screen.getByRole('button', { name: /InsightAI_Overview\.pdf · page 4/i })
    fireEvent.click(sourceToggle)

    expect(
      screen.getAllByText(/cosine similarity/i).length
    ).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /view in pdf/i })).toBeInTheDocument()

    // Follow up question button
    const followUpBtn = screen.getByRole('button', { name: 'What overlap size is used?' })
    expect(followUpBtn).toBeInTheDocument()
    fireEvent.click(followUpBtn)
    expect(mockAsk).toHaveBeenCalledWith('What overlap size is used?', '', 'en')
  })

  it('triggers regenerate when Regenerate button is clicked', () => {
    vi.spyOn(useChatHook, 'default').mockReturnValue({
      messages: [
        { id: 'u-1', role: 'user', content: 'Tell me a joke' },
        { id: 'a-1', role: 'assistant', content: 'Why did the AI cross the road?', isStreaming: false, sources: [] },
      ],
      isSending: false,
      isLoadingSession: false,
      ask: mockAsk,
      regenerate: mockRegenerate,
      clearSession: mockClearSession,
      loadSession: mockLoadSession,
    })

    render(<Chat />)

    const regenBtn = screen.getByRole('button', { name: /regenerate/i })
    fireEvent.click(regenBtn)
    expect(mockRegenerate).toHaveBeenCalled()
  })

  it('triggers clearSession when "New chat" button is clicked', () => {
    vi.spyOn(useChatHook, 'default').mockReturnValue({
      messages: [{ id: 'u-1', role: 'user', content: 'Existing chat' }],
      isSending: false,
      isLoadingSession: false,
      ask: mockAsk,
      regenerate: mockRegenerate,
      clearSession: mockClearSession,
      loadSession: mockLoadSession,
    })

    render(<Chat />)

    const newChatBtn = screen.getByRole('button', { name: /new chat/i })
    fireEvent.click(newChatBtn)
    expect(mockClearSession).toHaveBeenCalled()
  })

  it('provides speech-to-text dictation button in ChatInput and enables hands-free voice prompt entry', () => {
    render(<Chat />)

    const micBtn = screen.getByTestId('speech-to-text-mic-button')
    expect(micBtn).toBeInTheDocument()
    expect(micBtn).toHaveAttribute('aria-label', 'Start hands-free voice dictation')

    fireEvent.click(micBtn)
    expect(screen.getByText(/Listening\.\.\. Speak clearly/i)).toBeInTheDocument()
  })
})
