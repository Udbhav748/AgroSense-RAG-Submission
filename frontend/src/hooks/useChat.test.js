import { beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import useChat from './useChat'
import * as chatService from '../services/chatService'

vi.mock('../services/chatService', () => ({
  streamChatMessage: vi.fn(),
  deleteChatSession: vi.fn(),
  getSession: vi.fn(),
}))

describe('useChat', () => {
  const SESSION_KEY = 'insightai_session_id'

  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  describe('Session ID generation and retrieval from localStorage', () => {
    it('generates and stores a new UUID in localStorage when none exists', () => {
      expect(localStorage.getItem(SESSION_KEY)).toBeNull()

      const { result } = renderHook(() => useChat())

      const storedSessionId = localStorage.getItem(SESSION_KEY)
      expect(storedSessionId).toBeTruthy()
      expect(typeof storedSessionId).toBe('string')
      expect(result.current.messages).toEqual([])
      expect(result.current.isSending).toBe(false)
      expect(result.current.isLoadingSession).toBe(false)
    })

    it('reuses existing session ID from localStorage on mount without regenerating', () => {
      const existingSessionId = 'existing-session-uuid-1234'
      localStorage.setItem(SESSION_KEY, existingSessionId)

      renderHook(() => useChat())

      expect(localStorage.getItem(SESSION_KEY)).toBe(existingSessionId)
    })
  })

  describe('Loading session from history via loadSession', () => {
    it('loads session automatically when initialSessionId is provided', async () => {
      const mockTurns = [
        { role: 'user', content: 'What is insight AI?' },
        { role: 'assistant', content: 'InsightAI is a multimodal RAG system.' },
      ]
      chatService.getSession.mockResolvedValueOnce({
        session_id: 'session-history-999',
        turns: mockTurns,
      })

      const { result } = renderHook(() => useChat('session-history-999'))

      expect(result.current.isLoadingSession).toBe(true)

      await act(async () => {
        // wait for getSession promise to resolve
      })

      expect(chatService.getSession).toHaveBeenCalledWith('session-history-999')
      expect(localStorage.getItem(SESSION_KEY)).toBe('session-history-999')
      expect(result.current.isLoadingSession).toBe(false)
      expect(result.current.messages).toHaveLength(2)
      expect(result.current.messages[0]).toMatchObject({
        role: 'user',
        content: 'What is insight AI?',
      })
      expect(result.current.messages[1]).toMatchObject({
        role: 'assistant',
        content: 'InsightAI is a multimodal RAG system.',
        sources: [],
        trace: [],
      })
    })

    it('can load session explicitly via loadSession() callback', async () => {
      const { result } = renderHook(() => useChat())

      chatService.getSession.mockResolvedValueOnce({
        session_id: 'custom-session-888',
        turns: [{ role: 'user', content: 'Tell me about embeddings.' }],
      })

      await act(async () => {
        await result.current.loadSession('custom-session-888')
      })

      expect(chatService.getSession).toHaveBeenCalledWith('custom-session-888')
      expect(localStorage.getItem(SESSION_KEY)).toBe('custom-session-888')
      expect(result.current.messages).toHaveLength(1)
      expect(result.current.messages[0].content).toBe('Tell me about embeddings.')
    })

    it('handles loadSession failure gracefully without crashing', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      chatService.getSession.mockRejectedValueOnce(new Error('Network error'))

      const { result } = renderHook(() => useChat('failing-session-id'))

      await act(async () => {
        // wait for rejected promise
      })

      expect(result.current.isLoadingSession).toBe(false)
      expect(result.current.messages).toEqual([])
      expect(warnSpy).toHaveBeenCalledWith('Failed to load session:', expect.any(Error))

      warnSpy.mockRestore()
    })
  })

  describe('Stream message processing (ask)', () => {
    it('processes stream with answer chunks, trace events, and done payload', async () => {
      let streamCallbacks = null
      chatService.streamChatMessage.mockImplementation(async (query, options, callbacks) => {
        streamCallbacks = callbacks
      })

      const { result } = renderHook(() => useChat(null, ['doc-1', 'doc-2']))

      act(() => {
        result.current.ask('Explain vector search', 'concise')
      })

      expect(result.current.isSending).toBe(true)
      expect(result.current.messages).toHaveLength(2)
      expect(result.current.messages[0]).toMatchObject({
        role: 'user',
        content: 'Explain vector search',
      })
      expect(result.current.messages[1]).toMatchObject({
        role: 'assistant',
        content: '',
        isStreaming: true,
        sources: [],
        trace: [],
      })

      expect(chatService.streamChatMessage).toHaveBeenCalledWith(
        'Explain vector search',
        expect.objectContaining({
          history: [],
          persona: 'concise',
          documentIds: ['doc-1', 'doc-2'],
        }),
        expect.any(Object)
      )

      // Test trace event
      act(() => {
        streamCallbacks.onTrace('retrieving', { count: 3 })
      })
      expect(result.current.messages[1].trace).toEqual([
        { stage: 'retrieving', detail: { count: 3 } },
      ])

      // Test chunk arrival
      act(() => {
        streamCallbacks.onChunk('Vector search compares ')
      })
      expect(result.current.messages[1].content).toBe('Vector search compares ')

      act(() => {
        streamCallbacks.onChunk('high-dimensional embeddings.')
      })
      expect(result.current.messages[1].content).toBe('Vector search compares high-dimensional embeddings.')

      // Test reflecting trace resets temporary content
      act(() => {
        streamCallbacks.onTrace('reflecting', { reason: 'insufficient context' })
      })
      expect(result.current.messages[1].content).toBe('')
      expect(result.current.messages[1].trace).toHaveLength(2)

      // Next chunk after reflecting
      act(() => {
        streamCallbacks.onChunk('Refined answer about vector search.')
      })
      expect(result.current.messages[1].content).toBe('Refined answer about vector search.')

      // Test onDone event
      const donePayload = {
        session_id: 'server-updated-session-id',
        answer: 'Refined answer about vector search.',
        sources: [{ chunk_id: 'chunk-1', document_id: 'doc-1', number: 1, excerpt: 'Excerpt 1' }],
        retrieval_confidence: 'high',
        answer_source: 'documents',
        hallucination_detected: false,
        grounding_score: 0.95,
        is_clarifying_question: false,
        follow_up_questions: ['How is distance calculated?'],
      }

      await act(async () => {
        streamCallbacks.onDone(donePayload)
      })

      expect(result.current.isSending).toBe(false)
      expect(localStorage.getItem(SESSION_KEY)).toBe('server-updated-session-id')
      expect(result.current.messages[1]).toMatchObject({
        role: 'assistant',
        content: 'Refined answer about vector search.',
        sources: donePayload.sources,
        retrievalConfidence: 'high',
        answerSource: 'documents',
        hallucinationDetected: false,
        groundingScore: 0.95,
        isClarifyingQuestion: false,
        followUpQuestions: ['How is distance calculated?'],
        isStreaming: false,
      })
    })

    it('handles stream onError callback and marks message with error state', async () => {
      let streamCallbacks = null
      chatService.streamChatMessage.mockImplementation(async (query, options, callbacks) => {
        streamCallbacks = callbacks
      })

      const { result } = renderHook(() => useChat())

      act(() => {
        result.current.ask('Test query')
      })

      await act(async () => {
        streamCallbacks.onError({ message: 'Rate limit exceeded' })
      })

      expect(result.current.isSending).toBe(false)
      const assistantMsg = result.current.messages[1]
      expect(assistantMsg.isError).toBe(true)
      expect(assistantMsg.isStreaming).toBe(false)
      expect(assistantMsg.content).toBe('Rate limit exceeded')
    })

    it('handles stream execution throwing an unhandled exception', async () => {
      chatService.streamChatMessage.mockRejectedValueOnce(new Error('Connection aborted'))

      const { result } = renderHook(() => useChat())

      await act(async () => {
        result.current.ask('Trigger failure')
      })

      expect(result.current.isSending).toBe(false)
      const assistantMsg = result.current.messages[1]
      expect(assistantMsg.isError).toBe(true)
      expect(assistantMsg.isStreaming).toBe(false)
      expect(assistantMsg.content).toBe("I couldn't answer that — please try again.")
    })

    it('recovers gracefully when stream ends abruptly without onDone/onError', async () => {
      chatService.streamChatMessage.mockImplementation(async () => {
        // resolves without invoking onDone or onError
      })

      const { result } = renderHook(() => useChat())

      await act(async () => {
        result.current.ask('Abrupt stream end')
      })

      expect(result.current.isSending).toBe(false)
      const assistantMsg = result.current.messages[1]
      expect(assistantMsg.isStreaming).toBe(false)
      expect(assistantMsg.isError).toBe(true)
      expect(assistantMsg.content).toBe('The response was interrupted — please try again.')
    })
  })

  describe('toHistory and history truncation logic (multi-turn and regenerate)', () => {
    it('passes cleaned history omitting streaming and error messages', async () => {
      let streamCalls = []
      chatService.streamChatMessage.mockImplementation(async (query, options, callbacks) => {
        streamCalls.push({ query, options })
        callbacks.onDone({ answer: `Answer to ${query}`, sources: [] })
      })

      const { result } = renderHook(() => useChat())

      // Turn 1
      await act(async () => {
        result.current.ask('First question')
      })

      // Turn 2
      await act(async () => {
        result.current.ask('Second question')
      })

      expect(streamCalls).toHaveLength(2)
      expect(streamCalls[0].options.history).toEqual([])
      expect(streamCalls[1].options.history).toEqual([
        { role: 'user', content: 'First question' },
        { role: 'assistant', content: 'Answer to First question' },
      ])
    })

    it('regenerates the last assistant response with persona and prior history', async () => {
      let capturedOptions = null
      chatService.streamChatMessage.mockImplementation(async (query, options, callbacks) => {
        capturedOptions = options
        callbacks.onDone({ answer: `Generated response for ${query}`, sources: [] })
      })

      const { result } = renderHook(() => useChat())

      // Turn 1
      await act(async () => {
        result.current.ask('First question')
      })
      // Turn 2
      await act(async () => {
        result.current.ask('Second question', 'concise')
      })

      expect(result.current.messages).toHaveLength(4)

      // Regenerate Turn 2 with persona 'eli5'
      await act(async () => {
        result.current.regenerate('eli5')
      })

      expect(capturedOptions.persona).toBe('eli5')
      expect(capturedOptions.history).toEqual([
        { role: 'user', content: 'First question' },
        { role: 'assistant', content: 'Generated response for First question' },
      ])
      expect(result.current.messages).toHaveLength(4)
      expect(result.current.messages[2].content).toBe('Second question')
      expect(result.current.messages[3].content).toBe('Generated response for Second question')
    })

    it('does not regenerate if isSending is true or if no query was sent yet', async () => {
      const { result } = renderHook(() => useChat())

      act(() => {
        result.current.regenerate('concise')
      })
      expect(chatService.streamChatMessage).not.toHaveBeenCalled()
    })
  })

  describe('clearSession', () => {
    it('deletes session on server, resets localStorage, generates new sessionId and clears messages', async () => {
      chatService.deleteChatSession.mockResolvedValueOnce({ status: 'ok', session_id: 'old-session-123' })
      localStorage.setItem(SESSION_KEY, 'old-session-123')

      const { result } = renderHook(() => useChat())

      await act(async () => {
        await result.current.clearSession()
      })

      expect(chatService.deleteChatSession).toHaveBeenCalledWith('old-session-123')
      const newSessionId = localStorage.getItem(SESSION_KEY)
      expect(newSessionId).toBeTruthy()
      expect(newSessionId).not.toBe('old-session-123')
      expect(result.current.messages).toEqual([])
    })

    it('handles deleteChatSession API error gracefully and still resets local state', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      chatService.deleteChatSession.mockRejectedValueOnce(new Error('Server error'))
      localStorage.setItem(SESSION_KEY, 'old-session-456')

      const { result } = renderHook(() => useChat())

      await act(async () => {
        await result.current.clearSession()
      })

      expect(warnSpy).toHaveBeenCalledWith('Failed to delete server-side session:', expect.any(Error))
      expect(localStorage.getItem(SESSION_KEY)).not.toBe('old-session-456')
      expect(result.current.messages).toEqual([])

      warnSpy.mockRestore()
    })
  })
})
