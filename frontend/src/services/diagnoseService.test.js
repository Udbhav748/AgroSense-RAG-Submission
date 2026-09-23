import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  checkLeafSenseHealth,
  diagnoseImageStream,
  diagnoseLeaf,
} from './diagnoseService'
import api, { AUTH_TOKEN_KEY } from './api'

vi.mock('./api', () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
  },
  AUTH_TOKEN_KEY: 'test_auth_token',
}))

describe('diagnoseService', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    globalThis.fetch = vi.fn()
    window.localStorage.clear()
  })

  describe('diagnoseLeaf', () => {
    it('sends multipart form data to POST /chat/diagnose', async () => {
      const mockResult = {
        answer: 'Early blight detected',
        diagnosis: { crop: 'tomato', disease: 'early blight', confidence: 0.95 },
      }
      api.post.mockResolvedValueOnce({ data: mockResult })

      const file = new File(['fake-image'], 'leaf.jpg', { type: 'image/jpeg' })
      const result = await diagnoseLeaf(file, {
        query: 'Check yellow spots',
        sessionId: 'session-123',
        confirmWebSearch: true,
      })

      expect(api.post).toHaveBeenCalledWith(
        '/chat/diagnose',
        expect.any(FormData),
        { timeout: 60000 }
      )
      expect(result).toEqual(mockResult)
    })
  })

  describe('diagnoseImageStream', () => {
    function createStreamResponse(events) {
      const ssePayload = events
        .map((evt) => `data: ${JSON.stringify(evt)}\n\n`)
        .join('')
      const encoder = new TextEncoder()
      const uint8 = encoder.encode(ssePayload)

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(uint8)
          controller.close()
        },
      })

      return {
        ok: true,
        status: 200,
        body: stream,
      }
    }

    it('parses SSE events (diagnosis, answer_chunk, done) progressively', async () => {
      const sseEvents = [
        {
          type: 'diagnosis',
          diagnosis: { crop: 'tomato', disease: 'early blight', confidence: 0.96 },
        },
        {
          type: 'answer_chunk',
          text: 'Foliar lesions observed. ',
        },
        {
          type: 'answer_chunk',
          text: 'Treatment: Copper spray.',
        },
        {
          type: 'done',
          payload: {
            answer: 'Foliar lesions observed. Treatment: Copper spray.',
            sources: [{ chunk_id: '1' }],
          },
        },
      ]

      globalThis.fetch.mockResolvedValueOnce(createStreamResponse(sseEvents))

      const file = new File(['fake-bytes'], 'leaf.png', { type: 'image/png' })
      const receivedEvents = []

      await diagnoseImageStream(
        file,
        'Test query',
        (evt) => receivedEvents.push(evt),
        vi.fn()
      )

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/chat/diagnose/stream',
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData),
        })
      )

      expect(receivedEvents).toHaveLength(4)
      expect(receivedEvents[0]).toEqual(sseEvents[0])
      expect(receivedEvents[1]).toEqual(sseEvents[1])
      expect(receivedEvents[2]).toEqual(sseEvents[2])
      expect(receivedEvents[3]).toEqual(sseEvents[3])
    })

    it('falls back to non-streaming POST /chat/diagnose when /stream returns 404', async () => {
      globalThis.fetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      })

      const fallbackResult = {
        answer: 'Non-streaming fallback answer',
        diagnosis: { crop: 'tomato', disease: 'healthy', confidence: 0.99 },
        sources: [],
      }
      api.post.mockResolvedValueOnce({ data: fallbackResult })

      const file = new File(['fake-bytes'], 'leaf.png', { type: 'image/png' })
      const receivedEvents = []

      await diagnoseImageStream(
        file,
        { query: 'Fallback query', sessionId: 'sess-456' },
        (evt) => receivedEvents.push(evt),
        vi.fn()
      )

      expect(api.post).toHaveBeenCalledWith(
        '/chat/diagnose',
        expect.any(FormData),
        { timeout: 60000 }
      )
      expect(receivedEvents).toEqual([
        {
          type: 'diagnosis',
          diagnosis: fallbackResult.diagnosis,
          payload: fallbackResult.diagnosis,
        },
        {
          type: 'answer_chunk',
          text: fallbackResult.answer,
        },
        {
          type: 'done',
          payload: fallbackResult,
        },
      ])
    })

    it('falls back to non-streaming POST /chat/diagnose when fetch encounters a network error', async () => {
      globalThis.fetch.mockRejectedValueOnce(new Error('Network offline'))

      const fallbackResult = {
        answer: 'Recovered via fallback',
        diagnosis: { crop: 'potato', disease: 'late blight', confidence: 0.88 },
      }
      api.post.mockResolvedValueOnce({ data: fallbackResult })

      const file = new File(['fake-bytes'], 'leaf.png', { type: 'image/png' })
      const receivedEvents = []

      await diagnoseImageStream(
        file,
        '',
        (evt) => receivedEvents.push(evt),
        vi.fn()
      )

      expect(receivedEvents[0].type).toBe('diagnosis')
      expect(receivedEvents[1].type).toBe('answer_chunk')
      expect(receivedEvents[2].type).toBe('done')
    })
  })

  describe('checkLeafSenseHealth', () => {
    it('returns online true when /health/vision returns online ok', async () => {
      api.get.mockResolvedValueOnce({
        data: { online: true, status: 'ok', has_gemini_fallback: false },
      })

      const health = await checkLeafSenseHealth()
      expect(health.online).toBe(true)
      expect(health.port).toBe(8001)
      expect(health.status).toBe('ok')
    })

    it('returns offline when probe fails', async () => {
      api.get.mockRejectedValueOnce(new Error('Vision offline'))
      globalThis.fetch.mockRejectedValueOnce(new Error('Connection refused'))

      const health = await checkLeafSenseHealth()
      expect(health.online).toBe(false)
      expect(health.port).toBe(8001)
    })
  })
})
