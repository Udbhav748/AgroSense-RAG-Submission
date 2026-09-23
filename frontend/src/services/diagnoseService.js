import api, { AUTH_TOKEN_KEY } from './api'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

/**
 * Diagnose a plant leaf photo via POST /chat/diagnose.
 *
 * The backend sends the image to the LeafSense vision service, then runs the
 * predicted disease through the same corrective-RAG loop as /chat — so the
 * response is a full ChatResponse with answer, sources, and a diagnosis
 * block describing the vision prediction itself.
 *
 * @param {File} image An image file (from camera capture or file picker).
 * @param {{ query?: string, sessionId?: string, confirmWebSearch?: boolean }} [options]
 * @returns {Promise<{answer: string, retrieved_chunks: object[], sources: object[], processing_time: number, tool_used: string, steps_taken: number, diagnosis: {raw_class: string, crop: string, disease: string, confidence: number, low_confidence: boolean} | null, session_id: string}>}
 */
export async function diagnoseLeaf(image, options = {}) {
  const formData = new FormData()
  formData.append('image', image)
  if (options.query?.trim()) formData.append('query', options.query.trim())
  if (options.sessionId) formData.append('session_id', options.sessionId)
  if (options.confirmWebSearch) formData.append('confirm_web_search', String(options.confirmWebSearch))
  if (options.engine) formData.append('engine', options.engine)
  if (options.language) formData.append('language', options.language)

  // Vision inference + retrieval + generation can be slow, so this gets the
  // same longer timeout as /chat rather than the shared api instance default.
  const { data } = await api.post('/chat/diagnose', formData, { timeout: 60000 })
  return data
}

/**
 * Diagnose a plant leaf photo via SSE streaming from POST /chat/diagnose/stream,
 * falling back to non-streaming POST /chat/diagnose if the stream endpoint is unavailable.
 *
 * SSE event types emitted to onEvent:
 * - 'diagnosis': { type: 'diagnosis', diagnosis: { raw_class, crop, disease, confidence, low_confidence } }
 * - 'trace': { type: 'trace', stage: string, detail: object }
 * - 'answer_chunk': { type: 'answer_chunk', text: string }
 * - 'done': { type: 'done', payload: object }
 * - 'error': { type: 'error', detail: object }
 *
 * @param {File} file An image file (from camera capture or file picker).
 * @param {string|{ query?: string, sessionId?: string, confirmWebSearch?: boolean, engine?: string, language?: string }} [queryOrOptions]
 * @param {(event: { type: string, [key: string]: any }) => void} [onEvent]
 * @param {(error: any) => void} [onError]
 * @returns {Promise<void>}
 */
export async function diagnoseImageStream(file, queryOrOptions = '', onEvent, onError) {
  let query = ''
  let sessionId = undefined
  let confirmWebSearch = false
  let engine = 'hybrid'
  let language = 'en'

  if (typeof queryOrOptions === 'string') {
    query = queryOrOptions
  } else if (queryOrOptions && typeof queryOrOptions === 'object') {
    query = queryOrOptions.query || ''
    sessionId = queryOrOptions.sessionId
    confirmWebSearch = queryOrOptions.confirmWebSearch
    engine = queryOrOptions.engine || 'hybrid'
    language = queryOrOptions.language || 'en'
  }

  const formData = new FormData()
  formData.append('image', file)
  if (query?.trim()) formData.append('query', query.trim())
  if (sessionId) formData.append('session_id', sessionId)
  if (confirmWebSearch) formData.append('confirm_web_search', String(confirmWebSearch))
  if (engine) formData.append('engine', engine)
  if (language) formData.append('language', language)

  const token = typeof window !== 'undefined' ? window.localStorage?.getItem(AUTH_TOKEN_KEY) : null

  async function fallbackNonStreaming() {
    try {
      const data = await diagnoseLeaf(file, { query, sessionId, confirmWebSearch, engine, language })
      if (data?.diagnosis) {
        onEvent?.({ type: 'diagnosis', diagnosis: data.diagnosis, payload: data.diagnosis })
      }
      if (data?.answer) {
        onEvent?.({ type: 'answer_chunk', text: data.answer })
      }
      onEvent?.({ type: 'done', payload: data })
      return data
    } catch (err) {
      onError?.(err)
      throw err
    }
  }

  let response
  try {
    response = await fetch(`${API_BASE_URL}/chat/diagnose/stream`, {
      method: 'POST',
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: formData,
    })
  } catch {
    // If fetch failed (network error, offline, or endpoint not implemented), fallback to non-streaming
    return await fallbackNonStreaming()
  }

  if (response.status === 404 || response.status === 405 || response.status === 501) {
    return await fallbackNonStreaming()
  }

  if (!response.ok || !response.body) {
    if (response.status === 401 && typeof window !== 'undefined' && window.location?.pathname !== '/login') {
      window.localStorage?.removeItem(AUTH_TOKEN_KEY)
      window.location.href = '/login'
    }
    const errData = await response.json().catch(() => null)
    const err = new Error(errData?.detail || `Request failed with status ${response.status}`)
    err.response = { status: response.status, data: errData }
    onError?.(errData || { message: err.message, status_code: response.status })
    throw err
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)

      const data = rawEvent
        .split('\n')
        .filter((line) => line.startsWith('data: '))
        .map((line) => line.slice('data: '.length))
        .join('\n')

      if (data) {
        try {
          const event = JSON.parse(data)
          onEvent?.(event)
          if (event.type === 'error') {
            onError?.(event.detail || event)
          }
        } catch {
          onError?.({
            error_type: 'stream_parse_error',
            message: 'Received an unexpected message from the server.',
            status_code: 0,
          })
        }
      }

      boundary = buffer.indexOf('\n\n')
    }
  }
}

/**
 * Check if the LeafSense vision service (port 8001) or vision pipeline is reachable.
 * @returns {Promise<{online: boolean, port: number, url: string, message?: string}>}
 */
export async function checkLeafSenseHealth() {
  try {
    const { data } = await api.get('/health/vision', { timeout: 3000 })
    return {
      online: Boolean(data?.online || data?.has_gemini_fallback),
      port: 8001,
      url: data?.url || 'http://localhost:8001',
      status: data?.status || 'ok',
      hasGeminiFallback: Boolean(data?.has_gemini_fallback),
    }
  } catch {
    // Direct browser probe fallback
    try {
      const direct = await fetch('http://localhost:8001/model-info', { method: 'GET', signal: AbortSignal.timeout(2000) })
      if (direct.ok) {
        return { online: true, port: 8001, url: 'http://localhost:8001', status: 'online' }
      }
    } catch {
      // offline
    }
    return {
      online: false,
      port: 8001,
      url: 'http://localhost:8001',
      message: 'LeafSense service offline on port 8001',
    }
  }
}
