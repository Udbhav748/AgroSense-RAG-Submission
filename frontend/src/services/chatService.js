import api, { AUTH_TOKEN_KEY } from './api'

// Mirrors api.js's own fallback/env resolution — fetch-based SSE can't
// reuse the axios instance directly (axios has no streaming-body request
// mode), so the base URL is resolved the same way here rather than
// reaching into api.defaults' internal shape. The auth token is read
// fresh per call (same reasoning as api.js's request interceptor: a
// login/logout between calls must take effect immediately).
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

/**
 * Assemble the JSON body for a chat request (POST /chat and /chat/stream),
 * mapping the camelCase option names the hooks pass into the snake_case
 * fields the backend's ChatRequest schema expects. Falsy optionals are
 * omitted so the server applies its own defaults (top_k, min_score, etc.).
 * @param {string} query
 * @param {{ topK?: number, minScore?: number, history?: {role: string, content: string}[], sessionId?: string, persona?: string, documentIds?: string[], confirmWebSearch?: boolean, structuredResponse?: boolean }} [options]
 * @returns {object} The serialized ChatRequest payload.
 */
function buildChatPayload(query, options) {
  const payload = { query }
  if (options.topK != null) payload.top_k = options.topK
  if (options.minScore != null) payload.min_score = options.minScore
  if (options.history?.length) payload.history = options.history
  if (options.sessionId) payload.session_id = options.sessionId
  if (options.persona) payload.persona = options.persona
  if (options.documentIds?.length) payload.document_ids = options.documentIds
  if (options.confirmWebSearch != null) payload.confirm_web_search = options.confirmWebSearch
  if (options.structuredResponse != null) payload.structured_response = options.structuredResponse
  if (options.language) payload.language = options.language
  return payload
}

/**
 * Send a chat query to POST /chat/stream and consume the Server-Sent
 * Events response as it arrives.
 *
 * Uses fetch + a manually-parsed ReadableStream rather than the browser's
 * EventSource: EventSource only supports GET with no custom headers, and
 * this endpoint needs POST (a query body) plus the Authorization header
 * (the JWT, same as every authenticated call — read fresh per request so
 * a login/logout between calls takes effect immediately).
 *
 * @param {string} query
 * @param {{ topK?: number, minScore?: number, history?: {role: string, content: string}[], sessionId?: string, persona?: string, documentIds?: string[], confirmWebSearch?: boolean, structuredResponse?: boolean }} [options]
 * @param {{
 *   onTrace?: (stage: string, detail: object) => void,
 *   onChunk?: (text: string) => void,
 *   onDone?: (payload: object) => void,
 *   onError?: (detail: { error_type: string, message: string, status_code: number }) => void,
 * }} [callbacks]
 */
export async function streamChatMessage(query, options = {}, callbacks = {}) {
  const { onTrace, onChunk, onDone, onError } = callbacks

  const token = window.localStorage.getItem(AUTH_TOKEN_KEY)
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(buildChatPayload(query, options)),
  })

  if (!response.ok || !response.body) {
    // streamChatMessage uses raw fetch(), so it bypasses the shared api
    // instance's axios 401 interceptor (clear token + redirect to /login).
    // An expired/invalid token here must not surface as a generic chat
    // error that leaves the user stuck on a broken session — mirror the
    // exact 401 contract every other request path follows.
    if (response.status === 401 && window.location.pathname !== '/login') {
      window.localStorage.removeItem(AUTH_TOKEN_KEY)
      window.location.href = '/login'
    }
    const error = new Error(`Request failed with status ${response.status}`)
    error.response = { status: response.status, data: await response.json().catch(() => null) }
    throw error
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by a blank line; a single read() can
    // contain zero, one, or several complete frames, and a frame can also
    // be split across reads — buffer and only consume complete frames.
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
        // Isolate per-frame parse failures: a single malformed SSE frame
        // shouldn't abort the whole stream and discard a perfectly good
        // in-progress answer — route the failure to the consumer's error
        // channel and keep draining the response.
        try {
          const event = JSON.parse(data)
          if (event.type === 'trace') onTrace?.(event.stage, event.detail)
          else if (event.type === 'answer_chunk') onChunk?.(event.text)
          else if (event.type === 'error') onError?.(event.detail)
          else if (event.type === 'done') onDone?.(event.payload)
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
 * Send a chat query to POST /chat/agent-graph/stream and consume the multi-agent
 * StateGraph Server-Sent Events as node transitions and tokens arrive.
 *
 * @param {string} query
 * @param {{ topK?: number, minScore?: number, history?: {role: string, content: string}[], sessionId?: string, persona?: string, documentIds?: string[], confirmWebSearch?: boolean, structuredResponse?: boolean }} [options]
 * @param {{
 *   onNodeStart?: (node: string, timestamp: number) => void,
 *   onNodeComplete?: (node: string, output: object, duration_ms: number) => void,
 *   onChunk?: (text: string) => void,
 *   onDone?: (finalState: object) => void,
 *   onError?: (detail: { error_type: string, message: string, status_code: number }) => void,
 * }} [callbacks]
 */
export async function streamAgentGraphChatMessage(query, options = {}, callbacks = {}) {
  const { onNodeStart, onNodeComplete, onChunk, onDone, onError } = callbacks

  const token = window.localStorage.getItem(AUTH_TOKEN_KEY)
  const response = await fetch(`${API_BASE_URL}/chat/agent-graph/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(buildChatPayload(query, options)),
  })

  if (!response.ok || !response.body) {
    if (response.status === 401 && window.location.pathname !== '/login') {
      window.localStorage.removeItem(AUTH_TOKEN_KEY)
      window.location.href = '/login'
    }
    const error = new Error(`Request failed with status ${response.status}`)
    error.response = { status: response.status, data: await response.json().catch(() => null) }
    throw error
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
          if (event.type === 'node_start') onNodeStart?.(event.node, event.timestamp)
          else if (event.type === 'node_complete') onNodeComplete?.(event.node, event.output, event.duration_ms)
          else if (event.type === 'token') onChunk?.(event.text)
          else if (event.type === 'graph_done') onDone?.(event.final_state)
          else if (event.type === 'error') onError?.(event.detail)
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
 * Delete the server-side chat session by id (DELETE /chat/sessions/{id}).
 * @param {string} sessionId
 * @returns {Promise<{status: string, session_id: string}>}
 */
export async function deleteChatSession(sessionId) {
  const { data } = await api.delete(`/chat/sessions/${encodeURIComponent(sessionId)}`)
  return data
}

/**
 * List the caller's own past conversations (title, timestamps),
 * newest-accessed first — the history sidebar's data source.
 * @returns {Promise<{sessions: {session_id: string, title: string|null, created_at: string|null, last_accessed_at: string|null}[], total: number}>}
 */
export async function listSessions() {
  const { data } = await api.get('/chat/sessions')
  return data
}

/**
 * Full turn history for one session — used to resume a past
 * conversation from the history list.
 * @param {string} sessionId
 * @returns {Promise<{session_id: string, turns: {role: string, content: string}[]}>}
 */
export async function getSession(sessionId) {
  const { data } = await api.get(`/chat/sessions/${encodeURIComponent(sessionId)}`)
  return data
}
