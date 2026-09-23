import { useCallback, useEffect, useRef, useState } from 'react'
import {
  streamChatMessage,
  streamAgentGraphChatMessage,
  deleteChatSession,
  getSession,
} from '../services/chatService'
import getErrorMessage from '../utils/errorMessage'

export const DEFAULT_GRAPH_NODE_STATES = {
  planner: { status: 'idle', duration_ms: null, output: null },
  document_analyst: { status: 'idle', duration_ms: null, output: null },
  synthesizer: { status: 'idle', duration_ms: null, output: null },
  fact_checker: { status: 'idle', duration_ms: null, output: null },
}

let idCounter = 0
const nextId = () => `msg-${++idCounter}-${Date.now()}`

const SESSION_KEY = 'insightai_session_id'

// crypto.randomUUID() is restricted to secure contexts (HTTPS or
// localhost) — a plain-HTTP deployment (e.g. by IP address, no TLS)
// throws "crypto.randomUUID is not a function" there. crypto.getRandomValues()
// has no such restriction, so build a UUID v4 from it by hand as the
// fallback; only a session identifier, not a security boundary, so
// Math.random() as a last-resort fallback (no Web Crypto at all) is fine too.
function generateUUID() {
  if (typeof crypto?.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  const bytes = new Uint8Array(16)
  if (typeof crypto?.getRandomValues === 'function') {
    crypto.getRandomValues(bytes)
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256)
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0'))
  return `${hex.slice(0, 4).join('')}-${hex.slice(4, 6).join('')}-${hex.slice(6, 8).join('')}-${hex.slice(8, 10).join('')}-${hex.slice(10, 16).join('')}`
}

// Backend only wants role/content pairs (see ChatRequest.history) — strip
// UI-only fields, and drop error/in-progress bubbles since they aren't
// real answers.
function toHistory(messages) {
  return messages
    .filter((message) => !message.isError && !message.isStreaming)
    .map((message) => ({ role: message.role, content: message.content }))
}

// History for a regenerate() should be everything before the user turn
// being replayed, not including that turn itself (it's sent separately
// as the query).
function historyBeforeQuery(messages, query) {
  let cutoff = messages.length
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'user' && messages[index].content === query) {
      cutoff = index
      break
    }
  }
  return toHistory(messages.slice(0, cutoff))
}

function getOrCreateSessionId() {
  let sessionId = localStorage.getItem(SESSION_KEY)
  if (!sessionId) {
    sessionId = generateUUID()
    localStorage.setItem(SESSION_KEY, sessionId)
  }
  return sessionId
}

export default function useChat(initialSessionId, initialDocumentIds) {
  const [messages, setMessages] = useState([])
  const [isSending, setIsSending] = useState(false)
  const [isLoadingSession, setIsLoadingSession] = useState(Boolean(initialSessionId))
  const [isGraphMode, setIsGraphMode] = useState(false)
  const [graphNodeStates, setGraphNodeStates] = useState(DEFAULT_GRAPH_NODE_STATES)
  const [activeGraphNode, setActiveGraphNode] = useState(null)
  const lastQueryRef = useRef('')
  const messagesRef = useRef([])
  const sessionIdRef = useRef(null)
  const documentIdsRef = useRef(initialDocumentIds ?? null)
  // Keep the ref in sync with the latest initialDocumentIds — a caller can
  // pass a different scoped collection than the one this hook instance was
  // constructed with (e.g. navigating via a different link), and without this
  // the stale mount-time value would scope every retrieval to the wrong set.
  useEffect(() => {
    documentIdsRef.current = initialDocumentIds ?? null
  }, [initialDocumentIds])

  // Resume a past conversation (opened from the History page) instead of
  // starting/continuing the localStorage-tracked session — additive to
  // the existing flow below, which is unchanged when no id is passed.
  const loadSession = useCallback(async (sessionId) => {
    setIsLoadingSession(true)
    try {
      const data = await getSession(sessionId)
      sessionIdRef.current = sessionId
      localStorage.setItem(SESSION_KEY, sessionId)
      setMessages(
        data.turns.map((turn) => ({
          id: nextId(),
          role: turn.role,
          content: turn.content,
          sources: [],
          trace: [],
        }))
      )
    } catch (error) {
      console.warn('Failed to load session:', error)
    } finally {
      setIsLoadingSession(false)
    }
  }, [])

  // Initialize session_id on mount: resume initialSessionId if given
  // (navigated here from History), otherwise the existing
  // localStorage-tracked session/fresh-session behavior, unchanged.
  useEffect(() => {
    if (initialSessionId) {
      loadSession(initialSessionId)
    } else {
      sessionIdRef.current = getOrCreateSessionId()
    }
    // Only ever run once per mount — Chat.jsx already remounts this
    // hook via location.key on navigation, so initialSessionId/loadSession
    // changing identity across re-renders (without a real navigation)
    // must not re-trigger a reload mid-conversation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  const updateMessage = useCallback((id, updater) => {
    setMessages((current) => current.map((message) => (message.id === id ? updater(message) : message)))
  }, [])

  const fetchAnswer = useCallback(
    async (query, history, persona, language) => {
      setIsSending(true)
      const assistantId = nextId()
      setMessages((current) => [
        ...current,
        { id: assistantId, role: 'assistant', content: '', sources: [], trace: [], isStreaming: true },
      ])

      // Whether the stream reached a terminal event (done/error). If the
      // connection ends without either (e.g. the backend dropped the
      // stream mid-answer), we still resolve the bubble in the finally
      // block rather than leaving it spinning as isStreaming forever.
      let reachedTerminal = false

      try {
        if (isGraphMode) {
          setGraphNodeStates(DEFAULT_GRAPH_NODE_STATES)
          setActiveGraphNode('planner')

          await streamAgentGraphChatMessage(
            query,
            { history, sessionId: sessionIdRef.current, persona, documentIds: documentIdsRef.current, language },
            {
              onNodeStart: (node, timestamp) => {
                setActiveGraphNode(node)
                setGraphNodeStates((current) => ({
                  ...current,
                  [node]: { ...(current[node] || {}), status: 'running', timestamp },
                }))
              },
              onNodeComplete: (node, output, duration_ms) => {
                const isFailed = Boolean(output?.error)
                setGraphNodeStates((current) => ({
                  ...current,
                  [node]: {
                    ...(current[node] || {}),
                    status: isFailed ? 'failed' : 'completed',
                    output,
                    duration_ms,
                  },
                }))
              },
              onChunk: (text) => {
                updateMessage(assistantId, (message) => ({ ...message, content: message.content + text }))
              },
              onDone: (finalState) => {
                reachedTerminal = true
                setActiveGraphNode(null)
                if (finalState.session_id) {
                  sessionIdRef.current = finalState.session_id
                  localStorage.setItem(SESSION_KEY, finalState.session_id)
                }
                const answer = finalState.draft_answer || finalState.final_response?.answer || ''
                const sources = finalState.final_response?.sources ?? []
                const factCheck = finalState.fact_check_result ?? {}
                const score =
                  typeof factCheck === 'object' && factCheck !== null && factCheck.score != null
                    ? factCheck.score
                    : null
                const isGrounded =
                  typeof factCheck === 'object' && factCheck !== null
                    ? factCheck.verified ?? true
                    : Boolean(factCheck)

                updateMessage(assistantId, (message) => ({
                  ...message,
                  content: answer,
                  sources,
                  retrievalConfidence: finalState.final_response?.retrieval_confidence,
                  answerSource: finalState.final_response?.answer_source ?? 'documents',
                  hallucinationDetected: !isGrounded,
                  groundingScore: score,
                  isStreaming: false,
                }))
              },
              onError: (detail) => {
                reachedTerminal = true
                setActiveGraphNode(null)
                updateMessage(assistantId, (message) => ({
                  ...message,
                  content: getErrorMessage(
                    { response: { data: { detail: detail?.message } } },
                    "I couldn't answer that — please try again."
                  ),
                  isError: true,
                  isStreaming: false,
                }))
              },
            }
          )
        } else {
          await streamChatMessage(
            query,
            { history, sessionId: sessionIdRef.current, persona, documentIds: documentIdsRef.current, language },
            {
              onTrace: (stage, detail) => {
                updateMessage(assistantId, (message) => ({
                  ...message,
                  // A "reflecting" stage means the pipeline is discarding
                  // the current attempt and starting a fresh generation —
                  // any text streamed so far belongs to that discarded
                  // attempt, so the visible answer resets alongside it
                  // (see backend rag_service.py's stream_query docstring).
                  content: stage === 'reflecting' ? '' : message.content,
                  trace: [...message.trace, { stage, detail }],
                }))
              },
              onChunk: (text) => {
                updateMessage(assistantId, (message) => ({ ...message, content: message.content + text }))
              },
              onDone: (payload) => {
                reachedTerminal = true
                // Persist session_id returned by server (new or echoed)
                if (payload.session_id) {
                  sessionIdRef.current = payload.session_id
                  localStorage.setItem(SESSION_KEY, payload.session_id)
                }
                updateMessage(assistantId, (message) => ({
                  ...message,
                  content: payload.answer,
                  sources: payload.sources ?? [],
                  retrievalConfidence: payload.retrieval_confidence,
                  answerSource: payload.answer_source ?? 'documents',
                  hallucinationDetected: payload.hallucination_detected ?? false,
                  groundingScore: payload.grounding_score ?? null,
                  isClarifyingQuestion: payload.is_clarifying_question ?? false,
                  followUpQuestions: payload.follow_up_questions ?? [],
                  isStreaming: false,
                }))
              },
              onError: (detail) => {
                reachedTerminal = true
                updateMessage(assistantId, (message) => ({
                  ...message,
                  content: getErrorMessage(
                    { response: { data: { detail: detail?.message } } },
                    "I couldn't answer that — please try again."
                  ),
                  isError: true,
                  isStreaming: false,
                }))
              },
            }
          )
        }
      } catch (error) {
        reachedTerminal = true
        setActiveGraphNode(null)
        updateMessage(assistantId, (message) => ({
          ...message,
          content: getErrorMessage(error, "I couldn't answer that — please try again."),
          isError: true,
          isStreaming: false,
        }))
      } finally {
        setIsSending(false)
        setActiveGraphNode(null)
        // Stream ended without a done/error event: stop the spinner and,
        // if nothing at all was received, show a degraded notice instead
        // of a permanently-empty bubble.
        if (!reachedTerminal) {
          updateMessage(assistantId, (message) => {
            if (!message.isStreaming) return message
            return {
              ...message,
              content: message.content || getErrorMessage(
                {},
                "The response was interrupted — please try again."
              ),
              isError: message.content ? false : true,
              isStreaming: false,
            }
          })
        }
      }
    },
    [updateMessage, isGraphMode]
  )

  const ask = useCallback(
    (query, persona, language) => {
      lastQueryRef.current = query
      const history = toHistory(messagesRef.current)
      setMessages((current) => [...current, { id: nextId(), role: 'user', content: query }])
      fetchAnswer(query, history, persona, language)
    },
    [fetchAnswer]
  )

  const regenerate = useCallback((persona, language) => {
    if (!lastQueryRef.current || isSending) return
    const history = historyBeforeQuery(messagesRef.current, lastQueryRef.current)
    setMessages((current) => {
      const lastIndex = current.length - 1
      if (lastIndex >= 0 && current[lastIndex].role === 'assistant') {
        return current.slice(0, lastIndex)
      }
      return current
    })
    // Reuse the persona that was active for the original turn. ask() already
    // forwards it from ChatInput; regenerate did not, so a regenerate silently
    // dropped personality and fell back to the default prompt.
    fetchAnswer(lastQueryRef.current, history, persona, language)
  }, [fetchAnswer, isSending])

  // Clear session (start fresh conversation)
  const clearSession = useCallback(async () => {
    const oldSessionId = sessionIdRef.current
    if (oldSessionId) {
      try {
        await deleteChatSession(oldSessionId)
      } catch (err) {
        // Log but don't block UI — server-side cleanup is best-effort
        console.warn('Failed to delete server-side session:', err)
      }
    }
    localStorage.removeItem(SESSION_KEY)
    sessionIdRef.current = getOrCreateSessionId()
    setMessages([])
    setGraphNodeStates(DEFAULT_GRAPH_NODE_STATES)
    setActiveGraphNode(null)
  }, [])

  return {
    messages,
    isSending,
    isLoadingSession,
    ask,
    regenerate,
    clearSession,
    loadSession,
    isGraphMode,
    setIsGraphMode,
    graphNodeStates,
    setGraphNodeStates,
    activeGraphNode,
  }
}
