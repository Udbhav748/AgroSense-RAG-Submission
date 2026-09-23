import { useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Download,
  FileDown,
  GitFork,
  Leaf,
  MessageCircle,
  Plus,
  Printer,
  Sparkles,
  Sprout,
  UploadCloud,
  X,
} from 'lucide-react'
import AgentGraphVisualizer from '../components/chat/AgentGraphVisualizer'
import ChatBubble from '../components/chat/ChatBubble'
import ChatInput from '../components/chat/ChatInput'
import EmptyState from '../components/ui/EmptyState'
import ProgressBar from '../components/ui/ProgressBar'
import useChat from '../hooks/useChat'
import useUpload from '../hooks/useUpload'
import useToast from '../hooks/useToast'
import { getDocumentName } from '../services/documentService'

const DEFAULT_SUGGESTIONS = [
  'Summarize the key points of my uploaded document.',
  'What are the main findings mentioned?',
  'List any dates or figures referenced in the document.',
]

function sourceLine(source) {
  const name = getDocumentName(source.document_id) ?? 'Uploaded document'
  const page = source.page_number != null ? `, page ${source.page_number}` : ''
  return `[${source.number}] ${name}${page}`
}

function conversationToMarkdown(messages) {
  const header = `# AgroSense-RAG Conversation\n\nExported ${new Date().toLocaleString()}\n\n`
  const turns = messages.map((message) => {
    const roleLabel = message.role === 'user' ? 'You' : 'AgroSense-RAG'
    let block = `### ${roleLabel}\n\n${message.content}\n`
    if (message.role === 'assistant' && message.sources?.length > 0) {
      block += `\n**Sources:**\n${message.sources.map((source) => `- ${sourceLine(source)}`).join('\n')}\n`
    }
    return block
  })
  return header + turns.join('\n---\n\n')
}

export default function Chat() {
  const location = useLocation()
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search])

  const cropParam = searchParams.get('crop') || location.state?.crop
  const diseaseParam = searchParams.get('disease') || location.state?.disease
  const severityParam = searchParams.get('severity') || location.state?.severity

  const [dismissedContext, setDismissedContext] = useState(false)

  const hasDiagnosticContext = Boolean((cropParam || diseaseParam) && !dismissedContext)

  const formattedCrop = cropParam
    ? cropParam.charAt(0).toUpperCase() + cropParam.slice(1)
    : 'Plant'
  const formattedDisease = diseaseParam
    ? diseaseParam
        .split(' ')
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ')
    : 'Diagnosed Condition'
  const formattedSeverity = severityParam
    ? severityParam.charAt(0).toUpperCase() + severityParam.slice(1)
    : 'Moderate'

  const diagnosticSuggestions = useMemo(() => {
    if (!hasDiagnosticContext) return DEFAULT_SUGGESTIONS
    return [
      `What is the organic spray rotation schedule for this ${formattedDisease}?`,
      `What are the chemical rate dosages and REI intervals for ${formattedDisease} on ${formattedCrop}?`,
      `How do I prevent ${formattedDisease} from recurring next season?`,
    ]
  }, [hasDiagnosticContext, formattedDisease, formattedCrop])

  const {
    messages,
    isSending,
    isLoadingSession,
    ask,
    regenerate,
    clearSession,
    isGraphMode = false,
    setIsGraphMode = () => {},
    graphNodeStates = {},
    activeGraphNode = null,
  } = useChat(location.state?.sessionId, location.state?.documentIds)
  const [persona, setPersona] = useState('')
  const [language, setLanguage] = useState(
    searchParams.get('lang') || location.state?.language || 'en'
  )
  const bottomRef = useRef(null)
  const containerRef = useRef(null)
  const isAtBottomRef = useRef(true)

  const handleScroll = () => {
    const el = containerRef.current
    if (!el) return
    const { scrollHeight, scrollTop, clientHeight } = el
    isAtBottomRef.current = scrollHeight - scrollTop - clientHeight < 80
  }

  useEffect(() => {
    if (isAtBottomRef.current && bottomRef.current) {
      bottomRef.current.scrollIntoView({ block: 'end' })
    }
  }, [messages])

  const [exportMenuOpen, setExportMenuOpen] = useState(false)
  const exportMenuRef = useRef(null)

  useEffect(() => {
    if (!exportMenuOpen) return undefined
    const handleClickOutside = (event) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(event.target)) {
        setExportMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [exportMenuOpen])

  const handleExportMarkdown = () => {
    const blob = new Blob([conversationToMarkdown(messages)], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `agrosense-conversation-${Date.now()}.md`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
    setExportMenuOpen(false)
  }

  const handlePrint = () => {
    setExportMenuOpen(false)
    window.print()
  }

  const { file, progress, status: uploadStatus, selectFile, startUpload, reset: resetUpload } = useUpload()
  const { showToast } = useToast()
  const [isDragging, setIsDragging] = useState(false)
  const dragCounterRef = useRef(0)

  useEffect(() => {
    if (file && uploadStatus === 'idle') {
      startUpload()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file])

  useEffect(() => {
    if (uploadStatus === 'success' || uploadStatus === 'error') {
      const timeout = setTimeout(resetUpload, 400)
      return () => clearTimeout(timeout)
    }
    return undefined
  }, [uploadStatus, resetUpload])

  const handleDragEnter = (event) => {
    event.preventDefault()
    if (event.dataTransfer.types.includes('Files')) {
      dragCounterRef.current += 1
      setIsDragging(true)
    }
  }

  const handleDragOver = (event) => {
    event.preventDefault()
  }

  const handleDragLeave = (event) => {
    event.preventDefault()
    dragCounterRef.current = Math.max(0, dragCounterRef.current - 1)
    if (dragCounterRef.current === 0) setIsDragging(false)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    dragCounterRef.current = 0
    setIsDragging(false)
    if (uploadStatus === 'uploading') return
    const dropped = event.dataTransfer.files?.[0]
    if (!dropped) return
    if (dropped.type !== 'application/pdf') {
      showToast('Please drop a PDF file.', 'error')
      return
    }
    selectFile(dropped)
  }

  const lastAssistantId = [...messages].reverse().find((m) => m.role === 'assistant')?.id

  const precedingUserQuery = useMemo(() => {
    const map = new Map()
    let lastUserContent = ''
    for (const message of messages) {
      map.set(message.id, lastUserContent)
      if (message.role === 'user') lastUserContent = message.content
    }
    return map
  }, [messages])

  return (
    <div
      className="relative mx-auto flex h-[calc(100vh-7.5rem)] max-w-3xl flex-col"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 rounded-panel border-2 border-dashed border-accent-500/60 bg-surface-light/90 backdrop-blur-sm dark:bg-surface-dark/90"
          >
            <span className="flex h-14 w-14 items-center justify-center rounded-xl border border-accent-500/40 bg-accent-500/10 text-accent-600 dark:text-accent-500">
              <UploadCloud size={26} strokeWidth={1.5} />
            </span>
            <p className="font-display text-base font-semibold text-slate-800 dark:text-ink-primary">
              Drop your PDF to add it to your documents
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {uploadStatus === 'uploading' && (
        <div className="absolute inset-x-0 top-2 z-10 mx-auto w-full max-w-sm rounded-panel border border-border-light bg-card-light px-4 py-3 shadow-soft-light dark:border-border dark:bg-card dark:shadow-soft">
          <p className="mb-1.5 truncate text-xs font-medium text-slate-600 dark:text-ink-secondary">
            Uploading {file?.name}…
          </p>
          <ProgressBar value={progress} />
        </div>
      )}

      {/* Top Header Controls / Multi-Agent Execution Graph Toggle */}
      <div className="mb-2 flex items-center justify-between px-1 print:hidden">
        <div className="flex items-center gap-2">
          <button
            type="button"
            data-testid="toggle-agent-graph"
            onClick={() => setIsGraphMode((prev) => !prev)}
            className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
              isGraphMode
                ? 'border border-accent-500/40 bg-accent-500/10 text-accent-700 dark:border-accent-500/30 dark:bg-accent-500/15 dark:text-accent-300 shadow-sm'
                : 'border border-border-light bg-card-light text-slate-500 hover:bg-slate-100 dark:border-border dark:bg-card dark:text-ink-muted dark:hover:bg-white/[0.05]'
            }`}
            title="Toggle multi-agent execution StateGraph stream and visualization"
          >
            <GitFork size={13} className={isGraphMode ? 'text-accent-500' : 'text-slate-400 dark:text-ink-muted'} />
            <span>Multi-Agent Execution Graph: {isGraphMode ? 'ON' : 'OFF'}</span>
          </button>
        </div>
      </div>

      {/* Multi-Agent Graph Visualizer (when enabled) */}
      <AnimatePresence>
        {isGraphMode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="mb-3 print:hidden"
          >
            <AgentGraphVisualizer
              nodeStates={graphNodeStates}
              activeNode={activeGraphNode}
              isExecuting={isSending}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Diagnostic Context Active Banner */}
      {hasDiagnosticContext && (
        <div
          data-testid="diagnostic-context-banner"
          className="mb-2 rounded-xl border border-accent-500/30 bg-accent-500/10 p-3 text-xs text-accent-950 dark:border-accent-500/30 dark:bg-accent-500/15 dark:text-accent-200 print:hidden"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-2.5">
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-500/20 text-accent-700 dark:text-accent-300">
                <Sprout size={13} />
              </span>
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2 font-semibold text-slate-900 dark:text-ink-primary">
                  <span>
                    Diagnosed Leaf Context Active: {formattedCrop} - {formattedDisease} ({formattedSeverity})
                  </span>
                  <span className="rounded bg-accent-500/20 px-1.5 py-0.2 text-[10px] font-bold text-accent-800 dark:text-accent-300">
                    Live Memory Bridge
                  </span>
                </div>
                <p className="text-[11px] text-slate-600 dark:text-ink-secondary">
                  Your questions are grounded in this specific leaf diagnosis and agronomic pathology data.
                </p>

                {/* Quick Action Suggestion Pill */}
                <div className="pt-1">
                  <button
                    type="button"
                    onClick={() =>
                      ask(`What is the organic spray rotation schedule for this ${formattedDisease}?`, persona)
                    }
                    className="inline-flex items-center gap-1.5 rounded-full border border-accent-500/40 bg-white/80 px-3 py-1 text-[11px] font-medium text-accent-800 transition-colors hover:bg-white dark:border-accent-500/30 dark:bg-card dark:text-accent-300 dark:hover:bg-white/10"
                  >
                    <Sparkles size={11} className="text-accent-500" />
                    <span>What is the organic spray rotation schedule for this {formattedDisease}?</span>
                  </button>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setDismissedContext(true)}
              className="rounded p-1 text-slate-400 hover:bg-slate-200/50 hover:text-slate-600 dark:hover:bg-white/10 dark:hover:text-ink-primary"
              title="Dismiss diagnostic context"
              aria-label="Dismiss diagnostic context banner"
            >
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      <div ref={containerRef} onScroll={handleScroll} className="flex-1 space-y-5 overflow-y-auto px-1 py-4 sm:px-3 print:hidden">
        {isLoadingSession ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-slate-400 dark:text-ink-muted">
            <ProgressBar indeterminate className="h-1.5 w-40" />
            <p className="text-xs">Loading conversation…</p>
          </div>
        ) : messages.length === 0 ? (
          <EmptyState
            icon={hasDiagnosticContext ? Sprout : MessageCircle}
            title={
              hasDiagnosticContext
                ? `Ask about your ${formattedCrop} ${formattedDisease}`
                : 'Ask anything about your documents'
            }
            description={
              hasDiagnosticContext
                ? `AgroSense-RAG has loaded the active diagnostic context for ${formattedCrop} (${formattedDisease}). Ask about organic treatments, chemical rates, or prevention schedules.`
                : "Upload a PDF, then ask questions here. Answers are grounded in the content you've uploaded."
            }
            action={
              <div className="flex flex-wrap justify-center gap-2">
                {diagnosticSuggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => ask(suggestion, persona)}
                    className="rounded-full border border-border-light bg-white/60 px-3.5 py-1.5 text-xs text-slate-600 transition-colors hover:bg-white dark:border-border dark:bg-white/[0.03] dark:text-ink-secondary dark:hover:bg-white/[0.06]"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            }
          />
        ) : (
          <AnimatePresence initial={false}>
            {messages.map((message) => (
              <ChatBubble
                key={message.id}
                message={message}
                isLast={message.id === lastAssistantId}
                onRegenerate={() => regenerate(persona, language)}
                isRegenerating={isSending}
                onFollowUpClick={(q) => ask(q, persona, language)}
                query={precedingUserQuery.get(message.id) ?? ''}
              />
            ))}
          </AnimatePresence>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="hidden print:block px-1 py-4">
        <h1 className="mb-4 font-display text-xl font-bold !text-slate-900">AgroSense-RAG Conversation</h1>
        {messages.map((message) => (
          <div key={message.id} className="mb-4 break-inside-avoid">
            <p className="text-sm font-semibold text-slate-900">{message.role === 'user' ? 'You' : 'AgroSense-RAG'}</p>
            <p className="whitespace-pre-wrap text-sm text-slate-700">{message.content}</p>
            {message.role === 'assistant' && message.sources?.length > 0 && (
              <ul className="mt-1 list-disc pl-5 text-xs text-slate-500">
                {message.sources.map((source) => (
                  <li key={source.chunk_id}>{sourceLine(source)}</li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>

      <div className="sticky bottom-0 pt-2 print:hidden">
        <div className="flex items-center justify-between mb-2 px-1 sm:px-3">
          <ChatInput
            onSend={ask}
            disabled={isSending}
            persona={persona}
            onPersonaChange={setPersona}
            language={language}
            onLanguageChange={setLanguage}
          />
          <div className="flex items-center gap-1">
            <div className="relative" ref={exportMenuRef}>
              <button
                type="button"
                onClick={() => setExportMenuOpen((open) => !open)}
                disabled={messages.length === 0}
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-slate-500 transition-colors hover:bg-slate-100 dark:text-ink-muted dark:hover:bg-white/[0.05] disabled:opacity-50 disabled:cursor-not-allowed"
                title="Export this conversation"
              >
                <Download size={14} strokeWidth={2} />
                <span className="hidden sm:inline">Export</span>
              </button>
              <AnimatePresence>
                {exportMenuOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 4, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 4, scale: 0.98 }}
                    transition={{ duration: 0.12 }}
                    className="panel absolute bottom-full right-0 z-20 mb-2 w-52 overflow-hidden rounded-xl p-1"
                  >
                    <button
                      type="button"
                      onClick={handleExportMarkdown}
                      className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs text-slate-600 transition-colors hover:bg-slate-100 dark:text-ink-secondary dark:hover:bg-white/[0.05]"
                    >
                      <FileDown size={14} strokeWidth={1.75} />
                      Download as Markdown
                    </button>
                    <button
                      type="button"
                      onClick={handlePrint}
                      className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs text-slate-600 transition-colors hover:bg-slate-100 dark:text-ink-secondary dark:hover:bg-white/[0.05]"
                    >
                      <Printer size={14} strokeWidth={1.75} />
                      Print / Save as PDF
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <button
              type="button"
              onClick={clearSession}
              disabled={isSending || messages.length === 0}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-slate-500 transition-colors hover:bg-slate-100 dark:text-ink-muted dark:hover:bg-white/[0.05] disabled:opacity-50 disabled:cursor-not-allowed"
              title="Start a new conversation"
            >
              <Plus size={14} strokeWidth={2} />
              <span className="hidden sm:inline">New chat</span>
            </button>
          </div>
        </div>
        <p className="mt-2 text-center text-[11px] text-slate-400 dark:text-ink-muted">
          AgroSense-RAG can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  )
}
