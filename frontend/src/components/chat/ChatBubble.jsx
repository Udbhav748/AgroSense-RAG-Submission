import { useState } from 'react'
import { motion } from 'framer-motion'
import { Check, ClipboardList, Copy, FileText, Globe, Layers, Loader2, RotateCcw, Sparkles, ThumbsDown, ThumbsUp, User } from 'lucide-react'
import AgentTraceStrip from './AgentTraceStrip'
import CitedAnswer from './CitedAnswer'
import RubricReviewModal from './RubricReviewModal'
import Button from '../ui/Button'
import { sendFeedback } from '../../services/feedbackService'
import useToast from '../../hooks/useToast'
import getErrorMessage from '../../utils/errorMessage'

const ANSWER_SOURCE_META = {
  documents: { label: 'Documents', icon: FileText },
  web: { label: 'Web', icon: Globe },
  mixed: { label: 'Documents + web', icon: Layers },
}

export default function ChatBubble({ message, isLast, onRegenerate, isRegenerating, onFollowUpClick, query }) {
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState(null) // null | 'up' | 'down'
  const [rubricModalOpen, setRubricModalOpen] = useState(false)
  const [rubricSubmitted, setRubricSubmitted] = useState(false)
  const { showToast } = useToast()
  const isUser = message.role === 'user'
  const answerSourceMeta = ANSWER_SOURCE_META[message.answerSource ?? 'documents'] ?? ANSWER_SOURCE_META.documents

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard API unavailable — silently ignore, copy is a nicety.
    }
  }

  const handleFeedback = async (rating) => {
    if (feedback) return // already rated — send once per message
    setFeedback(rating) // optimistic + disables both buttons immediately
    try {
      await sendFeedback(message.id, rating)
    } catch (error) {
      setFeedback(null) // let the user retry
      showToast(getErrorMessage(error, 'Could not record feedback.'), 'error')
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
    >
      <span
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border ${
          isUser
            ? 'border-border-light bg-slate-200 text-slate-600 dark:border-border dark:bg-white/[0.06] dark:text-ink-secondary'
            : 'border-border-light bg-slate-100 text-slate-500 dark:border-border dark:bg-white/[0.04] dark:text-ink-secondary'
        }`}
      >
        {isUser ? <User size={15} strokeWidth={1.75} /> : <Sparkles size={15} strokeWidth={1.75} />}
      </span>

      <div className={`flex max-w-[85%] flex-col sm:max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={
            isUser
              ? 'rounded-2xl rounded-tr-md bg-slate-900/[0.04] px-4 py-3 text-sm text-slate-800 dark:bg-white/[0.06] dark:text-ink-primary'
              : message.isClarifyingQuestion
                ? 'rounded-2xl rounded-tl-md border border-accent-500/40 px-4 py-3.5 text-sm text-slate-700 dark:text-ink-secondary'
                : 'panel rounded-2xl rounded-tl-md px-4 py-3.5 text-sm text-slate-700 dark:text-ink-secondary'
          }
        >
          {!isUser && <AgentTraceStrip trace={message.trace} isStreaming={message.isStreaming} />}
          {!isUser && !message.isStreaming ? (
            <CitedAnswer text={message.content} sources={message.sources} query={query} />
          ) : (
            <p className="whitespace-pre-wrap leading-relaxed">
              {message.content}
              {message.isStreaming && (
                <span className="ml-0.5 inline-block h-3.5 w-[2px] animate-pulse bg-slate-400 align-middle dark:bg-ink-muted" />
              )}
            </p>
          )}
          {!isUser && !message.isStreaming && message.retrievalConfidence && message.retrievalConfidence !== 'good' && (
            <p className="mt-1.5 text-xs text-amber-600 dark:text-amber-500">
              Low confidence — double-check this against the source.
            </p>
          )}
        </div>

      {!isUser && !message.isStreaming && (
        <>
          {/* answer_source badge — only worth showing when retrieval pulled
              from web / mixed, since "documents" is the default. */}
          {!message.isError && message.answerSource && message.answerSource !== 'documents' && (
            <span className="mt-1.5 inline-flex items-center gap-1.5 rounded-full border border-border-light bg-slate-900/5 px-2 py-0.5 text-[10px] font-medium text-slate-500 dark:border-border dark:bg-white/[0.03] dark:text-ink-muted">
              <answerSourceMeta.icon size={10} strokeWidth={2} />
              {answerSourceMeta.label}
            </span>
          )}
          {message.hallucinationDetected && (
            <p className="mt-1.5 text-xs text-amber-600 dark:text-amber-500">
              This answer may be under-grounded — double-check it against the cited sources.
            </p>
          )}
          {!message.hallucinationDetected && message.groundingScore != null && message.groundingScore < 0.5 && (
            <p className="mt-1.5 text-xs text-amber-600 dark:text-amber-500">
              Low groundedness ({Math.round(message.groundingScore * 100)}%) — double-check this against the cited sources.
            </p>
          )}
        </>
      )}

      {!isUser && !message.isStreaming && message.followUpQuestions?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {message.followUpQuestions.map((q) => (
              <Button key={q} variant="ghost" size="sm" onClick={() => onFollowUpClick?.(q)}>
                {q}
              </Button>
            ))}
          </div>
        )}

        {!isUser && !message.isStreaming && (
          <div className="mt-1.5 flex items-center gap-1 px-1">
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center gap-1 rounded-lg px-1.5 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-900/5 hover:text-slate-600 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-secondary"
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
            {isLast && (
              <button
                type="button"
                onClick={onRegenerate}
                disabled={isRegenerating}
                className="flex items-center gap-1 rounded-lg px-1.5 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-900/5 hover:text-slate-600 disabled:pointer-events-none disabled:opacity-50 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-secondary"
              >
                {isRegenerating ? (
                  <Loader2 size={12} strokeWidth={1.75} className="animate-spin" />
                ) : (
                  <RotateCcw size={12} strokeWidth={1.75} />
                )}
                Regenerate
              </button>
            )}
            {!message.isError && (
              <>
                <button
                  type="button"
                  onClick={() => handleFeedback('up')}
                  disabled={feedback != null}
                  aria-pressed={feedback === 'up'}
                  aria-label="Good response"
                  className={`flex items-center rounded-lg p-1.5 transition-colors disabled:pointer-events-none ${
                    feedback === 'up'
                      ? 'text-success'
                      : 'text-slate-400 hover:bg-slate-900/5 hover:text-slate-600 disabled:opacity-30 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-secondary'
                  }`}
                >
                  <ThumbsUp size={12} strokeWidth={1.75} />
                </button>
                <button
                  type="button"
                  onClick={() => handleFeedback('down')}
                  disabled={feedback != null}
                  aria-pressed={feedback === 'down'}
                  aria-label="Bad response"
                  className={`flex items-center rounded-lg p-1.5 transition-colors disabled:pointer-events-none ${
                    feedback === 'down'
                      ? 'text-danger'
                      : 'text-slate-400 hover:bg-slate-900/5 hover:text-slate-600 disabled:opacity-30 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-secondary'
                  }`}
                >
                  <ThumbsDown size={12} strokeWidth={1.75} />
                </button>
                {feedback && !rubricSubmitted && (
                  <button
                    type="button"
                    onClick={() => setRubricModalOpen(true)}
                    className="flex items-center gap-1 rounded-lg px-1.5 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-900/5 hover:text-slate-600 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-secondary"
                  >
                    <ClipboardList size={12} strokeWidth={1.75} />
                    Detailed review
                  </button>
                )}
                {rubricSubmitted && (
                  <span className="flex items-center gap-1 px-1.5 py-1 text-xs text-success">
                    <Check size={12} strokeWidth={1.75} />
                    Reviewed
                  </span>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {feedback && (
        <RubricReviewModal
          open={rubricModalOpen}
          onClose={() => setRubricModalOpen(false)}
          messageId={message.id}
          rating={feedback}
          onSubmitted={() => setRubricSubmitted(true)}
        />
      )}
    </motion.div>
  )
}
