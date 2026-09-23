import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Check, ChevronDown, Loader2 } from 'lucide-react'

// stage -> human label, mirroring the stage vocabulary emitted by
// ChatService.stream_query (backend/app/services/rag_service.py):
// planning, retrieval, grading, web_search, generating, reflecting.
function traceLabel(stage, detail = {}) {
  switch (stage) {
    case 'planning':
      return 'Planning'
    case 'retrieval':
      return detail.chunk_count != null
        ? `Retrieved ${detail.chunk_count} chunk${detail.chunk_count === 1 ? '' : 's'}`
        : 'Retrieving document'
    case 'grading':
      return `Grading: ${detail.grade}`
    case 'web_search':
      return `Searched the web (${detail.result_count ?? 0} result${detail.result_count === 1 ? '' : 's'})`
    case 'local_research':
      return 'Researching your documents'
    case 'generating':
      return 'Generating'
    case 'reflecting':
      return 'Reflecting — regenerating'
    default:
      return stage
  }
}

// Live view while streaming: a left-to-right strip of steps, each with a
// checkmark once superseded by the next event, and a spinner on whichever
// step is currently in flight (the last one in the array).
function LiveTrace({ trace }) {
  return (
    <div className="mb-2 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs">
      {trace.map((event, index) => {
        const isActive = index === trace.length - 1
        return (
          <span key={index} className="flex items-center gap-1.5">
            {index > 0 && <span className="text-slate-300 dark:text-ink-muted">→</span>}
            <span
              className={`flex items-center gap-1 ${
                isActive ? 'text-slate-600 dark:text-ink-secondary' : 'text-slate-400 dark:text-ink-muted'
              }`}
            >
              {isActive ? (
                <Loader2 size={11} className="animate-spin" />
              ) : (
                <Check size={11} className="text-success" />
              )}
              {traceLabel(event.stage, event.detail)}
            </span>
          </span>
        )
      })}
    </div>
  )
}

// Once the answer is done, the strip collapses into a small expandable
// summary — useful for understanding what the agent did, but not worth
// permanent screen space on every past message.
function CollapsedTrace({ trace }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="mb-2">
      <button
        type="button"
        onClick={() => setExpanded((current) => !current)}
        className="flex items-center gap-1.5 rounded-lg px-1 py-0.5 text-xs text-slate-400 transition-colors hover:bg-slate-900/5 hover:text-slate-600 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-secondary"
      >
        <motion.span animate={{ rotate: expanded ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown size={12} />
        </motion.span>
        Agent steps ({trace.length})
      </button>
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <ul className="mt-1.5 space-y-1 px-1 py-1 text-xs text-slate-500 dark:text-ink-secondary">
              {trace.map((event, index) => (
                <li key={index} className="flex items-center gap-1.5">
                  <Check size={11} className="shrink-0 text-success" />
                  {traceLabel(event.stage, event.detail)}
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function AgentTraceStrip({ trace, isStreaming }) {
  if (!trace || trace.length === 0) return null
  return isStreaming ? <LiveTrace trace={trace} /> : <CollapsedTrace trace={trace} />
}
