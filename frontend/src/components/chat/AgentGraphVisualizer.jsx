import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  AlertCircle,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronRight,
  Code,
  FileSearch,
  GitFork,
  Layers,
  Loader2,
  MessageSquare,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'

// Workflow node definitions
export const GRAPH_NODES = [
  {
    id: 'planner',
    label: 'Planner',
    displayName: 'Planner',
    icon: GitFork,
    description: 'Intent routing & action planning',
  },
  {
    id: 'document_analyst',
    label: 'Document Analyst / Research',
    displayName: 'Document Analyst / Research',
    icon: FileSearch,
    description: 'Dense & BM25 retrieval across chunks',
  },
  {
    id: 'synthesizer',
    label: 'Synthesizer',
    displayName: 'Synthesizer',
    icon: Sparkles,
    description: 'Context synthesis & citation drafting',
  },
  {
    id: 'fact_checker',
    label: 'Fact Checker',
    displayName: 'Fact Checker',
    icon: ShieldCheck,
    description: 'Citation verification & grounding audit',
  },
]

function getStatusBadge(status) {
  switch (status) {
    case 'running':
      return (
        <span
          data-testid="status-badge-running"
          className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-600 dark:border-amber-500/40 dark:text-amber-400"
        >
          <Loader2 size={10} className="animate-spin" />
          Running
        </span>
      )
    case 'completed':
      return (
        <span
          data-testid="status-badge-completed"
          className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-600 dark:border-emerald-500/40 dark:text-emerald-400"
        >
          <Check size={10} strokeWidth={2.5} />
          Completed
        </span>
      )
    case 'failed':
      return (
        <span
          data-testid="status-badge-failed"
          className="inline-flex items-center gap-1 rounded-full border border-red-500/30 bg-red-500/10 px-2 py-0.5 text-[10px] font-semibold text-red-600 dark:border-red-500/40 dark:text-red-400"
        >
          <AlertCircle size={10} />
          Failed
        </span>
      )
    default:
      return (
        <span
          data-testid="status-badge-idle"
          className="inline-flex items-center gap-1 rounded-full border border-slate-300 bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500 dark:border-border dark:bg-white/[0.04] dark:text-ink-muted"
        >
          Idle
        </span>
      )
  }
}

export function StateDrawer({ node, stateData, onClose }) {
  if (!node) return null

  const output = stateData?.output ?? {}
  const durationMs = stateData?.duration_ms ?? null
  const status = stateData?.status ?? 'idle'

  const isFactChecker = node.id === 'fact_checker'
  const isPlanner = node.id === 'planner'
  const isDocAnalyst = node.id === 'document_analyst'
  const isSynthesizer = node.id === 'synthesizer'

  const factCheckScore =
    output?.score != null
      ? typeof output.score === 'number'
        ? output.score
        : parseFloat(output.score)
      : null

  const formattedScore =
    factCheckScore != null ? `${Math.round(factCheckScore * 100)}%` : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 8, scale: 0.98 }}
      transition={{ duration: 0.15 }}
      data-testid="state-drawer"
      className="mt-3 rounded-xl border border-border-light bg-card-light p-4 shadow-soft-light dark:border-border dark:bg-card dark:shadow-soft"
    >
      <div className="flex items-center justify-between border-b border-border-light pb-3 dark:border-border">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent-500/30 bg-accent-500/10 text-accent-600 dark:text-accent-400">
            <node.icon size={16} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-display text-sm font-semibold text-slate-900 dark:text-ink-primary">
                {node.displayName} State
              </h3>
              {getStatusBadge(status)}
            </div>
            <p className="text-[11px] text-slate-500 dark:text-ink-muted">
              {node.description}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {durationMs != null && (
            <span
              data-testid="state-drawer-duration"
              className="rounded bg-slate-100 px-2 py-0.5 font-mono text-[11px] text-slate-600 dark:bg-white/[0.06] dark:text-ink-secondary"
            >
              {durationMs}ms
            </span>
          )}
          <button
            type="button"
            onClick={onClose}
            data-testid="state-drawer-close"
            className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-white/10 dark:hover:text-ink-primary"
            aria-label="Close state drawer"
          >
            <X size={15} />
          </button>
        </div>
      </div>

      {/* Structured Details Cards */}
      <div className="mt-3 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        {/* Fact Checker Specific Metrics */}
        {isFactChecker && (
          <>
            <div className="rounded-lg border border-border-light bg-white/70 p-2.5 dark:border-border dark:bg-white/[0.02]">
              <span className="text-[11px] font-medium text-slate-500 dark:text-ink-muted">
                Grounding Verification
              </span>
              <div className="mt-1 flex items-center justify-between">
                <span
                  data-testid="fact-check-grounded"
                  className={`text-xs font-semibold ${
                    output?.grounded ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'
                  }`}
                >
                  {output?.grounded ? 'Verified Grounded' : 'Unverified / Flagged'}
                </span>
                <ShieldCheck size={14} className="text-accent-500" />
              </div>
            </div>

            <div className="rounded-lg border border-border-light bg-white/70 p-2.5 dark:border-border dark:bg-white/[0.02]">
              <span className="text-[11px] font-medium text-slate-500 dark:text-ink-muted">
                Confidence / Grounding Score
              </span>
              <div className="mt-1 flex items-center justify-between">
                <span
                  data-testid="fact-check-score"
                  className="font-mono text-xs font-bold text-slate-900 dark:text-ink-primary"
                >
                  {formattedScore ?? '100%'}
                </span>
                <span className="text-[10px] text-slate-400 dark:text-ink-muted">
                  Score: {output?.score != null ? output.score : 1.0}
                </span>
              </div>
            </div>
          </>
        )}

        {/* Planner Specific Metrics */}
        {isPlanner && (
          <>
            <div className="rounded-lg border border-border-light bg-white/70 p-2.5 dark:border-border dark:bg-white/[0.02]">
              <span className="text-[11px] font-medium text-slate-500 dark:text-ink-muted">
                Routed Action
              </span>
              <p
                data-testid="planner-action"
                className="mt-1 font-mono text-xs font-semibold uppercase text-accent-600 dark:text-accent-400"
              >
                {output?.action ?? 'retrieve'}
              </p>
            </div>
            {output?.reason && (
              <div className="rounded-lg border border-border-light bg-white/70 p-2.5 dark:border-border dark:bg-white/[0.02]">
                <span className="text-[11px] font-medium text-slate-500 dark:text-ink-muted">
                  Routing Rationale
                </span>
                <p className="mt-1 text-xs text-slate-700 dark:text-ink-secondary">
                  {output.reason}
                </p>
              </div>
            )}
          </>
        )}

        {/* Document Analyst Metrics */}
        {isDocAnalyst && (
          <div className="rounded-lg border border-border-light bg-white/70 p-2.5 dark:border-border dark:bg-white/[0.02]">
            <span className="text-[11px] font-medium text-slate-500 dark:text-ink-muted">
              Retrieved Context Chunks
            </span>
            <div className="mt-1 flex items-center gap-2">
              <span
                data-testid="chunks-retrieved-count"
                className="font-mono text-xs font-bold text-slate-900 dark:text-ink-primary"
              >
                {output?.chunks_retrieved != null ? `${output.chunks_retrieved} chunks` : '0 chunks'}
              </span>
            </div>
          </div>
        )}

        {/* Synthesizer Metrics */}
        {isSynthesizer && (
          <div className="rounded-lg border border-border-light bg-white/70 p-2.5 dark:border-border dark:bg-white/[0.02]">
            <span className="text-[11px] font-medium text-slate-500 dark:text-ink-muted">
              Synthesis Output
            </span>
            <p className="mt-1 text-xs text-slate-700 dark:text-ink-secondary">
              {output?.answer_length
                ? `${output.answer_length} characters generated`
                : 'Synthesizing response tokens'}
            </p>
          </div>
        )}
      </div>

      {/* Raw Payload Inspector */}
      <div className="mt-3">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="flex items-center gap-1 text-[11px] font-medium text-slate-500 dark:text-ink-muted">
            <Code size={12} />
            Intermediate Node Payload
          </span>
        </div>
        <pre
          data-testid="state-drawer-payload"
          className="max-h-48 overflow-auto rounded-lg border border-border-light bg-slate-900 p-3 font-mono text-[11px] leading-relaxed text-slate-200 dark:border-border"
        >
          {JSON.stringify(
            {
              node: node.id,
              status,
              duration_ms: durationMs,
              output: output,
            },
            null,
            2
          )}
        </pre>
      </div>
    </motion.div>
  )
}

export default function AgentGraphVisualizer({
  nodeStates = {},
  activeNode = null,
  isExecuting = false,
  className = '',
}) {
  const [selectedNodeId, setSelectedNodeId] = useState(null)

  const selectedNodeDef = GRAPH_NODES.find((n) => n.id === selectedNodeId)
  const selectedNodeState = selectedNodeId ? nodeStates[selectedNodeId] : null

  return (
    <div
      data-testid="agent-graph-visualizer"
      className={`rounded-panel border border-border-light bg-surface-light/95 p-3.5 shadow-soft-light backdrop-blur-sm dark:border-border dark:bg-surface-dark/95 dark:shadow-soft ${className}`}
    >
      {/* Header Bar */}
      <div className="mb-3 flex items-center justify-between border-b border-border-light pb-2.5 dark:border-border">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-md border border-accent-500/30 bg-accent-500/10 text-accent-600 dark:text-accent-400">
            <Layers size={13} />
          </span>
          <h2 className="font-display text-xs font-semibold tracking-wide text-slate-800 dark:text-ink-primary uppercase">
            Multi-Agent Execution Graph
          </h2>
        </div>

        <div className="flex items-center gap-2">
          {isExecuting ? (
            <span
              data-testid="pipeline-running-badge"
              className="inline-flex items-center gap-1.5 rounded-full border border-accent-500/30 bg-accent-500/10 px-2.5 py-0.5 text-[11px] font-medium text-accent-700 dark:border-accent-500/30 dark:bg-accent-500/15 dark:text-accent-300"
            >
              <Activity size={12} className="animate-pulse text-accent-500" />
              <span>Pipeline Active</span>
            </span>
          ) : (
            <span className="text-[11px] text-slate-400 dark:text-ink-muted">
              Click node to inspect state
            </span>
          )}
        </div>
      </div>

      {/* Nodes Workflow Pipeline Strip */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-4 sm:gap-2">
        {GRAPH_NODES.map((node, index) => {
          const stateData = nodeStates[node.id] || { status: 'idle' }
          const isNodeRunning = stateData.status === 'running' || activeNode === node.id
          const isCompleted = stateData.status === 'completed'
          const isFailed = stateData.status === 'failed'
          const isSelected = selectedNodeId === node.id
          const durationMs = stateData.duration_ms

          let containerStyle =
            'border-border-light bg-card-light text-slate-700 dark:border-border dark:bg-card dark:text-ink-primary hover:border-slate-300 dark:hover:border-slate-600'
          if (isNodeRunning) {
            containerStyle =
              'border-accent-500 bg-accent-500/10 text-accent-950 shadow-[0_0_12px_rgba(37,99,235,0.15)] animate-pulse dark:border-accent-500/80 dark:bg-accent-500/15 dark:text-accent-200'
          } else if (isCompleted) {
            containerStyle =
              'border-emerald-500/40 bg-emerald-500/5 text-slate-800 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-ink-primary'
          } else if (isFailed) {
            containerStyle =
              'border-red-500/40 bg-red-500/5 text-slate-800 dark:border-red-500/30 dark:bg-red-500/10 dark:text-ink-primary'
          }

          if (isSelected) {
            containerStyle += ' ring-2 ring-accent-500'
          }

          return (
            <div key={node.id} className="relative flex items-center">
              <button
                type="button"
                data-testid={`graph-node-${node.id}`}
                onClick={() => setSelectedNodeId(isSelected ? null : node.id)}
                className={`flex w-full flex-col justify-between rounded-xl border p-2.5 text-left transition-all ${containerStyle}`}
              >
                <div className="flex items-start justify-between gap-1">
                  <div className="flex items-center gap-1.5">
                    <node.icon
                      size={14}
                      className={
                        isNodeRunning
                          ? 'text-accent-500'
                          : isCompleted
                          ? 'text-emerald-500'
                          : 'text-slate-400 dark:text-ink-muted'
                      }
                    />
                    <span className="font-display text-xs font-semibold">
                      {node.displayName}
                    </span>
                  </div>

                  {isCompleted && (
                    <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-white">
                      <Check size={10} strokeWidth={3} />
                    </span>
                  )}
                  {isNodeRunning && (
                    <Loader2 size={12} className="shrink-0 animate-spin text-accent-500" />
                  )}
                </div>

                <div className="mt-2 flex items-center justify-between">
                  {getStatusBadge(stateData.status)}

                  {durationMs != null ? (
                    <span
                      data-testid={`duration-badge-${node.id}`}
                      className="font-mono text-[10px] font-medium text-slate-500 dark:text-ink-muted"
                    >
                      {durationMs}ms
                    </span>
                  ) : (
                    <span className="text-[10px] text-slate-400 dark:text-ink-muted">
                      {index + 1}/4
                    </span>
                  )}
                </div>
              </button>
            </div>
          )
        })}
      </div>

      {/* State Inspection Drawer */}
      <AnimatePresence>
        {selectedNodeDef && (
          <StateDrawer
            node={selectedNodeDef}
            stateData={selectedNodeState}
            onClose={() => setSelectedNodeId(null)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
