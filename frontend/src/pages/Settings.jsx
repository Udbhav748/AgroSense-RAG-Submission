import { useEffect, useState } from 'react'
import { CheckCircle2, Info, Moon, Server, Sun, ThumbsDown, ThumbsUp, Trash2, XCircle } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Modal from '../components/ui/Modal'
import Skeleton from '../components/ui/Skeleton'
import useTheme from '../hooks/useTheme'
import useToast from '../hooks/useToast'
import { getHealthStatus } from '../services/healthService'
import { getUploadHistory, removeFromUploadHistory } from '../services/documentService'
import { getFeedback } from '../services/adminService'
import getErrorMessage from '../utils/errorMessage'

function StatusRow({ label, value, ok, okLabel = 'Ready', badLabel = 'Missing' }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-slate-500 dark:text-ink-muted">{label}</span>
      {ok !== undefined ? (
        <span
          className={`inline-flex items-center gap-1.5 text-xs font-medium ${
            ok ? 'text-success' : 'text-danger'
          }`}
        >
          {ok ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
          {ok ? okLabel : badLabel}
        </span>
      ) : (
        <span className="font-medium text-slate-700 dark:text-ink-secondary">{value}</span>
      )}
    </div>
  )
}

function FeatureChip({ label, enabled }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${
        enabled
          ? 'bg-success/10 text-success'
          : 'bg-slate-500/10 text-slate-500 dark:text-ink-muted'
      }`}
    >
      {label}
    </span>
  )
}

export default function Settings() {
  const { theme, setTheme } = useTheme()
  const { showToast } = useToast()
  const [confirmClear, setConfirmClear] = useState(false)
  const [health, setHealth] = useState(null)
  const [healthLoading, setHealthLoading] = useState(true)
  const [healthError, setHealthError] = useState(null)

  useEffect(() => {
    let active = true
    getHealthStatus()
      .then((data) => {
        if (active) setHealth(data)
      })
      .catch((error) => {
        if (active) setHealthError(getErrorMessage(error, 'Backend is unreachable.'))
      })
      .finally(() => {
        if (active) setHealthLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  const clearHistory = () => {
    getUploadHistory().forEach((doc) => removeFromUploadHistory(doc.document_id))
    setConfirmClear(false)
    showToast('Document history cleared.', 'success')
  }

  const hasFallback = health?.llm?.fallback_provider

  const [feedback, setFeedback] = useState([])
  const [feedbackLoading, setFeedbackLoading] = useState(false)
  const [feedbackError, setFeedbackError] = useState(null)

  useEffect(() => {
    let active = true
    setFeedbackLoading(true)
    setFeedbackError(null)
    getFeedback(20)
      .then((data) => {
        if (active) setFeedback(data.events ?? [])
      })
      .catch((error) => {
        if (active) setFeedbackError(getErrorMessage(error, 'Could not load feedback.'))
      })
      .finally(() => {
        if (active) setFeedbackLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  return (
    <div className="mx-auto max-w-2xl space-y-5 py-4">
      <div>
        <h2 className="font-display text-xl font-bold">Settings</h2>
        <p className="text-sm text-slate-500 dark:text-ink-muted">Manage your preferences.</p>
      </div>

      <Card padding="lg">
        <h3 className="mb-4 text-sm font-semibold text-slate-700 dark:text-ink-secondary">Appearance</h3>
        <div className="grid grid-cols-2 gap-3">
          {[
            { value: 'dark', label: 'Dark', icon: Moon },
            { value: 'light', label: 'Light', icon: Sun },
          ].map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              type="button"
              onClick={() => setTheme(value)}
              className={`flex flex-col items-center gap-2 rounded-xl border px-4 py-5 text-sm font-medium transition-all duration-200 ${
                theme === value
                  ? 'border-accent-500/50 bg-accent-500/10 text-accent-600 dark:text-accent-500'
                  : 'border-border-light text-slate-500 hover:bg-slate-900/5 dark:border-border dark:text-ink-muted dark:hover:bg-white/[0.04]'
              }`}
            >
              <Icon size={19} strokeWidth={1.5} />
              {label}
            </button>
          ))}
        </div>
      </Card>

      <Card padding="lg">
        <h3 className="mb-1 text-sm font-semibold text-slate-700 dark:text-ink-secondary">Data</h3>
        <p className="mb-4 text-xs text-slate-500 dark:text-ink-muted">
          Document history is stored locally in this browser only.
        </p>
        <Button variant="secondary" icon={Trash2} onClick={() => setConfirmClear(true)}>
          Clear document history
        </Button>
      </Card>

      <Card padding="lg">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-ink-secondary">
          <Server size={15} strokeWidth={1.5} />
          System status
        </h3>
        {healthLoading ? (
          <div className="space-y-4">
            <div className="space-y-2">
              <Skeleton className="h-3 w-10" />
              {Array.from({ length: 4 }).map((_, index) => (
                <Skeleton key={index} className="h-4 w-full" />
              ))}
            </div>
            <div className="flex flex-wrap gap-2">
              {Array.from({ length: 4 }).map((_, index) => (
                <Skeleton key={index} className="h-6 w-24 rounded-full" />
              ))}
            </div>
          </div>
        ) : healthError ? (
          <p className="text-sm text-danger">{healthError}</p>
        ) : (
          <div className="space-y-5">
            <div>
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400 dark:text-ink-muted">
                Inference & Language Engine
              </p>
              <div className="space-y-2 text-sm">
                <StatusRow label="LLM Provider" value={health.llm.provider} />
                <StatusRow label="Provider API key" ok={health.llm.provider_configured} />
                <StatusRow
                  label="Fallback provider"
                  value={hasFallback ? health.llm.fallback_provider : 'None'}
                  ok={hasFallback ? health.llm.fallback_configured : undefined}
                  okLabel="Ready"
                  badLabel="Missing"
                />
                <StatusRow
                  label="Model routing"
                  ok={health.llm.model_routing_enabled}
                  okLabel="On"
                  badLabel="Off"
                />
              </div>
            </div>

            <div className="border-t border-border-light pt-4 dark:border-border">
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400 dark:text-ink-muted">
                Vision & Vector Infrastructure
              </p>
              <div className="space-y-2 text-sm">
                <StatusRow
                  label="LeafSense Vision (Port 8001)"
                  ok={health.vision_service?.online}
                  okLabel="Online"
                  badLabel="Offline"
                />
                <StatusRow
                  label="Plant pathology model"
                  value="Hybrid CBAM + ViT + EfficientNet (38 Classes)"
                />
                <StatusRow
                  label="Vector store"
                  value={
                    health.vector_store?.backend === 'pgvector'
                      ? 'PostgreSQL PgVector'
                      : 'Local FAISS (749 Vectors / 12 Crops)'
                  }
                />
                <StatusRow
                  label="Relational database"
                  value={health.database === 'connected' ? 'SQLite (db.sqlite3)' : 'Stateless'}
                />
              </div>
            </div>

            <div className="border-t border-border-light pt-4 dark:border-border">
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400 dark:text-ink-muted">
                Active Capabilities
              </p>
              <div className="flex flex-wrap gap-2">
                <FeatureChip
                  label="Leaf Vision Network"
                  enabled={Boolean(health.vision_service?.online)}
                />
                <FeatureChip label="Agronomic RAG Engine" enabled={true} />
                <FeatureChip label="Hybrid Search (Dense + BM25 RRF)" enabled={true} />
                <FeatureChip label="OCR Engine" enabled={health.multimodal?.ocr_available} />
              </div>
            </div>
          </div>
        )}
      </Card>

      <Card padding="lg">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-ink-secondary">
          <ThumbsUp size={15} strokeWidth={1.5} />
          Recent feedback
        </h3>
        {feedbackLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, index) => (
              <div key={index} className="flex items-start gap-3">
                <Skeleton className="h-6 w-6 shrink-0 rounded-full" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-3.5 w-4/5" />
                  <Skeleton className="h-3 w-1/3" />
                </div>
              </div>
            ))}
          </div>
        ) : feedbackError ? (
          <p className="text-sm text-danger">{feedbackError}</p>
        ) : feedback.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-ink-muted">
            No feedback yet — use the thumbs control on a chat reply to record one.
          </p>
        ) : (
          <ul className="space-y-3">
            {feedback.map((event) => {
              const positive = event.rating === 'up'
              const Icon = positive ? ThumbsUp : ThumbsDown
              return (
                <li key={event.timestamp + event.message_id} className="flex items-start gap-3 text-sm">
                  <span
                    className={`mt-0.5 inline-flex shrink-0 items-center rounded-full p-1.5 ${
                      positive ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'
                    }`}
                  >
                    <Icon size={12} />
                  </span>
                  <div className="min-w-0 flex-1">
                    {event.comment ? (
                      <p className="text-slate-700 dark:text-ink-secondary">{event.comment}</p>
                    ) : (
                      <p className="text-slate-400 dark:text-ink-muted">
                        {positive ? 'Liked this reply.' : 'Disliked this reply.'}
                      </p>
                    )}
                    <p className="mt-0.5 text-xs text-slate-400 dark:text-ink-muted">
                      {new Date(event.timestamp).toLocaleString()} · {event.reviewer_id ?? 'anonymous'}
                    </p>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </Card>

      <Card padding="lg" className="flex items-start gap-3">
        <Info size={18} className="mt-0.5 shrink-0 text-slate-400 dark:text-ink-muted" />
        <div className="text-sm text-slate-500 dark:text-ink-muted">
          <p className="font-medium text-slate-700 dark:text-ink-secondary">AgroSense-RAG</p>
          <p>AI-powered document intelligence, built with FastAPI, FAISS, and Gemini.</p>
        </div>
      </Card>

      <Modal
        open={confirmClear}
        onClose={() => setConfirmClear(false)}
        title="Clear document history"
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmClear(false)}>
              Cancel
            </Button>
            <Button variant="danger" onClick={clearHistory}>
              Clear history
            </Button>
          </>
        }
      >
        This removes all documents from your local history. It doesn't delete anything on the server.
      </Modal>
    </div>
  )
}
