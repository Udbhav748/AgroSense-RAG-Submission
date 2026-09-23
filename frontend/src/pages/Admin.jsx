import { useEffect, useState } from 'react'
import { Navigate } from 'react-router-dom'
import { BarChart3, ShieldCheck } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Skeleton from '../components/ui/Skeleton'
import useToast from '../hooks/useToast'
import useAuth from '../hooks/useAuth'
import { getUsageSummary, listApprovals, resolveApproval } from '../services/adminService'
import getErrorMessage from '../utils/errorMessage'

function approvalSummary(approval) {
  if (approval.action === 'web_search') return `Search the web for: "${approval.payload?.query ?? ''}"`
  if (approval.action === 'document_delete') return `Delete document ${approval.payload?.document_id ?? ''}`
  return approval.action
}

// Only rendered once App.jsx's route already confirms an authenticated
// session — this just adds the second gate (role), same redirect target
// ProtectedRoute uses for "not allowed here".
export default function Admin() {
  const { user } = useAuth()
  const { showToast } = useToast()
  const [usage, setUsage] = useState([])
  const [usageLoading, setUsageLoading] = useState(true)
  const [usageError, setUsageError] = useState(null)
  const [approvals, setApprovals] = useState([])
  const [approvalsLoading, setApprovalsLoading] = useState(true)
  const [approvalsError, setApprovalsError] = useState(null)
  const [resolvingId, setResolvingId] = useState(null)

  const isAdmin = user?.role === 'admin'

  useEffect(() => {
    if (!isAdmin) return undefined
    let active = true
    setUsageLoading(true)
    setUsageError(null)
    getUsageSummary()
      .then((data) => {
        if (active) setUsage(data.rows ?? [])
      })
      .catch((error) => {
        if (active) setUsageError(getErrorMessage(error, 'Could not load usage analytics.'))
      })
      .finally(() => {
        if (active) setUsageLoading(false)
      })
    return () => {
      active = false
    }
  }, [isAdmin])

  useEffect(() => {
    if (!isAdmin) return undefined
    let active = true
    setApprovalsLoading(true)
    setApprovalsError(null)
    listApprovals({ status: 'pending' })
      .then((data) => {
        if (active) setApprovals(data.approvals ?? [])
      })
      .catch((error) => {
        if (active) setApprovalsError(getErrorMessage(error, 'Could not load the approval queue.'))
      })
      .finally(() => {
        if (active) setApprovalsLoading(false)
      })
    return () => {
      active = false
    }
  }, [isAdmin])

  if (!isAdmin) return <Navigate to="/" replace />

  const maxUsageCount = usage.length > 0 ? Math.max(...usage.map((row) => row.request_count)) : 1

  // Per-day rows each carry their own avg_latency_ms; the headline figure
  // should be a request-weighted average across all days, not just the
  // newest row's value.
  const weightedAvgLatencyMs =
    usage.length === 0
      ? 0
      : Math.round(
          usage.reduce((total, row) => total + (row.request_count ?? 0) * (row.avg_latency_ms ?? 0), 0) /
            Math.max(1, usage.reduce((total, row) => total + (row.request_count ?? 0), 0))
        )

  const handleResolveApproval = async (approvalId, approved) => {
    setResolvingId(approvalId)
    try {
      await resolveApproval(approvalId, approved)
      setApprovals((current) => current.filter((approval) => approval.approval_id !== approvalId))
      showToast(approved ? 'Approved.' : 'Rejected.', 'success')
    } catch (error) {
      showToast(getErrorMessage(error, 'Could not resolve the approval. Please try again.'), 'error')
    } finally {
      setResolvingId(null)
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-5 py-4">
      <div>
        <h2 className="font-display text-xl font-bold">Admin</h2>
        <p className="text-sm text-slate-500 dark:text-ink-muted">Usage analytics and the human approval queue.</p>
      </div>

      <Card padding="lg">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-ink-secondary">
          <BarChart3 size={15} strokeWidth={1.5} />
          Usage analytics
        </h3>
        {usageLoading ? (
          <div className="space-y-2.5">
            {Array.from({ length: 5 }).map((_, index) => (
              <div key={index} className="flex items-center gap-2">
                <Skeleton className="h-3 w-20 shrink-0" />
                <Skeleton className="h-2 flex-1" />
                <Skeleton className="h-3 w-8 shrink-0" />
              </div>
            ))}
          </div>
        ) : usageError ? (
          <p className="text-sm text-danger">{usageError}</p>
        ) : usage.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-ink-muted">
            No usage data yet — this is populated as requests come in.
          </p>
        ) : (
          <div className="space-y-2">
            {usage.map((row) => (
              <div key={row.day} className="flex items-center gap-2 text-xs">
                <span className="w-20 shrink-0 text-slate-500 dark:text-ink-muted">{row.day}</span>
                <div className="h-2 flex-1 rounded bg-slate-900/5 dark:bg-white/[0.05]">
                  <div
                    className="h-2 rounded bg-accent-500"
                    style={{ width: `${Math.min(100, (row.request_count / maxUsageCount) * 100)}%` }}
                  />
                </div>
                <span className="w-10 shrink-0 text-right text-slate-500 dark:text-ink-muted">
                  {row.request_count}
                </span>
              </div>
            ))}
            <p className="pt-1 text-[11px] text-slate-400 dark:text-ink-muted">
              Requests per day · avg latency {weightedAvgLatencyMs} ms
            </p>
          </div>
        )}
      </Card>

      <Card padding="lg">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-ink-secondary">
          <ShieldCheck size={15} strokeWidth={1.5} />
          Approval queue
        </h3>
        {approvalsLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 2 }).map((_, index) => (
              <div key={index} className="rounded-lg border border-border-light px-3.5 py-3 dark:border-border">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="mt-2 h-4 w-3/4" />
                <Skeleton className="mt-2 h-3 w-1/2" />
              </div>
            ))}
          </div>
        ) : approvalsError ? (
          <p className="text-sm text-danger">{approvalsError}</p>
        ) : approvals.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-ink-muted">
            No pending approvals — nothing is waiting on web_search_requires_approval or
            document_delete_requires_approval right now.
          </p>
        ) : (
          <div className="space-y-3">
            {approvals.map((approval) => {
              const isResolving = resolvingId === approval.approval_id
              const disabled = resolvingId != null && !isResolving
              return (
                <div
                  key={approval.approval_id}
                  className="rounded-lg border border-border-light px-3.5 py-3 dark:border-border"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-400 dark:text-ink-muted">
                        {approval.action === 'web_search' ? 'Web search' : 'Document deletion'}
                      </p>
                      <p className="mt-0.5 truncate text-sm text-slate-700 dark:text-ink-secondary">
                        {approvalSummary(approval)}
                      </p>
                      <p className="mt-0.5 text-[11px] text-slate-400 dark:text-ink-muted">
                        Requested by {approval.requested_by ?? 'unknown'} ·{' '}
                        {new Date(approval.created_at * 1000).toLocaleString()}
                      </p>
                    </div>
                    <div className="flex shrink-0 gap-1.5">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => handleResolveApproval(approval.approval_id, false)}
                        loading={isResolving}
                        disabled={disabled}
                      >
                        Reject
                      </Button>
                      <Button
                        size="sm"
                        variant="primary"
                        onClick={() => handleResolveApproval(approval.approval_id, true)}
                        loading={isResolving}
                        disabled={disabled}
                      >
                        Approve
                      </Button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </Card>
    </div>
  )
}
