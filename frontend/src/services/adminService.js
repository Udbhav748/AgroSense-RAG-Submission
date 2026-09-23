import api from './api'

/**
 * Admin-only usage analytics (GET /admin/usage-summary) — daily request
 * counts and average latencies, newest first. The backend 403s this for
 * any non-admin role, so callers should gate the UI on user.role === 'admin'.
 * @returns {Promise<{rows: {day: string, request_count: number, avg_latency_ms: number}[]}>}
 */
export async function getUsageSummary() {
  const { data } = await api.get('/admin/usage-summary')
  return data
}

/**
 * Recent feedback events (GET /feedback) — thumbs up/down, comments, and
 * rubric reviews captured by the chat thumbs control. For admins the
 * backend returns everything (or a filtered reviewer's); for members,
 * only their own. @returns {Promise<{events: FeedbackEvent[], count: number}>}
 */
export async function getFeedback(limit = 20) {
  const { data } = await api.get('/feedback', { params: { limit } })
  return data
}

/**
 * Admin-only human approval queue (GET /approvals) — gated actions (web
 * search, document deletion) that were requested while their deployment's
 * *_requires_approval setting was on and the caller didn't send the
 * matching confirm/approved flag. The backend 403s this for any non-admin
 * role, same as getUsageSummary above.
 * @param {{ status?: 'pending'|'approved'|'rejected', action?: 'web_search'|'document_delete' }} [filters]
 * @returns {Promise<{approvals: {approval_id: string, action: string, requested_by: string|null, payload: object, status: string, note: string|null, created_at: number, resolved_at: number|null, resolved_by: string|null}[], count: number}>}
 */
export async function listApprovals(filters = {}) {
  const { data } = await api.get('/approvals', { params: filters })
  return data
}

/**
 * Grant or deny a pending approval (POST /approvals/{id}/resolve).
 * @param {string} approvalId
 * @param {boolean} approved
 * @param {string} [note] Optional reason, recorded in the audit log.
 */
export async function resolveApproval(approvalId, approved, note) {
  const { data } = await api.post(`/approvals/${encodeURIComponent(approvalId)}/resolve`, { approved, note })
  return data
}