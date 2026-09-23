/**
 * Extract a human-readable message from an axios error, matching the
 * backend's error shapes: {"detail": "message"} from AppError, or
 * {"detail": [{"msg": "..."}]} from FastAPI's request validation.
 * New format: {"detail": "message", "error_code": "CODE", "taxonomy_category": "cat"}
 */
export default function getErrorMessage(error, fallback = 'Something went wrong. Please try again.') {
  const data = error?.response?.data

  // LeafSense (the external vision service) is unreachable/down — a raw
  // "Could not reach LeafSense at http://localhost:8001: ..." string with
  // the error code tacked on is technically true but useless to a user.
  // This error_code is only ever produced by the diagnose endpoint, so
  // handling it here can't shadow any other flow's message.
  if (data?.error_code === 'VISION_SERVICE_ERROR') {
    return 'The plant diagnosis service isn\'t running right now. Please try again in a moment.'
  }

  // New structured error format: {detail, error_code, taxonomy_category}
  if (data?.detail && typeof data.detail === 'string') {
    const parts = [data.detail]
    if (data.error_code) parts.push(`[${data.error_code}]`)
    if (data.taxonomy_category) parts.push(`(${data.taxonomy_category})`)
    return parts.join(' ')
  }

  // FastAPI validation error format
  const detail = data?.detail
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg

  // Network/timeout errors
  if (error?.code === 'ECONNABORTED') return 'The request timed out. Please try again.'
  if (error?.message === 'Network Error') return 'Unable to reach the server. Please check your connection.'

  return fallback
}

/**
 * Extract structured error info for programmatic handling.
 * Returns { error_code, taxonomy_category, detail } or null if not available.
 */
export function getErrorInfo(error) {
  const data = error?.response?.data
  if (data?.error_code) {
    return {
      error_code: data.error_code,
      taxonomy_category: data.taxonomy_category,
      detail: data.detail,
    }
  }
  return null
}
