import api from './api'

/**
 * Record a thumbs up/down (and optional comment and/or detailed 1-5
 * rubric) on a chat answer. The rubric is additive — a plain thumbs up/
 * down remains a complete, valid submission on its own.
 * @param {string} messageId - The frontend's own id for the assistant message.
 * @param {'up'|'down'} rating
 * @param {string|null} [comment]
 * @param {{correctness:number, helpfulness:number, completeness:number, safety:number, tone:number, groundedness:number, citation_quality:number}|null} [rubric]
 * @returns {Promise<{status: string}>}
 */
export async function sendFeedback(messageId, rating, comment = null, rubric = null) {
  const { data } = await api.post('/chat/feedback', {
    message_id: messageId,
    rating,
    comment,
    rubric,
  })
  return data
}
